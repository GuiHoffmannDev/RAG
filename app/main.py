import os
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = {
    "Código de erro": "O que significa ARC-17 e como corrigir o erro?",
    "Recuperação": "Por quanto tempo posso recuperar um arquivo excluído?",
    "Fora do runbook": "Qual é o preço da assinatura empresarial?",
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_settings(url):
    response = httpx.get(f"{url}/settings", timeout=3)
    response.raise_for_status()
    return response.json()


def sidebar():
    st.header("Configurações")
    options = {}
    try:
        defaults = fetch_settings(API_URL)
        st.caption(f"Provedor · {defaults['provider']}")
        options["model"] = st.selectbox("Modelo", defaults["models"])
        options["temperature"] = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
        options["max_rewrites"] = st.slider(
            "Máximo de reescritas",
            0,
            3,
            defaults["max_rewrites"],
            help="Cada reescrita permite uma nova busca. Zero mantém apenas a busca inicial.",
        )
    except (httpx.HTTPError, ValueError, KeyError):
        st.caption("Configurações indisponíveis. A consulta usará os padrões da API.")
    debug = st.toggle(
        "Modo debug", help="Exibe o estado final do LangGraph para a consulta selecionada."
    )
    st.divider()
    return options, debug


def use_example(question):
    st.session_state.question = question


def composer():
    with st.form("ask"):
        question = st.text_area(
            "Sua pergunta",
            key="question",
            max_chars=2000,
            height=90,
            placeholder="O que você precisa saber sobre o serviço de arquivos?",
        )
        submitted = st.form_submit_button(
            "Consultar runbook", type="primary", icon=":material/search:"
        )
    st.caption("Experimente uma pergunta")
    for column, (label, example) in zip(st.columns(3), EXAMPLES.items(), strict=True):
        column.button(example, help=label, on_click=use_example, args=(example,), width="stretch")
    return question.strip() if submitted else None


def submit(question, options):
    if not question:
        st.warning("Escreva uma pergunta ou escolha um exemplo.")
        return
    try:
        with st.spinner("Buscando documentos e avaliando as evidências…", show_time=True):
            response = httpx.post(
                f"{API_URL}/query",
                json={"question": question, "inspect_run": True, **options},
                timeout=600,
            )
            response.raise_for_status()
            answer = response.json()
            if (
                not isinstance(answer, dict)
                or not {"answer", "abstained", "citations", "rewrites"} <= answer.keys()
            ):
                raise ValueError("Invalid query response")
    except httpx.TimeoutException:
        st.error("A consulta excedeu o tempo de espera. Tente uma pergunta mais específica.")
        return
    except httpx.HTTPStatusError as exc:
        st.error(
            "A API recusou a consulta. Revise a pergunta e as configurações."
            if exc.response.status_code == 422
            else "Não foi possível consultar o runbook. Verifique o índice e o provedor do modelo."
        )
        return
    except (httpx.HTTPError, ValueError):
        st.error("Não foi possível ler a resposta da API. Verifique a conexão e tente novamente.")
        return
    entry = {"id": uuid4().hex, "question": question, "answer": answer, "options": options}
    st.session_state.history = [entry, *st.session_state.history[:11]]
    st.session_state.selected = entry["id"]


def show_history():
    history = st.session_state.history
    st.sidebar.subheader("Histórico")
    if not history:
        st.sidebar.caption("Suas consultas aparecerão aqui.")
        return None
    entries = {entry["id"]: entry for entry in history}
    active = st.sidebar.selectbox(
        "Consultas desta sessão",
        list(entries),
        key="selected",
        format_func=lambda key: entries[key]["question"],
    )
    return entries[active]


def show_metrics(metrics):
    latency, tokens, cost = st.columns(3)
    seconds, total, usd = (
        metrics.get(k) for k in ("latency_s", "total_tokens", "estimated_cost_usd")
    )
    latency.metric(
        "Latência total",
        f"{seconds:.2f} s" if seconds is not None else "—",
        help="Tempo total de execução do grafo na API.",
    )
    tokens.metric("Tokens usados", total if total is not None else "—", help="Chat e embeddings.")
    cost.metric(
        "Custo estimado",
        f"${usd:.6f}" if usd is not None else "—",
        help="USD em tarifas do provedor. Um traço indica dados indisponíveis.",
    )


def send_vote(entry, key):
    vote = st.session_state[key]
    if vote is None:
        return
    answer = entry["answer"]
    try:
        response = httpx.post(
            f"{API_URL}/feedback",
            json={
                "trace_id": answer["observability"]["trace_id"],
                "feedback_token": answer["feedback_token"],
                "positive": bool(vote),
            },
            timeout=10,
        )
        response.raise_for_status()
        entry["feedback"] = vote
        entry.pop("failed_vote", None)
    except httpx.HTTPError:
        entry["failed_vote"] = vote


def show_feedback(entry, metrics):
    answer = entry["answer"]
    if metrics.get("trace_url"):
        st.link_button(
            "Abrir trace no Langfuse", metrics["trace_url"], icon=":material/open_in_new:"
        )
    if metrics.get("trace_id") and answer.get("feedback_token"):
        key = f"vote-{entry['id']}"
        if key not in st.session_state:
            st.session_state[key] = entry.get("failed_vote", entry.get("feedback"))
        st.caption("A resposta foi útil?")
        st.feedback("thumbs", key=key, on_change=send_vote, args=(entry, key))
        if "failed_vote" in entry:
            st.warning("Não foi possível enviar o feedback.")
            st.button("Reenviar avaliação", on_click=send_vote, args=(entry, key))
        elif "feedback" in entry:
            st.caption("Avaliação enviada ao Langfuse.")
    else:
        st.caption("Feedback indisponível para esta consulta sem trace ativo.")


def show_answer(entry):
    answer = entry["answer"]
    with st.container(border=True):
        st.subheader("Resposta final")
        st.caption(entry["question"])
        if options := entry["options"]:
            st.caption(
                f"{options['model']} · Temperatura {options['temperature']:g} · Limite de reescritas {options['max_rewrites']}"
            )
        if answer["abstained"]:
            st.warning("Os trechos disponíveis não sustentam uma resposta completa.")
        else:
            st.text(answer["answer"])
        if answer["citations"]:
            st.markdown("**Fontes citadas**")
            for citation in answer["citations"]:
                with st.expander(
                    f"[{citation['label']}] {citation['source']} · trecho {citation['chunk']}"
                ):
                    st.text(citation["text"])
        st.divider()
        metrics = answer.get("observability") or {}
        with st.container(key="run_metrics"):
            show_metrics(metrics)
        show_feedback(entry, metrics)


def show_attempts(entry):
    answer = entry["answer"]
    inspection = answer.get("inspection")
    st.subheader("Percurso do agente")
    if not inspection:
        st.info("O servidor não retornou as etapas desta consulta.")
        return
    steps = inspection["steps"]
    attempts = sum(step["node"] == "retrieve" for step in steps)
    rewrites = answer["rewrites"]
    st.caption(f"Buscas realizadas · {attempts}　/　Reescritas · {rewrites}")
    query = inspection["state"]["question"]
    number = 0
    for step in steps:
        node = step["node"]
        if node == "rewrite_query":
            st.caption("O grading não reuniu evidências suficientes. O agente reformulou a busca.")
            query = step["query"]
        elif node == "retrieve":
            number += 1
            candidates = step["candidates"]
            st.markdown(f"**{number}. {'Busca inicial' if number == 1 else 'Busca reescrita'}**")
            st.text(query)
        elif node == "grade_documents":
            accepted = {(p["source"], p["chunk"]) for p in step["evidence"]}
            valid = step.get("grading_valid", True)
            if not candidates:
                st.caption("Nenhum trecho recuperado.")
                continue
            if not valid:
                st.warning(
                    "O avaliador retornou um grading inválido. Todos os trechos foram descartados."
                )
            with st.expander(
                f"Trechos aceitos · {len(accepted)} de {len(candidates)}",
                expanded=number == attempts,
            ):
                for passage in candidates:
                    source = (passage["source"], passage["chunk"])
                    verdict = (
                        "Aceito" if source in accepted else "Rejeitado" if valid else "Descartado"
                    )
                    st.text(f"{verdict} · {source[0]} · trecho {source[1]}")
        elif node in ("generate_answer", "abstain"):
            st.caption(
                "Sem resposta sustentada pelas fontes."
                if answer["abstained"]
                else "O agente gerou a resposta com os trechos aceitos."
            )


def main():
    st.set_page_config(
        page_title="Runbook · Agentic Corrective RAG",
        page_icon=":material/account_tree:",
        layout="wide",
    )
    st.session_state.setdefault("history", [])
    with st.sidebar:
        options, debug = sidebar()
    st.html(Path(__file__).with_name("banner.html"))
    question = composer()
    if question is not None:
        submit(question, options)
    entry = show_history()
    if entry:
        st.divider()
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            show_answer(entry)
        with right:
            show_attempts(entry)
        if debug and (inspection := entry["answer"].get("inspection")):
            with st.expander("Debug · estado final do LangGraph", expanded=True):
                st.json(inspection["state"], expanded=1)


main()
