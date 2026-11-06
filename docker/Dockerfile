# syntax=docker/dockerfile:1
# Targets: `runtime` (default; API + UI) and `eval` (adds RAGAS and the datasets).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FORWARDED_ALLOW_IPS=*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /service

FROM base AS deps
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM deps AS deps-eval
COPY evals/requirements.txt evals/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r evals/requirements.txt

FROM base AS app
COPY src src
COPY configs configs
COPY app app
COPY .streamlit .streamlit
COPY assets/corpus assets/corpus
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
USER appuser
ENTRYPOINT ["entrypoint.sh"]

FROM app AS eval
COPY --from=deps-eval /opt/venv /opt/venv
COPY evals evals
CMD ["python", "-m", "src.eval.run_evaluation"]

# Last stage, so it is the default build target.
FROM app AS runtime
COPY --from=deps /opt/venv /opt/venv
# SERVICE selects the process: api | ui | all.
ENV SERVICE=api
EXPOSE 8000 8501
