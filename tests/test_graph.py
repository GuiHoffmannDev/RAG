from unittest.mock import Mock

import pytest

from src.agents.graph import Claim, Draft, build_graph, render
from tests.conftest import PASSAGE, FakeChat


def run_state(grades, budget=1, hits=(PASSAGE,), **options):
    chat = FakeChat(grades)
    search = Mock(return_value=list(hits))
    state = build_graph(search, chat.model, budget, **options).invoke(
        {"question": "What is ARC-17?", "query": "What is ARC-17?", "rewrites": 0}
    )
    return state, [c.args[0] for c in search.call_args_list], chat.prompts


def run(grades, budget=1, hits=(PASSAGE,), **options):
    state, queries, prompts = run_state(grades, budget, hits, **options)
    return state["result"], queries, prompts


def test_rewrite_preserves_original_question_and_returns_evidence():
    answer, queries, prompts = run([[(1, False)], [(1, True)]])
    assert queries == ["What is ARC-17?", "archive checksum"]
    assert "What is ARC-17?" in prompts[-1][-1][1]
    assert answer.rewrites == 1 and not answer.abstained
    assert answer.answer.endswith("[1]") and answer.citations[0].text == PASSAGE.text


@pytest.mark.parametrize(
    "grades,budget,hits,count",
    [
        ([[(1, False)], [(1, False)]], 1, [PASSAGE], 2),
        ([[(99, True)]], 0, [PASSAGE], 1),
        ([], 3, [], 4),
    ],
)
def test_abstention_is_bounded(grades, budget, hits, count):
    answer, queries, _ = run(grades, budget, hits)
    assert len(queries) == count
    assert answer.abstained and not answer.citations


def test_direct_answer_skips_rewrite():
    answer, queries, _ = run([[(1, True)]])
    assert len(queries) == 1 and answer.rewrites == 0 and not answer.abstained


@pytest.mark.parametrize("ids", [[], [0], [2], [1, 3]])
def test_invalid_claim_abstains(ids):
    answer = render(Draft(claims=[Claim(text="Claim.", passage_ids=ids)]), [PASSAGE], 0)
    assert answer.abstained and answer.citations == []


def test_irrelevant_majority_rewrites_and_only_cites_approved_hits():
    hits = [PASSAGE.model_copy(update={"chunk": i, "score": 1 / i}) for i in (1, 2, 3)]
    minority = [(1, False), (3, True), (2, False)]
    majority = [(3, True), (1, False), (2, True)]
    answer, queries, _ = run([minority, majority], hits=hits)
    assert len(queries) == 2 and not answer.abstained
    assert [(c.chunk, c.score) for c in answer.citations] == [(2, 0.5)]


def test_tie_generates_and_budget_exhaustion_can_use_one_approved_passage():
    tie = [(1, False), (2, True)]
    direct, queries, _ = run([tie], hits=[PASSAGE] * 2)
    assert len(queries) == 1 and not direct.abstained
    minority = [*tie, (3, False)]
    final, queries, _ = run([minority] * 3, budget=2, hits=[PASSAGE] * 3)
    assert len(queries) == 3 and final.rewrites == 2 and not final.abstained


@pytest.mark.parametrize("invalid", [[(1, True)], [(1, True), (1, False)]])
def test_incomplete_or_duplicate_grades_never_authorize_generation(invalid):
    answer, _, _ = run([invalid], budget=0, hits=[PASSAGE] * 2)
    assert answer.abstained


def test_grader_verdicts_are_kept_in_state_even_when_invalid():
    state, _, _ = run_state([[(1, True), (2, False)]], hits=[PASSAGE] * 2)
    assert [(v.passage_id, v.relevant) for v in state["verdicts"]] == [(1, True), (2, False)]
    state, _, _ = run_state([[(1, True)]], budget=0, hits=[PASSAGE] * 2)
    assert state["grading_valid"] is False and len(state["verdicts"]) == 1


def test_any_rule_accepts_a_single_approved_passage_without_rewriting():
    minority = [(1, True), (2, False), (3, False)]
    hits = [PASSAGE] * 3
    answer, queries, _ = run([minority] * 2, hits=hits)
    assert len(queries) == 2 and answer.rewrites == 1  # majority rule rewrites first
    answer, queries, _ = run([minority], hits=hits, grade_rule="any")
    assert len(queries) == 1 and answer.rewrites == 0 and not answer.abstained
    answer, queries, _ = run([[(1, False)], [(1, False)]], hits=[PASSAGE], grade_rule="any")
    assert answer.abstained and len(queries) == 2


def test_fewshot_prompt_is_opt_in():
    _, _, plain = run([[(1, True)]])
    _, _, fewshot = run([[(1, True)]], grader_prompt="fewshot")
    assert "Example 1" not in plain[0][0][1] and "Example 1" in fewshot[0][0][1]


def test_separate_grader_model_grades_while_main_model_rewrites_and_generates():
    grader = FakeChat([[(1, False)], [(1, True)]])
    main = FakeChat([])
    search = Mock(return_value=[PASSAGE])
    state = build_graph(search, main.model, 1, grader_model=grader.model).invoke(
        {"question": "What is ARC-17?", "query": "What is ARC-17?", "rewrites": 0}
    )
    assert len(grader.prompts) == 2 and not state["result"].abstained
