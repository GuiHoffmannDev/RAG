from src.eval.report import (
    breakdown,
    breakdown_table,
    outcome,
    rewrite_causes,
    variants_table,
)


def grade(evidence, valid=True):
    return {"node": "grade_documents", "evidence": evidence, "grading_valid": valid}


def test_outcome_covers_all_four_cases():
    assert outcome(True, False) == "answered"
    assert outcome(True, True) == "false_abstention"
    assert outcome(False, True) == "correct_abstention"
    assert outcome(False, False) == "hallucination"


def test_rewrite_causes_are_read_from_steps():
    rewrite = {"node": "rewrite_query"}
    retrieve = {"node": "retrieve"}
    steps = [
        retrieve,
        grade([], valid=False),
        rewrite,
        retrieve,
        grade([]),
        rewrite,
        retrieve,
        grade(["p"]),
        rewrite,
        retrieve,
        grade(["p"]),
        {"node": "generate_answer"},
    ]
    assert rewrite_causes(steps) == ["invalid_grading", "zero_relevant", "below_threshold"]
    # Traces written before grading_valid existed still classify.
    assert rewrite_causes([{"node": "grade_documents", "evidence": []}, rewrite]) == [
        "zero_relevant"
    ]


def rows():
    def row(system, out, causes=()):
        return {"system": system, "outcome": out, "rewrite_causes": list(causes)}

    return [
        row("baseline", "hallucination"),
        row("baseline", "answered"),
        row("corrective", "correct_abstention", ["zero_relevant"]),
        row("corrective", "false_abstention", ["below_threshold", "below_threshold"]),
        row("corrective", None),
    ]


def test_breakdown_counts_and_ignores_failed_cases():
    info = breakdown(rows())
    assert info["baseline"] == {
        "unanswerable": 1,
        "correct_abstention": 0,
        "hallucination": 1,
        "answerable": 1,
        "false_abstention": 0,
    }
    assert info["corrective"]["correct_abstention"] == 1
    assert info["corrective"]["answerable"] == 1
    assert info["rewrite_causes"] == {"below_threshold": 2, "zero_relevant": 1}
    text = breakdown_table(info)
    assert "0/1 (0%)" in text and "1/1 (100%)" in text and "below_threshold: 2" in text


def test_variants_table_has_baseline_then_one_row_per_run():
    def report(label, rewritten):
        case = {
            "system": "corrective",
            "latency_s": 2.0,
            "answer": {"rewrites": int(rewritten)},
        }
        base = {"system": "baseline", "latency_s": 1.0, "answer": {"rewrites": 0}}
        metrics = {
            m: {"baseline": 0.5, "corrective": 0.75}
            for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        }
        return {
            "label": label,
            "cases": [base, case],
            "summary": metrics,
            "breakdown": breakdown(rows()),
        }

    lines = variants_table([report("majority", True), report("any", False)]).splitlines()
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| baseline", "| majority", "| any"]
    assert "1/1" in lines[3] and "0/1" in lines[4]
