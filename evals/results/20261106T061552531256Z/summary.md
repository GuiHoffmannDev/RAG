| Metric | Baseline RAG | Agentic Corrective RAG | Delta | Pairs |
| --- | ---: | ---: | ---: | ---: |
| faithfulness | 0.9000 | 0.7556 | -0.1444 | 15 |
| answer_relevancy | 0.8085 | 0.7349 | -0.0736 | 16 |
| context_precision | 0.9583 | 0.9062 | -0.0521 | 16 |
| context_recall | 0.9375 | 0.8698 | -0.0677 | 16 |
| answer_coverage | 1.0000 | 0.9375 | -0.0625 | 16 |

| Outcome | Baseline RAG | Agentic Corrective RAG |
| --- | ---: | ---: |
| Correct abstention (unanswerable) | 4/10 (40%) | 4/10 (40%) |
| Hallucination (unanswerable) | 6/10 (60%) | 6/10 (60%) |
| False abstention (answerable) | 0/16 (0%) | 1/16 (6%) |

Corrective rewrite causes: invalid_grading: 2, zero_relevant: 6
