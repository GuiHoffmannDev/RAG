| Metric | Baseline RAG | Agentic Corrective RAG | Delta | Pairs |
| --- | ---: | ---: | ---: | ---: |
| faithfulness | 0.9062 | 0.8021 | -0.1042 | 16 |
| answer_relevancy | 0.8053 | 0.7919 | -0.0134 | 16 |
| context_precision | 0.9583 | 0.9687 | 0.0104 | 16 |
| context_recall | 0.9375 | 0.9219 | -0.0156 | 16 |
| answer_coverage | 1.0000 | 1.0000 | 0.0000 | 16 |

| Outcome | Baseline RAG | Agentic Corrective RAG |
| --- | ---: | ---: |
| Correct abstention (unanswerable) | 4/10 (40%) | 4/10 (40%) |
| Hallucination (unanswerable) | 6/10 (60%) | 6/10 (60%) |
| False abstention (answerable) | 0/16 (0%) | 0/16 (0%) |

Corrective rewrite causes: invalid_grading: 1, zero_relevant: 8
