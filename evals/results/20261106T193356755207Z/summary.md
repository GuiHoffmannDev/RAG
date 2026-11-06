| Metric | Baseline RAG | Agentic Corrective RAG | Delta | Pairs |
| --- | ---: | ---: | ---: | ---: |
| faithfulness | 0.8750 | 0.8229 | -0.0521 | 16 |
| answer_relevancy | 0.8032 | 0.7975 | -0.0057 | 16 |
| context_precision | 0.9583 | 0.9583 | -0.0000 | 16 |
| context_recall | 0.9375 | 0.9635 | 0.0260 | 16 |
| answer_coverage | 1.0000 | 1.0000 | 0.0000 | 16 |

| Outcome | Baseline RAG | Agentic Corrective RAG |
| --- | ---: | ---: |
| Correct abstention (unanswerable) | 4/10 (40%) | 3/9 (33%) |
| Hallucination (unanswerable) | 6/10 (60%) | 6/9 (67%) |
| False abstention (answerable) | 0/16 (0%) | 0/16 (0%) |

Corrective rewrite causes: zero_relevant: 5
