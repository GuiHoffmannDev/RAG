| Metric | Baseline RAG | Agentic Corrective RAG | Delta | Pairs |
| --- | ---: | ---: | ---: | ---: |
| faithfulness | 0.9062 | 0.8646 | -0.0417 | 16 |
| answer_relevancy | 0.8085 | 0.7911 | -0.0174 | 16 |
| context_precision | 0.9583 | 0.9687 | 0.0104 | 16 |
| context_recall | 0.9375 | 0.9219 | -0.0156 | 16 |
| answer_coverage | 1.0000 | 1.0000 | 0.0000 | 16 |

| Outcome | Baseline RAG | Agentic Corrective RAG |
| --- | ---: | ---: |
| Correct abstention (unanswerable) | 4/10 (40%) | 3/8 (38%) |
| Hallucination (unanswerable) | 6/10 (60%) | 5/8 (62%) |
| False abstention (answerable) | 0/16 (0%) | 0/16 (0%) |

Corrective rewrite causes: below_threshold: 38, zero_relevant: 7
