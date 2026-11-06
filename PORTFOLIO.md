# Portfolio checklist and roadmap

## Done

- [x] Hybrid retrieval (dense + BM25, reciprocal rank fusion) with atomic alias reindexing
- [x] LangGraph corrective loop: grading, bounded rewrites, abstention, validated citations
- [x] Paired RAGAS evaluation against a baseline, with pinned snapshot and dataset hash
- [x] Langfuse tracing, token and cost accounting, user feedback scores
- [x] Streamlit workspace and FastAPI service with health and readiness endpoints
- [x] Multi-stage Docker image, Compose profiles (dev, proxy, eval), CI with ruff and pytest
- [x] MIT license, `.gitignore`, pinned dependencies, honest limitations in the README

## Before showing this in an interview

- [ ] Replace the two placeholder SVGs in `assets/` with real screenshots (UI and one Langfuse trace) and update the image links in `README.md`
- [ ] Commit the repository (it has no commits yet) and push; confirm the CI badge URL in `README.md` matches the real GitHub path
- [ ] Confirm the copyright holder name in `LICENSE`
- [x] Re-run the evaluation after the code changes and update the results table (done 2026-11-06)

## Improvements, in suggested order

1. ~~**Explain the negative result.**~~ Done: per-passage verdicts are logged, the `majority` rule caused the rewrites, and `any` plus few-shot variants are in the results table. The Qwen3 8B grader (`GRADER_MODEL`) cut rewrites further but did not improve faithfulness or abstention.
2. ~~**Score unanswerable questions.**~~ Done: 10 unanswerable cases in the golden set with abstention metrics. Result: no gain yet (6 of 10 hallucinated in both systems). Next: make the grader stricter about near-miss passages, or add a reranker score threshold.
3. **Larger, realistic corpus.** Add a public documentation set (hundreds of pages) and 100+ questions, so differences are larger than noise. Report confidence intervals.
4. **Reranker.** Add a cross-encoder rerank after fusion and measure it as its own ablation.
5. **Stronger judge.** Repeat the evaluation with a hosted judge to separate model-family bias from real quality differences.
6. **Better chunking.** Structure-aware splitting for Markdown headings and tables; PDF and HTML loaders.
7. **Latency.** Run the grader on a small model, cache grading per passage, and stream tokens to the UI.
8. **Security for public hosting.** Authentication, rate limits, per-tenant collections, and PII handling for traces.
9. **Deployment proof.** Deploy once to Render, Railway or Fly.io, verify `/ready` with Qdrant Cloud, and add the live link and notes to the README.
10. ~~**Test depth.**~~ Done: coverage gate (75%), `/query` schema contract snapshot, and an end-to-end test with a fake provider.
11. **UI.** English/Portuguese language switch.
