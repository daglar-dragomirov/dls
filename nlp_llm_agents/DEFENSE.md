# Five-minute defense

1. Open `/` and point to the baseline, agent, LLM-audit, NDCG, and search-use cards.
2. Run the default jazz-bar example and show the `search_evidence` trace.
3. Load the hard negative and show that explicit `no_pets` evidence overrides a superficially relevant cafe card.
4. Open `reports/final_report.md`: compare baseline, no-search ablation, and full agent on the same 500 rows.
5. Mention the independent 50-case Instructor/OpenRouter audit and the private-data adapter.

Key conclusion: search adds information unavailable to the baseline, and the measured gain survives a real structured LLM review with a guardrail for disagreements.
