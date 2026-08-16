# Phase 8.2 — Retrieval Cascade: Results & Analysis

**Setup:** 80 hand-labeled postings (`eval/golden_set.json`, labels 0/1/2). 4 variants scored with `python -m eval.ranking_eval -n 80`. Metrics: precision@10, nDCG@10, MRR. Full raw output: `ranking_results.json` in this directory.

## Results

| Variant | Precision@10 | nDCG@10 | MRR | Latency | Tokens |
|---|---|---|---|---|---|
| V0 — BM25 (keyword baseline) | 0.60 | 0.439 | 0.5 | 0.02s | 0 |
| **V1 — cosine (bi-encoder)** | **1.00** | **0.850** | 1.0 | 3.3s | 0 |
| V2 — cosine → cross-encoder | 0.90 | 0.757 | 1.0 | 10.9s | 0 |
| V3 — LLM-only (Gemini) | 0.80 | 0.680 | 1.0 | 69.4s | 20,793 |

## Headline finding

V1 (plain cosine) wins on every quality metric. It also costs the least, aside from BM25. Adding the cross-encoder (V2) or replacing retrieval with an LLM call (V3) made results worse, not better, and both cost much more time.

## Why V1 wins here

- **Your labels track keyword and topic overlap.** You labeled postings by reading title, company, and description. Cosine similarity measures the same thing: how much a posting's text overlaps with your profile's target-roles and skills text. It's a good match for how you're actually judging fit.
- **The cross-encoder isn't trained for this task.** `ms-marco-MiniLM-L-6-v2` (Phase 2.6) was trained for general search relevance, not job matching. Its whole point is catching things cosine misses, like "8+ years required" against a 4-year profile. If your 80 postings don't have many of those tricky cases, it has nothing to fix — it just adds latency.
- **V3 is expensive and didn't help.** One Gemini call to rank everything cost 20.8K tokens and over a minute, and still scored below plain cosine.

## What's different from the earlier 40-sample run

- V0 (BM25) dropped a lot: precision 0.90 → 0.60, MRR 1.0 → 0.5. At 80 postings its weak keyword matching struggles more.
- V1 stayed identical: precision 1.00, nDCG 0.850. Cosine held up well at double the sample size.
- V2 and V3 both dropped slightly too.
- V1's lead over V2/V3 got *more* visible with more data, not less. That's a useful signal — it's not just a fluke of a small sample.

## Caveats

- n = 80 is still small for statistics, but it's a real improvement over 40, and the pattern held. That's a good sign the result is real, not noise.
- This is about ranking quality, not the cascade's real job. The cascade (Phases 2.5-2.6) exists to cut postings down to a size Gemini can afford to reason over, not to guarantee the best possible order. V1 winning here doesn't mean the cascade's filtering step is wrong — it means the extra reranking stage isn't earning its cost.
- The cross-encoder's value is in hard edge cases. This golden set may just not have many of those.

## Takeaway

Cosine similarity alone does most of the work here. The fancier stages (cross-encoder, LLM) cost more and perform worse on this data. That's a real, useful finding for the closing project: "we added a reranker, measured it, and it didn't help — here's the likely reason why" is exactly the kind of honest, quantified result Phase 8 is meant to produce.
