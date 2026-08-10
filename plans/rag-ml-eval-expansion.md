# RAG / ML / Evaluation Expansion — Study Project Plan

## Why this document exists

The project started as "build an agent that hosts multiple MCP servers." The goal has broadened: it should now demonstrate the range of topics covered in coursework — vector DB, RAG, NLP, deep learning, fine-tuning, classical ML, evaluation, agents, MCP — as a closing study project.

This plan adds those topics **without restructuring what already works**. The LangGraph agent loop, the JobSpy MCP server, and the Sheets logging all stay exactly as designed in `job-application-copilot-guide.md`. Everything below either sits *in front of* the LLM (narrowing what it sees) or *beside* it (measuring what it does).

**Scope discipline:** each phase is marked Core or Optional. Core is the closing project. Optional phases are additive slides if time allows. Nothing here needs accumulated usage history — the project is a demonstration, not a system that has been running for months.

---

## Topic coverage map

| Course topic | Where it lives | Status |
|---|---|---|
| LLM | Gemini via `langchain-google-genai` | already built |
| Agent | LangGraph loop, guide Phase 5 | already planned |
| MCP | JobSpy server + Sheets tools | already built |
| Vector DB | Chroma, two collections | Phase A |
| NLP | chunking, embeddings, semantic dedup | Phase A |
| Deep learning | cross-encoder reranker, local transformer inference | Phase B |
| RAG | grounded cover-letter drafting | Phase C |
| Evaluation (IR) | precision@k / nDCG over golden set | Phase D |
| Evaluation (LLM) | groundedness via LLM-as-judge | Phase D |
| Classical ML — unsupervised | k-means/HDBSCAN + UMAP over posting embeddings | Phase E |
| Classical ML — supervised | logistic regression on 40 hand labels, k-fold | Phase E |
| Fine-tuning | cross-encoder fine-tune, expected negative result | Phase F |

---

## Design principle carried forward

The guide's closing rule stands: **no `if match_score > 0.7: log_posting(...)`**. Every scoring mechanism added here is a *filter on what reaches the LLM*, never a substitute for the LLM's judgment. The cascade narrows 50 postings to 10; Gemini still decides which of those 10 are worth applying to, and still explains why. If a learned model ever auto-logs or auto-rejects, the project has stopped being agentic.

---

## New dependencies

```
chromadb
sentence-transformers
scikit-learn
umap-learn
matplotlib
rank-bm25          # keyword baseline for eval only
```

`numpy` is already pinned. Embeddings use `GoogleGenerativeAIEmbeddings` on `gemini-embedding-001` — no new API key, reuses the existing Gemini credential.

---

## Phase A — Vector store + semantic layer  *(Core)*

Introduces the vector DB and the NLP fundamentals. Nothing LLM-facing yet.

- [ ] Create `tools/vector_store.py` — a thin Chroma wrapper, persistent at `./vector_store/`, with two collections:
  - `cv_chunks` — bullet-level chunks of the CV and any past cover letters
  - `postings` — one entry per scraped posting, embedded from `title + company + description`
- [ ] Write the CV chunker: split at bullet/role boundaries rather than fixed character counts, target roughly 50–150 tokens per chunk. Store `role`, `company`, `date_range` as metadata so retrieved bullets can be cited. Expect on the order of 60–100 chunks from one CV.
- [ ] Index the CV once as a setup step, alongside the existing `profile.json` generation
- [ ] **Semantic pre-ranking**: embed the profile's target-role text, embed each posting from a JobSpy run, rank by cosine. Keep the top 30.
- [ ] **Near-duplicate detection**: within one run, flag posting pairs above a cosine threshold (start at 0.92, tune by eye on real data) as the same role cross-posted to different boards. This catches what the guide's link-based dedup in Phase 6 cannot.
- [ ] Persist posting embeddings so reruns don't re-embed — free-tier embedding quota is finite.

**Milestone:** a JobSpy run of 50 postings collapses to ~30 deduped, cosine-ranked candidates, and you can inspect the ranking to confirm it's sane before any LLM sees it.

---

## Phase B — Cross-encoder reranker  *(Core — this is the deep learning content)*

Phase A's cosine ranking is a **bi-encoder**: profile and posting are embedded separately and compared. Fast and precomputable, but the two texts never interact inside the model, so it misses constraint mismatches like "8+ years required" against a 2-year profile.

A **cross-encoder** feeds the pair through one transformer as `[profile] [SEP] [posting]`, with full cross-attention between them, and emits a single relevance score. Far more precise, but O(n) forward passes with nothing precomputable — so it only runs on a shortlist.

- [ ] Add `tools/reranker.py` using `sentence_transformers.CrossEncoder`
- [ ] Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — ~80MB, 6 layers, scores 30 pairs on CPU in a couple of seconds. (`BAAI/bge-reranker-base` is stronger but ~1.1GB and noticeably slower; try it only after MiniLM is working and measured.)
- [ ] Build the query text from the profile: target roles, must-haves, key skills — not the raw CV dump
- [ ] Batch the pairs; load the model once at module level, not per call
- [ ] Wire the full cascade: **50 scraped → cosine top 30 → cross-encoder top 10 → Gemini**
- [ ] Log per-stage timing so the latency cost is a real number you can quote

**Milestone:** the cascade runs end to end, and you can print the two rankings side by side and point at specific postings the cross-encoder demoted for a reason cosine couldn't see.

---

## Phase C — RAG for cover-letter drafting  *(Core — this is the RAG content)*

Phases A and B are vector search, not RAG — nothing retrieved conditions a generation. This phase is what makes "RAG" accurate: retrieved text is injected into a prompt and shapes the output.

- [ ] Add a `draft_cover_letter` tool: given a posting, embed the JD, retrieve the top 5–8 `cv_chunks`, inject only those into the Gemini prompt
- [ ] Prompt instruction: use only the supplied experience bullets; if the posting requires something not present in them, say so explicitly rather than inventing it
- [ ] Return the retrieved chunks alongside the letter, so every draft ships with its evidence — this is both the anti-hallucination mechanism and the demo visual
- [ ] Keep a no-RAG variant behind a flag (whole CV dumped into the prompt) purely so Phase D has something to compare against

**Milestone:** a generated letter displayed next to the exact CV bullets that produced it, with no claims that don't trace back to one of them.

---

## Phase D — Evaluation  *(Core — do not skip; it's what makes B and C defensible)*

Evaluation is its own discipline, distinct from fine-tuning and from steering. Two separate layers here, and naming them separately matters.

### D1 — Build the golden set (one afternoon, feeds Phase E too)

- [ ] Export ~40 real postings from JobSpy runs
- [ ] Label each yourself: 0 = bad match, 1 = plausible, 2 = strong match. Roughly an hour.
- [ ] Save as `eval/golden_set.json`. **These same labels are the training data for Phase E** — label once, use twice.

### D2 — IR / ranking evaluation

Classic information-retrieval metrics over the golden set:

- [ ] Implement precision@10, nDCG@10, MRR in `eval/ranking_eval.py`
- [ ] Score four variants:
  - **V0** BM25 keyword baseline (`rank-bm25`)
  - **V1** bi-encoder cosine (Phase A)
  - **V2** cosine → cross-encoder rerank (Phase B)
  - **V3** LLM-only — Gemini scores all 40 directly
- [ ] Report alongside each score: wall-clock latency and token cost per run

The comparison table is the centerpiece result. V3 may well win on quality — the interesting finding is what it costs relative to V2, and whether the gap justifies it.

### D3 — LLM output evaluation (groundedness)

- [ ] Generate 20 cover letters under both the RAG and no-RAG variants from Phase C
- [ ] LLM-as-judge rubric: decompose each letter into factual claims, mark each claim as traceable / not traceable to a supplied CV chunk
- [ ] Metric: percentage of claims grounded, plus token count per letter
- [ ] Expected result: RAG matches or beats no-RAG on groundedness at a fraction of the prompt tokens. Either outcome is reportable.

### D4 — Tracing

- [ ] `langsmith` is already in `requirements.txt` — enable it for trace-level inspection of the agent loop. Good screenshots, near-zero effort.

**Milestone:** one table quantifying the ranking cascade, one quantifying RAG groundedness. "Reranking moved precision@10 from X to Y at Z ms added latency" is the sentence the whole project builds toward.

---

## Phase E — Classical ML  *(Optional, but cheap given D1 exists)*

### E1 — Unsupervised (no labels needed)

- [ ] k-means and HDBSCAN over the posting embeddings from Phase A
- [ ] Project to 2D with UMAP, plot with matplotlib, label clusters by inspecting their top terms
- [ ] Output: a scatter plot of which market segments the job boards actually surface to you

### E2 — Supervised (uses the 40 golden-set labels)

- [ ] Features: posting embedding, plus structured signals — remote flag, seniority parsed from title, source board, description length
- [ ] Model: logistic regression, then gradient boosting for comparison
- [ ] **k-fold cross-validation, and report the confidence interval.** With n=40 it will be wide.
- [ ] Compare against the Phase B cross-encoder on the same folds

Be explicit in the writeup that n=40 is small. "CV precision 0.72 ± 0.11, and here is why the interval is wide" is a stronger result than a single impressive-looking number with no error bars. Correct methodology on small data is the point.

**Milestone:** a cluster plot and a cross-validated classifier with honest error bars.

---

## Phase F — Fine-tuning  *(Optional — expect a negative result, and say so up front)*

Included to cover the topic honestly rather than to win.

- [ ] Fine-tune `ms-marco-MiniLM-L-6-v2` on the 40 golden-set labels
- [ ] k-fold, evaluated with the same Phase D2 metrics so the numbers are directly comparable
- [ ] Predicted outcome: it **underperforms** the pretrained baseline, because 40 examples cannot beat the millions of pairs MS MARCO was trained on
- [ ] The deliverable is the analysis: what data volume would actually be needed, what overfitting looks like in the fold-by-fold numbers, and why "we tried fine-tuning and it lost" was the correct call

A documented negative result with sound methodology is a legitimate study-project finding. A fabricated win is not.

---

## Build order

1. **Phase A** — vector store, dedup, cosine ranking. Everything else depends on it.
2. **Phase D1** — label the golden set early. It gates D2, D3, E2, and F, and it's the one step that can't be rushed at the end.
3. **Phase B** — cross-encoder cascade.
4. **Phase D2** — rank the four variants. First real numbers.
5. **Phase C** — RAG cover letters.
6. **Phase D3** — groundedness eval.
7. **Phase E**, then **F**, as time allows.

Phases A–D are the closing project. E and F are bonus slides.

---

## Where this attaches to the existing guide

Mapped onto `job-application-copilot-guide.md`:

- **A and B** slot in as Phase 2.5 — after JobSpy returns postings, before anything reaches the LLM
- **C** slots in as Phase 4.5 — a new tool alongside `search_jobs`, `get_profile`, `log_posting`
- **D** is a standalone `eval/` directory, run offline, not part of the agent loop
- **E and F** are notebooks or scripts under `eval/`, also offline

The agent loop itself (guide Phase 5) is untouched. It gains one extra tool and a better-filtered candidate list; its structure doesn't change.
