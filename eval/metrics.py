"""
metrics.py

description: Phase 8.2 - classic IR ranking metrics over graded relevance
labels (0/1/2, from eval/golden_set.json). Shared by ranking_eval.py's four
variants so every variant is scored identically.

Each function takes `labels`: the golden-set label of each item in a
ranked list, in ranked order (best-first) - not the raw golden set. An id
with no golden-set entry (the variant surfaced something unlabeled) should
be passed in as relevance 0, same as an explicit "bad match" label.
"""
import math


def precision_at_k(labels: list[int], k: int) -> float:
    """Fraction of the top k that are relevant (label >= 1)."""
    top_k = labels[:k]
    if not top_k:
        return 0.0
    return sum(1 for label in top_k if label >= 1) / len(top_k)


def _dcg_at_k(labels: list[int], k: int) -> float:
    return sum(label / math.log2(i + 2) for i, label in enumerate(labels[:k]))


def ndcg_at_k(labels: list[int], k: int) -> float:
    """Graded nDCG@k - relevance is the label itself (0/1/2), so a strong
    match ranked highly counts more than a merely-plausible one."""
    dcg = _dcg_at_k(labels, k)
    ideal_dcg = _dcg_at_k(sorted(labels, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def mrr(labels: list[int]) -> float:
    """Reciprocal rank of the first relevant (label >= 1) item, 0 if none."""
    for i, label in enumerate(labels):
        if label >= 1:
            return 1 / (i + 1)
    return 0.0
