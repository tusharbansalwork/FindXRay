"""Retrieval metrics for similar-case search.

Conventions (stated explicitly because "recall@k" means different things in
different communities):

- P@k     : fraction of the top-k results that are relevant.
- Hit@k   : 1 if at least one of the top-k results is relevant, else 0.
            (Image-retrieval / metric-learning papers often call this "Recall@k".)
- R@k     : relevant items found in the top-k / all relevant items in the library.
            (The information-retrieval meaning of recall@k.)
- mAP@k   : mean over queries of AP@k = (1 / min(k, n_rel)) * sum_i P@i * rel_i, i <= k.
- nDCG@k  : graded relevance; gain = Jaccard overlap of the query's and the case's
            label sets, discount = 1 / log2(rank + 1).
- MRR     : 1 / rank of the first relevant result (no cutoff).

Binary relevance: a library case is relevant to a query if their label sets share
at least one label ("No Finding" counts as a label, so normal matches normal).

Queries with no relevant item in the library get NaN for R@k, mAP@k, MRR and
nDCG@k and are skipped by the nan-aware means; P@k and Hit@k are 0 for them.
"""

from __future__ import annotations

import warnings

import numpy as np

DEFAULT_KS = (1, 5, 10)


def relevance(q_labels: np.ndarray, lib_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary relevance and Jaccard gain between multi-hot label matrices.

    q_labels: (Q, C) bool/int, lib_labels: (N, C) bool/int.
    Returns rel (Q, N) bool and gain (Q, N) float in [0, 1].
    """
    q = np.asarray(q_labels, dtype=np.int32)
    lib = np.asarray(lib_labels, dtype=np.int32)
    inter = q @ lib.T
    union = q.sum(1)[:, None] + lib.sum(1)[None, :] - inter
    gain = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return inter > 0, gain


def retrieval_metrics(
    sim: np.ndarray,
    rel: np.ndarray,
    gain: np.ndarray | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, np.ndarray]:
    """Per-query metrics. sim, rel, gain are (Q, N); higher sim = more similar."""
    sim = np.asarray(sim, dtype=np.float64)
    rel = np.asarray(rel, dtype=bool)
    gain = rel.astype(np.float64) if gain is None else np.asarray(gain, dtype=np.float64)
    q, n = sim.shape
    if rel.shape != (q, n) or gain.shape != (q, n):
        raise ValueError("sim, rel and gain must have the same shape")
    if max(ks) > n:
        raise ValueError(f"k={max(ks)} is larger than the library size {n}")

    order = np.argsort(-sim, axis=1, kind="stable")
    rel_s = np.take_along_axis(rel, order, axis=1)
    gain_s = np.take_along_axis(gain, order, axis=1)
    ideal = -np.sort(-gain, axis=1)
    n_rel = rel.sum(1)
    has_rel = n_rel > 0

    out: dict[str, np.ndarray] = {}
    for k in ks:
        top = rel_s[:, :k]
        hits = top.sum(1)
        out[f"P@{k}"] = hits / k
        out[f"Hit@{k}"] = (hits > 0).astype(np.float64)
        out[f"R@{k}"] = np.where(has_rel, hits / np.maximum(n_rel, 1), np.nan)

        prec_at_i = np.cumsum(top, axis=1) / np.arange(1, k + 1)
        ap = (prec_at_i * top).sum(1) / np.maximum(np.minimum(k, n_rel), 1)
        out[f"mAP@{k}"] = np.where(has_rel, ap, np.nan)

        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (gain_s[:, :k] * disc).sum(1)
        idcg = (ideal[:, :k] * disc).sum(1)
        out[f"nDCG@{k}"] = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), np.nan)

    first_rank = np.argmax(rel_s, axis=1) + 1  # argmax is 0 when nothing is relevant; masked below
    out["MRR"] = np.where(has_rel, 1.0 / first_rank, np.nan)
    return out


def summarize(
    per_query: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Mean and 95% bootstrap CI (resampling queries) for each metric."""
    rng = np.random.default_rng(seed)
    any_metric = next(iter(per_query.values()))
    q = len(any_metric)
    idx = rng.integers(0, q, size=(n_boot, q))
    summary = {}
    for name, vals in per_query.items():
        vals = np.asarray(vals, dtype=np.float64)
        mean = float(np.nanmean(vals))
        with warnings.catch_warnings():  # a resample can be all-NaN for tiny slices
            warnings.simplefilter("ignore", RuntimeWarning)
            boots = np.nanmean(vals[idx], axis=1)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        summary[name] = (mean, float(lo), float(hi))
    return summary
