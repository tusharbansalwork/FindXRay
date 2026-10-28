"""Report-quality metrics on CheXbert labels (pure functions; the labeller itself lives in scripts/eval_score.py).

Label vectors follow f1chexbert's "rrg" convention: 14 CheXpert observations, 1 if the report states
the finding as present or uncertain, 0 if negative or not mentioned.

F1 conventions match the radiology report generation literature (sklearn classification_report):
- micro-F1 pools true/false positives over all labels and studies;
- macro-F1 averages per-label F1, counting a label with no positives in either reference or
  prediction as F1 = 0 (sklearn's zero_division=0), which lowers macro-F1 on small samples.
"""

from __future__ import annotations

import numpy as np

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding"]
TOP5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
TOP5_IDX = [CHEXPERT_LABELS.index(x) for x in TOP5]
NO_FINDING_IDX = CHEXPERT_LABELS.index("No Finding")


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return (t & p).sum(0), (~t & p).sum(0), (t & ~p).sum(0)  # per-label TP, FP, FN


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def f1_scores(y_true, y_pred, labels: list[int] | None = None) -> dict[str, float]:
    """micro/macro F1, precision and recall for binary label matrices (studies x labels)."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if labels is not None:
        t, p = t[:, labels], p[:, labels]
    tp, fp, fn = _counts(t, p)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    return {
        "micro_f1": float(_f1(TP, FP, FN)),
        "micro_precision": float(TP / (TP + FP)) if TP + FP else 0.0,
        "micro_recall": float(TP / (TP + FN)) if TP + FN else 0.0,
        "macro_f1": float(np.mean(_f1(tp, fp, fn))),
    }


def bootstrap_ci(y_true, y_pred, metric: str = "micro_f1", labels=None, n_boot: int = 2000, seed: int = 0):
    """Mean and 95% percentile CI of an f1_scores metric, resampling studies."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    vals = np.array([f1_scores(t[i], p[i], labels)[metric] for i in idx])
    return f1_scores(t, p, labels)[metric], float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: str = "micro_f1", labels=None, n_boot: int = 2000,
                          seed: int = 0):
    """metric(A) - metric(B) on the same studies, with a 95% CI from resampling studies jointly.
    A CI that excludes 0 means the difference is unlikely to be sampling noise."""
    t, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    diffs = np.array([f1_scores(t[i], a[i], labels)[metric] - f1_scores(t[i], b[i], labels)[metric] for i in idx])
    point = f1_scores(t, a, labels)[metric] - f1_scores(t, b, labels)[metric]
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def normal_abnormal_accuracy(y_true, y_pred) -> float:
    """Share of studies where the prediction agrees with the reference on 'No Finding'."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(t[:, NO_FINDING_IDX] == p[:, NO_FINDING_IDX]))


def citation_stats(sentences: list[dict], label_of: dict[str, list[int]], case_label_of: dict[int, list[int]]):
    """How well case citations back up the findings they are attached to.

    For each sentence that asserts at least one positive finding (other than No Finding), and each
    case it cites, check whether that case's report also has each asserted finding.
    Returns (cases cited per sentence, consistency = supported pairs / all pairs, number of pairs).
    """
    n_case_citations, supported, pairs = 0, 0, 0
    for s in sentences:
        cited = [int(src.split(":", 1)[1]) for src in s["sources"]
                 if src.strip().lower().startswith("case:") and src.split(":", 1)[1].strip().isdigit()]
        n_case_citations += len(cited)
        positives = [i for i, v in enumerate(label_of[s["text"]]) if v and i != NO_FINDING_IDX]
        for uid in cited:
            case_labels = case_label_of.get(uid)
            if case_labels is None:
                continue
            for i in positives:
                pairs += 1
                supported += int(bool(case_labels[i]))
    per_sentence = n_case_citations / len(sentences) if sentences else 0.0
    return per_sentence, (supported / pairs if pairs else float("nan")), pairs
