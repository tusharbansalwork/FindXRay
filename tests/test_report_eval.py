import math

import numpy as np
import pytest
from sklearn.metrics import f1_score

from casefinder.report_eval import (CHEXPERT_LABELS, NO_FINDING_IDX, TOP5_IDX, bootstrap_ci, citation_stats,
                                    f1_scores, normal_abnormal_accuracy, paired_bootstrap_diff)


def random_labels(n=40, seed=0, p=0.3):
    rng = np.random.default_rng(seed)
    return (rng.random((n, len(CHEXPERT_LABELS))) < p).astype(int)


def test_matches_sklearn_micro_and_macro():
    t, p = random_labels(seed=1), random_labels(seed=2)
    s = f1_scores(t, p)
    assert s["micro_f1"] == pytest.approx(f1_score(t, p, average="micro", zero_division=0))
    assert s["macro_f1"] == pytest.approx(f1_score(t, p, average="macro", zero_division=0))
    s5 = f1_scores(t, p, TOP5_IDX)
    assert s5["micro_f1"] == pytest.approx(f1_score(t[:, TOP5_IDX], p[:, TOP5_IDX], average="micro", zero_division=0))


def test_hand_worked_example():
    t = np.zeros((2, 14), int); p = np.zeros((2, 14), int)
    t[0, 1] = 1; p[0, 1] = 1          # TP
    t[1, 9] = 1                      # FN
    p[1, 4] = 1                      # FP
    s = f1_scores(t, p)
    assert s["micro_precision"] == 0.5 and s["micro_recall"] == 0.5 and s["micro_f1"] == 0.5
    # per-label F1: label1 = 1, label4 = 0, label9 = 0, eleven labels empty -> 0 (sklearn convention)
    assert s["macro_f1"] == pytest.approx(1 / 14)


def test_perfect_and_empty():
    t = random_labels(seed=3)
    assert f1_scores(t, t)["micro_f1"] == 1.0
    z = np.zeros_like(t)
    assert f1_scores(z, z)["micro_f1"] == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    t, p = random_labels(seed=4), random_labels(seed=5)
    point, lo, hi = bootstrap_ci(t, p, n_boot=300)
    assert lo <= point <= hi and hi - lo > 0


def test_paired_diff_of_identical_systems_is_zero():
    t, p = random_labels(seed=6), random_labels(seed=7)
    point, lo, hi = paired_bootstrap_diff(t, p, p, n_boot=200)
    assert point == 0 and lo == 0 and hi == 0


def test_paired_diff_detects_better_system():
    t = random_labels(n=60, seed=8)
    worse = random_labels(n=60, seed=9)
    point, lo, hi = paired_bootstrap_diff(t, t, worse, n_boot=300)
    assert point > 0 and lo > 0


def test_normal_abnormal_accuracy():
    t = np.zeros((4, 14), int); p = np.zeros((4, 14), int)
    t[:2, NO_FINDING_IDX] = 1; p[:1, NO_FINDING_IDX] = 1
    assert normal_abnormal_accuracy(t, p) == 0.75


def vec(*names):
    v = [0] * 14
    for n in names:
        v[CHEXPERT_LABELS.index(n)] = 1
    return v


def test_citation_stats():
    sentences = [
        {"text": "Heart is enlarged.", "sources": ["image", "case:1", "case:2"]},     # 2 pairs, 1 supported
        {"text": "No effusion.", "sources": ["image", "case:1"]},                      # no positives -> no pairs
        {"text": "Right effusion.", "sources": ["case:3"]},                            # 1 pair, supported
        {"text": "Odd.", "sources": ["case:99", "case:x"]},                            # unknown case, malformed
    ]
    label_of = {"Heart is enlarged.": vec("Cardiomegaly"), "No effusion.": vec(),
                "Right effusion.": vec("Pleural Effusion"), "Odd.": vec("Fracture")}
    cases = {1: vec("Cardiomegaly"), 2: vec("No Finding"), 3: vec("Pleural Effusion", "Cardiomegaly")}
    per_sentence, consistency, pairs = citation_stats(sentences, label_of, cases)
    assert per_sentence == pytest.approx(5 / 4)          # case:1, case:2, case:1, case:3, case:99
    assert pairs == 3 and consistency == pytest.approx(2 / 3)


def test_citation_stats_without_positive_claims_is_nan():
    per_sentence, consistency, pairs = citation_stats(
        [{"text": "Normal.", "sources": ["case:1"]}], {"Normal.": vec("No Finding")}, {1: vec("No Finding")})
    assert pairs == 0 and math.isnan(consistency) and per_sentence == 1.0
