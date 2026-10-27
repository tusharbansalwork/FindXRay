import math

import numpy as np
import pytest

from casefinder.metrics import relevance, retrieval_metrics, summarize


def test_relevance_and_jaccard():
    # classes: [NoFinding, Cardiomegaly, Effusion]
    q = np.array([[0, 1, 1]])
    lib = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
    rel, gain = relevance(q, lib)
    assert rel.tolist() == [[True, False, True]]
    assert np.allclose(gain, [[0.5, 0.0, 1.0]])


def test_hand_worked_ranking():
    # library of 5; ranking by sim is items 0,1,2,3,4; relevant = {0, 2, 4}
    sim = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
    rel = np.array([[1, 0, 1, 0, 1]], dtype=bool)
    m = retrieval_metrics(sim, rel, ks=(1, 3, 5))
    assert m["P@1"][0] == 1.0
    assert m["P@3"][0] == pytest.approx(2 / 3)
    assert m["Hit@3"][0] == 1.0
    assert m["R@3"][0] == pytest.approx(2 / 3)
    assert m["R@5"][0] == 1.0
    # AP@3 = (1/1 + 2/3) / min(3, 3) = 5/9
    assert m["mAP@3"][0] == pytest.approx(5 / 9)
    # AP@5 = (1 + 2/3 + 3/5) / 3
    assert m["mAP@5"][0] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert m["MRR"][0] == 1.0
    # binary nDCG@3: dcg = 1/log2(2) + 1/log2(4) ; ideal = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG@3"][0] == pytest.approx(dcg / idcg)


def test_order_comes_from_sim_not_column_order():
    sim = np.array([[0.1, 0.9, 0.5]])
    rel = np.array([[0, 1, 0]], dtype=bool)
    m = retrieval_metrics(sim, rel, ks=(1,))
    assert m["P@1"][0] == 1.0 and m["MRR"][0] == 1.0


def test_first_relevant_at_rank_three():
    sim = np.array([[3.0, 2.0, 1.0]])
    rel = np.array([[0, 0, 1]], dtype=bool)
    m = retrieval_metrics(sim, rel, ks=(1, 3))
    assert m["MRR"][0] == pytest.approx(1 / 3)
    assert m["Hit@1"][0] == 0.0 and m["Hit@3"][0] == 1.0
    assert m["mAP@3"][0] == pytest.approx(1 / 3)


def test_query_without_relevant_items_is_nan_not_zero():
    sim = np.array([[0.9, 0.1], [0.9, 0.1]])
    rel = np.array([[0, 0], [1, 0]], dtype=bool)
    m = retrieval_metrics(sim, rel, ks=(1,))
    assert np.isnan(m["R@1"][0]) and np.isnan(m["MRR"][0]) and np.isnan(m["mAP@1"][0])
    assert m["P@1"][0] == 0.0
    s = summarize(m, n_boot=50)
    assert s["MRR"][0] == 1.0  # only the second query counts


def test_graded_ndcg_prefers_higher_overlap_first():
    gain = np.array([[0.5, 1.0]])
    rel = gain > 0
    good = retrieval_metrics(np.array([[0.1, 0.9]]), rel, gain, ks=(2,))
    bad = retrieval_metrics(np.array([[0.9, 0.1]]), rel, gain, ks=(2,))
    assert good["nDCG@2"][0] == pytest.approx(1.0)
    assert bad["nDCG@2"][0] < 1.0


def test_random_scores_match_base_rate():
    rng = np.random.default_rng(0)
    rel = rng.random((400, 300)) < 0.2
    m = retrieval_metrics(rng.random((400, 300)), rel, ks=(10,))
    assert abs(np.mean(m["P@10"]) - 0.2) < 0.02


def test_k_larger_than_library_raises():
    with pytest.raises(ValueError):
        retrieval_metrics(np.zeros((1, 3)), np.zeros((1, 3), bool), ks=(5,))


def test_bootstrap_ci_contains_mean():
    m = {"x": np.linspace(0, 1, 101)}
    mean, lo, hi = summarize(m, n_boot=500)["x"]
    assert lo <= mean <= hi and mean == pytest.approx(0.5)
