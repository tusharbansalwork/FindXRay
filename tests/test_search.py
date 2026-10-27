import numpy as np
import pytest

from casefinder.search import CaseLibrary


def make_lib(n=50, d=8, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    cases = [{"uid": i, "groups": ["No Finding"]} for i in range(n)]
    return CaseLibrary(emb, cases, model="test"), emb


def test_item_finds_itself_first():
    lib, emb = make_lib()
    for i in (0, 17, 49):
        hits = lib.search(emb[i], k=3)
        assert hits[0].case["uid"] == i
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_results_sorted_and_match_brute_force():
    lib, emb = make_lib()
    q = np.random.default_rng(1).standard_normal(8)
    hits = lib.search(q, k=10)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    expected = np.argsort(-(normed @ (q / np.linalg.norm(q))))[:10]
    assert [h.case["uid"] for h in hits] == expected.tolist()
    assert [h.rank for h in hits] == list(range(1, 11))


def test_query_scale_does_not_matter():
    lib, emb = make_lib()
    a = [h.case["uid"] for h in lib.search(emb[3], k=5)]
    b = [h.case["uid"] for h in lib.search(emb[3] * 42.0, k=5)]
    assert a == b


@pytest.mark.parametrize("bad, msg", [
    (np.zeros(8), "zero"),
    (np.full(8, np.nan), "NaN"),
    (np.ones(7), "expected 8"),
])
def test_bad_queries_rejected(bad, msg):
    lib, _ = make_lib()
    with pytest.raises(ValueError, match=msg):
        lib.search(bad, k=5)


def test_k_bounds():
    lib, emb = make_lib(n=5)
    assert len(lib.search(emb[0], k=5)) == 5
    with pytest.raises(ValueError):
        lib.search(emb[0], k=6)
    with pytest.raises(ValueError):
        lib.search(emb[0], k=0)


def test_mismatched_cases_rejected():
    with pytest.raises(ValueError):
        CaseLibrary(np.ones((3, 4)), [{"uid": 1}])
