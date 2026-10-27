import numpy as np
import pytest
from fastapi.testclient import TestClient

from casefinder.api import create_app
from casefinder.search import CaseLibrary

D = 8


@pytest.fixture
def client():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, D)).astype(np.float32)
    cases = [{"uid": 100 + i, "groups": ["Cardiomegaly"], "indication": None,
              "findings": f"finding {i}", "impression": f"impression {i}"} for i in range(30)]
    lib = CaseLibrary(emb, cases, model="test-model")
    with TestClient(create_app(lib)) as c:
        c.emb = emb
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "test-model", "library_size": 30, "dim": D}


def test_search_returns_self_first(client):
    r = client.post("/search", json={"embedding": client.emb[7].tolist(), "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["library_size"] == 30 and len(body["results"]) == 3
    top = body["results"][0]
    assert top["uid"] == 107 and top["rank"] == 1
    assert top["impression"] == "impression 7"
    assert top["image_url"].endswith("/images/107.jpg")


def test_image_url_template_is_configurable(monkeypatch):
    monkeypatch.setenv("CASEFINDER_IMAGE_URL", "https://example.org/x/{uid}.jpg")
    emb = np.eye(D, dtype=np.float32)
    lib = CaseLibrary(emb, [{"uid": i, "groups": ["No Finding"]} for i in range(D)])
    with TestClient(create_app(lib)) as c:
        r = c.post("/search", json={"embedding": emb[2].tolist(), "k": 1})
    assert r.json()["results"][0]["image_url"] == "https://example.org/x/2.jpg"


def test_image_url_template_without_uid_is_rejected(monkeypatch):
    monkeypatch.setenv("CASEFINDER_IMAGE_URL", "https://example.org/fixed.jpg")
    with pytest.raises(ValueError):
        create_app(CaseLibrary(np.eye(D), [{"uid": i, "groups": []} for i in range(D)]))


def test_default_k_is_5(client):
    r = client.post("/search", json={"embedding": client.emb[0].tolist()})
    assert len(r.json()["results"]) == 5


@pytest.mark.parametrize("payload", [
    {"embedding": [0.1] * (D - 1)},           # wrong length
    {"embedding": [0.0] * D},                 # zero vector
    {"embedding": [0.1] * D, "k": 0},         # k too small
    {"embedding": [0.1] * D, "k": 21},        # k too large
    {"embedding": "not a list"},
    {},
])
def test_bad_requests_are_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_nan_rejected(client):
    # JSON has no NaN literal; Python's json emits one anyway, which must not crash the server
    body = '{"embedding": [NaN, 1, 1, 1, 1, 1, 1, 1]}'
    r = client.post("/search", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_cors_allows_configured_origin_only(client):
    ok = client.options("/search", headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8000"
    bad = client.options("/search", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
