"""AWS helpers, tested against moto's in-memory DynamoDB and SSM (no AWS account needed)."""

import os
import threading
from datetime import date

import numpy as np
import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from fastapi.testclient import TestClient  # noqa: E402

from casefinder.cloud import DynamoDailyCap, load_ssm_secrets  # noqa: E402

REGION = "ap-south-1"


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "testing")
    with moto.mock_aws():
        yield


@pytest.fixture
def table(aws):
    ddb = boto3.client("dynamodb", region_name=REGION)
    ddb.create_table(TableName="findxray-cap", BillingMode="PAY_PER_REQUEST",
                     AttributeDefinitions=[{"AttributeName": "day", "AttributeType": "S"}],
                     KeySchema=[{"AttributeName": "day", "KeyType": "HASH"}])
    return ddb


def test_dynamo_cap_counts_up_to_the_limit(table):
    cap = DynamoDailyCap("findxray-cap", 3, client=table, clock=lambda: date(2026, 10, 30))
    assert cap.remaining() == 3
    assert [cap.take() for _ in range(5)] == [True, True, True, False, False]
    assert cap.remaining() == 0


def test_dynamo_cap_is_shared_between_instances(table):
    """Two Lambda instances = two cap objects on the same table: together they still stop at the limit."""
    day = lambda: date(2026, 10, 30)  # noqa: E731
    a = DynamoDailyCap("findxray-cap", 4, client=table, clock=day)
    b = DynamoDailyCap("findxray-cap", 4, client=table, clock=day)
    got = [a.take(), b.take(), a.take(), b.take(), a.take(), b.take()]
    assert sum(got) == 4 and a.remaining() == b.remaining() == 0


def test_dynamo_cap_never_over_issues_under_concurrency(table):
    cap = DynamoDailyCap("findxray-cap", 10, client=table, clock=lambda: date(2026, 10, 30))
    results = []
    lock = threading.Lock()

    def worker():
        ok = cap.take()
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(25)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 10


def test_dynamo_cap_resets_on_a_new_day(table):
    today = [date(2026, 10, 30)]
    cap = DynamoDailyCap("findxray-cap", 1, client=table, clock=lambda: today[0])
    assert cap.take() and not cap.take()
    today[0] = date(2026, 10, 31)
    assert cap.remaining() == 1 and cap.take()
    item = table.get_item(TableName="findxray-cap", Key={"day": {"S": "2026-10-30"}})["Item"]
    assert int(item["expires_at"]["N"]) > 0   # TTL attribute set, so old days are deleted automatically


def test_zero_limit_never_touches_dynamodb():
    cap = DynamoDailyCap("no-such-table", 0, client=None)
    assert cap.take() and cap.remaining() is None


def test_ssm_secrets_load_into_environment(aws, monkeypatch):
    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(Name="/findxray/GEMINI_API_KEY", Value="secret-gemini", Type="SecureString")
    ssm.put_parameter(Name="/findxray/LANGFUSE_PUBLIC_KEY", Value="pk-test", Type="SecureString")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "already-set")
    loaded = load_ssm_secrets("/findxray/", client=ssm)
    assert loaded == ["GEMINI_API_KEY"]
    assert os.environ["GEMINI_API_KEY"] == "secret-gemini"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "already-set"   # an existing value wins
    monkeypatch.delenv("GEMINI_API_KEY")


def test_no_prefix_means_no_ssm_call(monkeypatch):
    monkeypatch.delenv("FINDXRAY_SSM_PREFIX", raising=False)
    assert load_ssm_secrets() == []


def test_api_uses_the_dynamodb_cap_when_configured(table, monkeypatch):
    from test_api_report import OneShotModel, body, client_with
    monkeypatch.setenv("CASEFINDER_CAP_TABLE", "findxray-cap")
    monkeypatch.setenv("CASEFINDER_DAILY_REPORTS", "2")
    c, emb = client_with(OneShotModel())
    with c:
        codes = [c.post("/report", json=body(emb)).status_code for _ in range(3)]
        left = c.get("/health").json()["reports_left_today"]
    assert codes == [200, 200, 429] and left == 0
    items = table.scan(TableName="findxray-cap")["Items"]
    assert len(items) == 1 and items[0]["used"]["N"] == "2"


def test_killswitch_sets_concurrency_to_zero(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "killswitch", Path(__file__).resolve().parents[1] / "deploy" / "aws" / "killswitch" / "handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    class FakeLambda:
        def put_function_concurrency(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(mod.boto3, "client", lambda name: FakeLambda())
    monkeypatch.setenv("TARGET_FUNCTION", "findxray-api")
    out = mod.handler({"Records": [{"Sns": {"Subject": "AWS Budgets: findxray-monthly has exceeded"}}]}, None)
    assert calls == [{"FunctionName": "findxray-api", "ReservedConcurrentExecutions": 0}]
    assert out == {"throttled": "findxray-api"}


def test_health_survives_a_failing_cap(monkeypatch):
    from test_api_report import OneShotModel, client_with

    class Broken:
        def take(self):
            return True

        def remaining(self):
            raise RuntimeError("ProvisionedThroughputExceededException")

    c, _ = client_with(OneShotModel())
    with c:
        c.app.state.cap = Broken()
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["reports_left_today"] is None


def test_lambda_metrics_summary(monkeypatch, tmp_path, capsys):
    """scripts/lambda_metrics.py against a fake CloudWatch Logs client (Insights result format)."""
    import importlib.util
    import json as _json
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "lambda_metrics", Path(__file__).resolve().parents[1] / "scripts" / "lambda_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def row(**kw):
        return [{"field": k, "value": str(v)} for k, v in kw.items()]

    reports = [row(**{"@duration": 20, "@billedDuration": 20, "@memorySize": 1024e6, "@maxMemoryUsed": 300e6}) for _ in range(8)]
    reports += [row(**{"@duration": 25, "@billedDuration": 900, "@memorySize": 1024e6, "@maxMemoryUsed": 310e6,
                       "@initDuration": 850}) for _ in range(2)]
    reqs = [row(kind="request", path="/search", status=200, ms=m) for m in (4, 5, 6, 7, 30)]
    reqs += [row(kind="request", path="/search", status=502, ms=9), row(kind="report_done", ms=16000)]

    class FakeLogs:
        def __init__(self):
            self.n = 0

        def start_query(self, **kw):
            self.n += 1
            return {"queryId": "reports" if "REPORT" in kw["queryString"] else "requests"}

        def get_query_results(self, queryId):
            return {"status": "Complete", "results": reports if queryId == "reports" else reqs}

    monkeypatch.setattr(mod.boto3, "client", lambda name: FakeLogs())
    out_file = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", ["lambda_metrics.py", "--minutes", "5", "--out", str(out_file)])
    mod.main()
    out = _json.loads(out_file.read_text())
    assert out["invocations"] == 10 and out["cold_starts"] == 2 and out["cold_start_rate"] == 0.2
    assert out["init_ms_p50"] == 850 and out["max_memory_used_mb"] == 310
    s = out["endpoints"]["/search"]
    assert s["requests"] == 6 and s["server_ms_p50"] == 6.5 and s["error_rate"] == round(1 / 6, 4)
    assert out["report_generation_ms_p50"] == 16000
    # 8 x 0.02 s + 2 x 0.9 s at 1 GB = 1.96 GB-s
    assert out["billed_gb_seconds"] == 1.96
    assert out["list_price_usd_per_1000_invocations"] == round(1000 * (1.96 / 10 * 0.0000133334 + 0.2e-6), 5)
