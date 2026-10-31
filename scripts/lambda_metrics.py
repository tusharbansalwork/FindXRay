"""Server-side metrics for the FindXRay Lambda from its CloudWatch logs (run after a load test).

Reads two kinds of log lines over a time window with CloudWatch Logs Insights:
  - Lambda's own REPORT lines: invocations, cold starts (init duration), billed time, memory used;
  - the API's JSON request lines: per-endpoint server latency (p50/p95/p99) and error rate,
    plus "report_done" lines for full report generation time.
Then estimates cost per 1,000 requests at list price (actual cost is $0 inside the always-free tier).

Usage (AWS CLI credentials configured, e.g. `aws configure`):
  python scripts/lambda_metrics.py --minutes 15 --out results/03_cloud/aws_lambda_metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3
import numpy as np

# List prices for arm64 Lambda (USD). Check https://aws.amazon.com/lambda/pricing/ for your region.
PRICE_PER_GB_SECOND = 0.0000133334
PRICE_PER_REQUEST = 0.20 / 1_000_000


def insights(logs, group: str, query: str, start: int, end: int) -> list[dict]:
    qid = logs.start_query(logGroupName=group, startTime=start, endTime=end, queryString=query, limit=10000)["queryId"]
    while True:
        res = logs.get_query_results(queryId=qid)
        if res["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        time.sleep(1)
    if res["status"] != "Complete":
        raise SystemExit(f"Logs Insights query {res['status']}")
    return [{f["field"]: f["value"] for f in row} for row in res["results"]]


def pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--function", default="findxray-api")
    ap.add_argument("--minutes", type=int, default=15, help="look back this many minutes")
    ap.add_argument("--out", help="also write the summary as JSON here")
    args = ap.parse_args()

    logs = boto3.client("logs")
    group = f"/aws/lambda/{args.function}"
    end = int(time.time())
    start = end - 60 * args.minutes

    reports = insights(logs, group, """filter @type = "REPORT"
        | fields @duration, @billedDuration, @memorySize, @maxMemoryUsed, @initDuration""", start, end)
    requests = insights(logs, group, """filter event = "findxray"
        | fields kind, path, status, ms""", start, end)

    billed_ms = np.array([float(r["@billedDuration"]) for r in reports])
    mem_mb = np.array([float(r["@memorySize"]) / 1e6 for r in reports]) if reports else np.array([])
    init = np.array([float(r["@initDuration"]) for r in reports if r.get("@initDuration")])
    gb_s = float(np.sum(billed_ms / 1000 * (mem_mb / 1024))) if len(reports) else 0.0

    out = {
        "window_minutes": args.minutes,
        "invocations": len(reports),
        "cold_starts": int(len(init)),
        "cold_start_rate": round(len(init) / len(reports), 4) if reports else None,
        "init_ms_p50": pct(init, 50), "init_ms_p95": pct(init, 95),
        "max_memory_used_mb": max((float(r["@maxMemoryUsed"]) / 1e6 for r in reports), default=None),
        "billed_gb_seconds": round(gb_s, 2),
        "endpoints": {},
    }
    for path in sorted({r.get("path") for r in requests if r.get("kind") == "request"}):
        rows = [r for r in requests if r.get("kind") == "request" and r.get("path") == path]
        ms = np.array([float(r["ms"]) for r in rows])
        errors = sum(int(r["status"]) >= 500 for r in rows)
        out["endpoints"][path] = {"requests": len(rows), "server_ms_p50": pct(ms, 50), "server_ms_p95": pct(ms, 95),
                                  "server_ms_p99": pct(ms, 99), "error_rate": round(errors / len(rows), 4)}
    done = np.array([float(r["ms"]) for r in requests if r.get("kind") == "report_done"])
    if len(done):
        out["report_generation_ms_p50"], out["report_generation_ms_p95"] = pct(done, 50), pct(done, 95)

    # Cost per 1,000 requests at list price, from the average billed GB-seconds per invocation.
    if reports:
        per_inv = gb_s / len(reports) * PRICE_PER_GB_SECOND + PRICE_PER_REQUEST
        out["list_price_usd_per_1000_invocations"] = round(1000 * per_inv, 5)
    out["free_tier_note"] = "Always-free Lambda allowance: 1M requests and 400,000 GB-seconds a month."

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
