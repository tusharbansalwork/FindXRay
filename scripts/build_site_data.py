"""Build the website's data files from saved results (no new API calls):
  frontend/site/data/samples.json  sample gallery from the final evaluation run (+ samples/<uid>.jpg)
  frontend/site/data/metrics.json  benchmark numbers for the "How well it works" section

Every sample is a held-out test X-ray from the v2 evaluation (sample positions 11-60). The agent's
report, its similar cases and timings are exactly what eval_generate.py recorded; the radiologist's
report comes from the dataset. Images are copied unmodified.

Selection (done by reading each report next to the radiologist's, see DECISIONS D15):
  - SELECTED: per finding type, an example where the agent's report matched the radiologist well;
  - STRUGGLE: one representative miss, shown with the reason and the planned fix.

Usage:  python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.studies import fetch_studies  # noqa: E402

EVAL = ROOT / "results" / "02_report_eval"
OUT = ROOT / "frontend" / "site" / "data"
MODEL, CONFIG = "gemini-3.5-flash-lite", "v2"
IMAGE_URL = "https://huggingface.co/datasets/tush-2ar/findxray-case-library/resolve/main/images/{uid}.jpg"

SELECTED = [
    (549, "Scarred, shrunken upper left lung", "Left upper lobe volume loss"),
    (2046, "Fluid around the right lung", "Pleural effusion"),
    (3752, "Mildly enlarged heart", "Cardiomegaly"),
    (3951, "Shadow at the top of the right lung", "Lung mass / opacity"),
    (3881, "Implanted chest port", "Support device"),
    (3392, "Healthy chest", "No finding"),
]
STRUGGLE = {
    "uid": 465,
    "title": "Small air leak at the top of the right lung",
    "term": "Pneumothorax (missed)",
    "what_happened": (
        "The radiologist reported a small pneumothorax (air outside the lung) at the right apex. FindXRay "
        "correctly found the opacity in the right lower lung, but wrote \"no pneumothorax\", and its own "
        "checking steps approved that sentence."),
    "why": [
        "The finding is a thin line at the very top of the lung: a few pixels once the whole X-ray is "
        "reduced to the size the AI reads.",
        "Similar-case search compares whole images, which captures large patterns (heart size, lung volume) "
        "well but small local findings poorly (measured in the encoder benchmark).",
        "The checker is the same model family as the writer, so it tends to agree with the draft: it approved "
        "98.6% of sentences across the evaluation.",
    ],
    "fix": [
        "Zoom-in second look: crop the lung tops and corners at full resolution and ask targeted questions "
        "(pneumothorax, small effusions) before the report is finalised.",
        "Add a dedicated pneumothorax detector as a tool the agent must consult, instead of relying on the "
        "language model's eyes alone.",
        "Use a different, stronger model as the checker, so it does not share the writer's blind spots.",
    ],
}


def build_metrics() -> dict:
    bench = ROOT / "results" / "01_encoder_benchmark"
    ret = pd.read_csv(bench / "results_retrieval.csv")
    names = {"random": "Random pick", "dinov2-base": "General-purpose vision AI (DINOv2)",
             "biomedclip": "Medical image-text AI (BiomedCLIP)", "rad-dino-224": "RAD-DINO, chest X-ray AI (used)"}
    retrieval = []
    for m, label in names.items():
        row = {"model": m, "label": label}
        for sl in ("all", "abnormal"):
            for met in ("P@5", "mAP@10", "nDCG@10", "MRR"):
                r = ret[(ret.model == m) & (ret.slice == sl) & (ret.metric == met)].iloc[0]
                row[f"{sl}:{met}"] = [round(r["mean"], 4), round(r["lo"], 4), round(r["hi"], 4)]
        retrieval.append(row)
    browser = pd.read_csv(bench / "results_browser.csv").set_index("model").loc["rad-dino-224"]
    per_finding = pd.read_csv(bench / "results_per_finding.csv")
    lift = [{"finding": r["finding"], "n": int(r["n_queries"]), "base_rate": round(r["library base rate"], 4),
             "top5_rate": round(r["rad-dino-224"], 4)} for _, r in per_finding.iterrows()]

    rep = pd.read_csv(EVAL / "metrics.csv").set_index("system")
    report = [{"system": s, "label": lbl, "n": int(rep.loc[s, "n"]),
               "micro_f1": [round(rep.loc[s, k], 4) for k in ("micro_f1_14", "micro_f1_14_lo", "micro_f1_14_hi")],
               "precision": round(rep.loc[s, "precision_14"], 4), "recall": round(rep.loc[s, "recall_14"], 4),
               "normal_abnormal_acc": round(rep.loc[s, "normal_abnormal_acc"], 4)}
              for s, lbl in [("nearest_case", "Copy the most similar case's report"),
                             ("image_only", "Same AI, no similar cases"),
                             ("agent_final", "FindXRay agent")]]
    diffs = pd.read_csv(EVAL / "paired_differences.csv")
    paired = [{"comparison": r["meaning"], "diff": round(r["diff_micro_f1_14"], 4),
               "lo": round(r["lo"], 4), "hi": round(r["hi"], 4)} for _, r in diffs.iterrows()]
    stats = json.loads((EVAL / "agent_stats.json").read_text())
    return {
        "retrieval": {"queries": 720, "library": 2517, "models": retrieval, "per_finding": lift},
        "compression": {"fp32_mb": round(browser["fp32_MB"]), "int8_mb": round(browser["int8_MB"]),
                        "p5_fp32": round(browser["P@5_fp32"], 4), "p5_int8": round(browser["P@5_int8"], 4),
                        "cos_mean": round(browser["cos_fp32_vs_int8_mean"], 4),
                        "browser_ms_median": 484, "server_search_ms": 2},
        "report": {"studies": int(rep.loc["agent_final", "n"]), "model": MODEL, "systems": report,
                   "paired": paired, "agent": stats},
    }


def main() -> None:
    recs = [json.loads(line) for line in (EVAL / "generations.jsonl").read_text().splitlines()]
    agent = {r["uid"]: r for r in recs if r.get("ok") and r["system"] == "agent"
             and r["model"] == MODEL and r.get("config", "v1") == CONFIG}
    per = pd.read_csv(EVAL / "per_study.csv")
    per = per[per.system == "agent_final"].set_index("uid")
    library = {c["uid"]: c for c in json.loads((ROOT / "artifacts" / "library" / "cases.json").read_text())}

    uids = [u for u, *_ in SELECTED] + [STRUGGLE["uid"]]
    studies = fetch_studies(uids, EVAL / "cache")
    (OUT / "samples").mkdir(parents=True, exist_ok=True)

    def labels(u, col):
        v = per.loc[u, col]
        return v.split(";") if isinstance(v, str) else []

    def sample(uid, title, term):
        r, s = agent[uid], studies[uid]
        ext = "jpg" if s["mime_type"] == "image/jpeg" else "png"
        (OUT / "samples" / f"{uid}.{ext}").write_bytes(s["image"])
        cases = []
        for rank, c in enumerate(r["cases"], 1):
            lib = library[c["uid"]]
            cases.append({"rank": rank, "uid": c["uid"], "similarity": round(c["similarity"], 4),
                          "groups": lib["groups"], "findings": lib.get("findings"),
                          "impression": lib.get("impression"), "image_url": IMAGE_URL.format(uid=c["uid"])})
        return {
            "uid": uid, "title": title, "term": term, "image": f"data/samples/{uid}.{ext}",
            "indication": s.get("indication"),
            "radiologist": {"findings": s.get("findings"), "impression": s.get("impression")},
            "report": [{k: x[k] for k in ("section", "text", "sources", "reason")} for x in r["report"]],
            "removed": [{k: x[k] for k in ("section", "text", "sources", "reason")} for x in r["removed"]],
            "missed_findings": r["missed_findings"], "revisions": r["revisions"],
            "seconds": r["seconds"], "cases": cases,
            "agreement": {"radiologist": labels(uid, "ref_labels"), "ai": labels(uid, "pred_labels"),
                          "f1": round(float(per.loc[uid, "f1_14"]), 3)},
        }

    data = {
        "model": MODEL, "config": CONFIG,
        "samples": [sample(u, t, term) for u, t, term in SELECTED],
        "struggle": {**sample(STRUGGLE["uid"], STRUGGLE["title"], STRUGGLE["term"]),
                     **{k: STRUGGLE[k] for k in ("what_happened", "why", "fix")}},
    }
    (OUT / "samples.json").write_text(json.dumps(data, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(build_metrics(), indent=1))
    print(f"wrote samples.json ({len(data['samples'])} samples + 1 struggle case) and metrics.json to {OUT}")


if __name__ == "__main__":
    main()
