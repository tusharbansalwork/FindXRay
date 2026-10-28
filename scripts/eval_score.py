"""Score the generated reports against the radiologists' reports with CheXbert.

Systems compared on the same studies:
  nearest_case  copy the report of the most similar library case (no LLM; retrieval-only baseline)
  image_only    Gemini, same prompt, no similar cases
  agent_draft   the agent's first draft (retrieval + checklist + Gemini, before look-again and verification)
  agent_final   the agent's verified report

Outputs results/02_report_eval/{metrics.csv, per_study.csv, summary.md}. CheXbert labels are cached
in chexbert_labels.json, so rescoring is fast. --push-langfuse attaches per-report scores to traces.

Needs the eval extras:  pip install -e ".[eval]"   (PyTorch + CheXbert, ~1.3 GB weights on first run)
Usage:  python scripts/eval_score.py [--push-langfuse]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.report_eval import (CHEXPERT_LABELS, NO_FINDING_IDX, TOP5_IDX, bootstrap_ci,  # noqa: E402
                                    citation_stats, f1_scores, normal_abnormal_accuracy, paired_bootstrap_diff)
from casefinder.search import CaseLibrary  # noqa: E402
from casefinder.studies import reference_text  # noqa: E402

EVAL_DIR = ROOT / "results" / "02_report_eval"
SYSTEMS = ["nearest_case", "image_only", "agent_draft", "agent_final"]


class CheXbert:
    """CheXbert labeller from f1chexbert, with the checkpoint fetched via huggingface_hub
    (f1chexbert's own downloader uses an argument newer huggingface_hub releases removed)."""

    def __init__(self, cache_path: Path):
        import os

        import f1chexbert.f1chexbert as fc
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth")
        fc.CACHE_DIR = os.path.dirname(ckpt)  # F1CheXbert loads <CACHE_DIR>/chexbert.pth
        self.model = fc.F1CheXbert(device="cpu")
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def labels(self, text: str) -> list[int]:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = [int(x) for x in self.model.get_label(text.strip())]
        return self.cache[key]

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache))


def report_text(sentences: list[dict]) -> str:
    ordered = [s for s in sentences if s["section"] == "findings"] + [s for s in sentences if s["section"] == "impression"]
    return " ".join(s["text"].strip() for s in ordered)


def latest_ok(path: Path, model: str, config: str = "v1") -> dict[tuple[str, int], dict]:
    """Latest successful result per (system, uid) for one model and config label (never mix them)."""
    out = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("ok") and rec.get("model") == model and rec.get("config", "v1") == config:
            out[(rec["system"], rec["uid"])] = rec
    return out


def per_study_f1(t, p) -> float:
    """F1 for one study; if neither side has any positive label, they agree completely (1.0)."""
    tp = int(np.sum((t == 1) & (p == 1))); fp = int(np.sum((t == 0) & (p == 1))); fn = int(np.sum((t == 1) & (p == 0)))
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-langfuse", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--model", default=None, help="score results of this model (default: GEMINI_MODEL / app default)")
    ap.add_argument("--config", default="v2", help="config label to score (as given to eval_generate.py)")
    ap.add_argument("--start", type=int, default=1, help="first sample position to score (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last sample position to score (inclusive)")
    args = ap.parse_args()

    from casefinder.agent.model import DEFAULT_MODEL
    import os
    model = args.model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    gens = latest_ok(EVAL_DIR / "generations.jsonl", model, args.config)
    print(f"scoring results generated with {model}, config {args.config}")
    sample = pd.read_csv(EVAL_DIR / "sample.csv")["uid"].astype(int).tolist()
    sample = sample[args.start - 1:args.end]
    print(f"sample positions {args.start}-{args.start + len(sample) - 1}")
    studies = {int(k): v for k, v in json.loads((EVAL_DIR / "cache" / "studies.json").read_text()).items()}
    uids = [u for u in sample if ("agent", u) in gens and ("image_only", u) in gens]
    print(f"{len(uids)} of {len(sample)} sampled studies have both agent and image-only results")
    if not uids:
        raise SystemExit("nothing to score yet; run scripts/eval_generate.py first")

    library = CaseLibrary.load(ROOT / "artifacts" / "library")
    case_by_uid = {c["uid"]: c for c in library.cases}
    case_text = {u: reference_text(c) for u, c in case_by_uid.items()}
    all_uids = list(np.load(ROOT / "artifacts" / "embeddings" / "uids.npy"))
    emb = np.load(ROOT / "artifacts" / "embeddings" / "rad-dino-224.npy")

    texts = {"reference": {}, **{s: {} for s in SYSTEMS}}
    for u in uids:
        texts["reference"][u] = reference_text(studies[u])
        nearest = library.search(emb[all_uids.index(u)], k=1)[0].case["uid"]
        texts["nearest_case"][u] = case_text[nearest]
        texts["image_only"][u] = report_text(gens[("image_only", u)]["report"])
        texts["agent_draft"][u] = report_text(gens[("agent", u)]["first_draft"])
        texts["agent_final"][u] = report_text(gens[("agent", u)]["report"])

    print("loading CheXbert (first run downloads ~1.3 GB)...")
    cx = CheXbert(EVAL_DIR / "chexbert_labels.json")
    Y = {name: np.array([cx.labels(texts[name][u]) for u in uids]) for name in texts}

    # Citation quality of the agent's final reports
    cited_cases = {int(src.split(":")[1]) for u in uids for s in gens[("agent", u)]["report"]
                   for src in s["sources"] if src.lower().startswith("case:") and src.split(":")[1].isdigit()}
    case_labels = {c: cx.labels(case_text[c]) for c in cited_cases if c in case_text}
    sentence_labels = {s["text"]: cx.labels(s["text"]) for u in uids for s in gens[("agent", u)]["report"]}
    cx.save()

    rows, ref = [], Y["reference"]
    for name in SYSTEMS:
        m14, lo14, hi14 = bootstrap_ci(ref, Y[name], "micro_f1", n_boot=args.n_boot)
        m5, lo5, hi5 = bootstrap_ci(ref, Y[name], "micro_f1", labels=TOP5_IDX, n_boot=args.n_boot)
        s14 = f1_scores(ref, Y[name])
        rows.append({"system": name, "n": len(uids),
                     "micro_f1_14": m14, "micro_f1_14_lo": lo14, "micro_f1_14_hi": hi14,
                     "macro_f1_14": s14["macro_f1"], "precision_14": s14["micro_precision"],
                     "recall_14": s14["micro_recall"],
                     "micro_f1_5": m5, "micro_f1_5_lo": lo5, "micro_f1_5_hi": hi5,
                     "normal_abnormal_acc": normal_abnormal_accuracy(ref, Y[name])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(EVAL_DIR / "metrics.csv", index=False)

    diffs = []
    for a, b, meaning in [("agent_final", "nearest_case", "agent vs copying the nearest case"),
                          ("agent_final", "image_only", "agent vs Gemini without retrieval"),
                          ("agent_draft", "image_only", "effect of retrieval on the draft"),
                          ("agent_final", "agent_draft", "effect of look-again + verification")]:
        d, lo, hi = paired_bootstrap_diff(ref, Y[a], Y[b], "micro_f1", n_boot=args.n_boot)
        diffs.append({"comparison": f"{a} - {b}", "meaning": meaning, "diff_micro_f1_14": d, "lo": lo, "hi": hi})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(EVAL_DIR / "paired_differences.csv", index=False)

    agent = [gens[("agent", u)] for u in uids]
    finals = [s for g in agent for s in g["report"]]
    per_sent, consistency, pairs = citation_stats(finals, sentence_labels, case_labels)
    n_draft = sum(len(g["first_draft"]) for g in agent); n_removed = sum(len(g["removed"]) for g in agent)
    agent_stats = {
        "sentences_per_report_draft": n_draft / len(agent),
        "sentences_per_report_final": len(finals) / len(agent),
        "removed_share_of_final_draft": n_removed / max(1, n_removed + len(finals)),
        "reports_needing_revision": float(np.mean([g["revisions"] > 0 for g in agent])),
        "first_pass_supported_mean": float(np.mean([g["first_pass_supported"] for g in agent])),
        "reports_with_missed_findings": float(np.mean([bool(g.get("missed_findings")) for g in agent])),
        "missed_findings_added_per_report": float(np.mean([len(g.get("missed_findings", [])) for g in agent])),
        "case_citations_per_sentence": per_sent,
        "citation_consistency": consistency, "citation_pairs": pairs,
        "agent_seconds_median": float(np.median([g["seconds"] for g in agent])),
        "agent_seconds_p90": float(np.percentile([g["seconds"] for g in agent], 90)),
        "image_only_seconds_median": float(np.median([gens[("image_only", u)]["seconds"] for u in uids])),
    }
    (EVAL_DIR / "agent_stats.json").write_text(json.dumps(agent_stats, indent=2))

    per_study = []
    for i, u in enumerate(uids):
        for name in SYSTEMS:
            per_study.append({"uid": u, "system": name, "f1_14": per_study_f1(ref[i], Y[name][i]),
                              "ref_normal": int(ref[i, NO_FINDING_IDX]), "pred_normal": int(Y[name][i, NO_FINDING_IDX]),
                              "ref_labels": ";".join(CHEXPERT_LABELS[j] for j in np.flatnonzero(ref[i])),
                              "pred_labels": ";".join(CHEXPERT_LABELS[j] for j in np.flatnonzero(Y[name][i]))})
    pd.DataFrame(per_study).to_csv(EVAL_DIR / "per_study.csv", index=False)

    lines = [f"# Report evaluation (n = {len(uids)} held-out test studies, sample positions {args.start}-"
             f"{args.start + len(sample) - 1}, model {model}, config {args.config})", "",
             "CheXbert labels (14 CheXpert observations; positive or uncertain = 1) on generated vs radiologist reports. "
             "95% bootstrap CIs over studies.", "",
             "| System | micro-F1 (14) | micro-F1 (5) | macro-F1 (14) | precision | recall | normal/abnormal acc. |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['system']} | {r['micro_f1_14']:.3f} [{r['micro_f1_14_lo']:.3f}–{r['micro_f1_14_hi']:.3f}] | "
                     f"{r['micro_f1_5']:.3f} [{r['micro_f1_5_lo']:.3f}–{r['micro_f1_5_hi']:.3f}] | {r['macro_f1_14']:.3f} | "
                     f"{r['precision_14']:.3f} | {r['recall_14']:.3f} | {r['normal_abnormal_acc']:.3f} |")
    lines += ["", "Paired differences in micro-F1 (14), 95% CI:", ""]
    lines += [f"- {d.meaning}: {d.diff_micro_f1_14:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]" for d in diffs.itertuples()]
    lines += ["", "Agent behaviour:", ""] + [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                                             for k, v in agent_stats.items()]
    summary = "\n".join(lines)
    (EVAL_DIR / "summary.md").write_text(summary + "\n")
    print("\n" + summary)

    if args.push_langfuse:
        from casefinder.agent.tracing import langfuse_client
        client = langfuse_client()
        pushed = 0
        for i, u in enumerate(uids):
            for system, name in [("agent", "agent_final"), ("image_only", "image_only")]:
                trace_id = gens[(system, u)].get("trace_id")
                if not trace_id:
                    continue
                client.create_score(trace_id=trace_id, name="chexbert_f1", value=per_study_f1(ref[i], Y[name][i]),
                                    comment="per-study CheXbert F1 vs radiologist report (eval-v1)")
                client.create_score(trace_id=trace_id, name="normal_abnormal_correct", data_type="BOOLEAN",
                                    value=float(ref[i, NO_FINDING_IDX] == Y[name][i, NO_FINDING_IDX]))
                pushed += 2
        client.flush()
        print(f"\npushed {pushed} scores to Langfuse")


if __name__ == "__main__":
    main()
