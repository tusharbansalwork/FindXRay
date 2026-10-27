"""Builds notebooks/01_encoder_benchmark.ipynb from the cells below.

The metric and label cells are copied verbatim from src/casefinder so the
notebook uses exactly the code covered by tests/.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def module_source(rel: str) -> str:
    text = (ROOT / rel).read_text()
    return text.replace("from __future__ import annotations\n", "")


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# 01 · Which X-ray model finds similar cases best, and can it run in a browser?

**Settings:** Accelerator **GPU T4 ×1**, Internet **On**. Expected runtime: about 45–60 min.

What this notebook does:
1. Loads the Indiana University chest X-ray set (Open-i, 3,851 studies with radiologist reports and MeSH annotations) from Hugging Face.
2. Maps MeSH annotations to 18 finding groups and makes a fixed 70/10/20 train/val/test split (this split is reused for the whole project).
3. Embeds every frontal X-ray with 4 encoders plus a random baseline: DINOv2-base (general-purpose), BiomedCLIP, RAD-DINO at 224 px, RAD-DINO at 518 px (its native size).
4. Uses **test** X-rays as queries against the **train** library and reports P@k, Hit@k, R@k, mAP@k, nDCG@k and MRR with 95% bootstrap CIs, overall and on abnormal queries only, plus per-finding P@5.
5. Checks for duplicate and near-duplicate images across splits.
6. Exports the medical encoders to ONNX, compresses them to int8, and measures file size, CPU latency, and how much retrieval quality the compression costs — that decides whether the model can ship to the browser.

**When it finishes:** send back the output of the last cell, and download from the Output tab: `splits.csv`, `results_*.csv`, `embeddings/`, `onnx/*.int8.onnx`."""
    ),
    code("%pip install -q open_clip_torch onnx onnxscript onnxruntime"),
    code(
        """import io, os, time, random, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
(OUT / "embeddings").mkdir(parents=True, exist_ok=True)
(OUT / "onnx").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KS = (1, 5, 10)

# Browser check settings
EXPORT_MODELS = ["biomedclip", "rad-dino-224", "rad-dino-518"]
INT8_MAX_QUERIES = {"rad-dino-518": 200}   # 518 px is slow on CPU; others use all test queries
N_FIDELITY = 100

print("device:", DEVICE, "| torch", torch.__version__, "| CPUs:", os.cpu_count())
if DEVICE == "cpu":
    print("WARNING: no GPU — turn on GPU T4 in the notebook settings, or this will be very slow.")"""
    ),
    md("## Metric code (identical to `src/casefinder/metrics.py`, covered by unit tests)"),
    code(module_source("src/casefinder/metrics.py")),
    md("## Label mapping (identical to `src/casefinder/labels.py`)"),
    code(module_source("src/casefinder/labels.py")),
    md("## 1 · Load the IU X-ray data"),
    code(
        """from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

REPO = "ykumards/open-i"
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
paths = [hf_hub_download(REPO, f, repo_type="dataset") for f in files]
print(len(paths), "parquet shards")

STORE_SHORT = 518  # keep images at 518 px on the short side (the largest size any encoder needs)

def resize_short(img, s):
    w, h = img.size
    r = s / min(w, h)
    return img.resize((max(s, int(round(w * r))), max(s, int(round(h * r)))), Image.BICUBIC)

records, images, missing = [], {}, 0
for p in tqdm(paths, desc="shards"):
    pf = pq.ParquetFile(p)
    for rg in range(pf.num_row_groups):
        rows = pf.read_row_group(rg, columns=["uid", "MeSH", "findings", "impression", "img_frontal"]).to_pylist()
        for r in rows:
            b = r.pop("img_frontal")
            if not b:
                missing += 1
                continue
            img = Image.open(io.BytesIO(b)).convert("L")
            r["orig_w"], r["orig_h"] = img.size
            r["md5"] = hashlib.md5(b).hexdigest()
            images[r["uid"]] = resize_short(img, STORE_SHORT)
            records.append(r)

meta = pd.DataFrame(records)
print(f"{len(meta)} studies with a frontal image, {missing} without")"""
    ),
    md("## 2 · Labels and the fixed split"),
    code(
        """from sklearn.model_selection import train_test_split

meta["groups"] = meta["MeSH"].map(mesh_to_groups)
n_unusable = int(meta["groups"].isna().sum())
meta = meta[meta["groups"].notna()].reset_index(drop=True)
meta["is_normal"] = meta["groups"].map(lambda g: g == {NO_FINDING})
print(f"dropped {n_unusable} studies with no usable annotation -> {len(meta)} studies")

# IU X-ray releases no patient ID; each study is treated as one patient (as in published work).
# The duplicate checks below look for leakage that this assumption could hide.
train_uids, rest = train_test_split(meta["uid"], test_size=0.30, random_state=SEED, stratify=meta["is_normal"])
rest_normal = meta.set_index("uid").loc[rest, "is_normal"]
val_uids, test_uids = train_test_split(rest, test_size=2/3, random_state=SEED, stratify=rest_normal)
split_of = {**{u: "train" for u in train_uids}, **{u: "val" for u in val_uids}, **{u: "test" for u in test_uids}}
meta["split"] = meta["uid"].map(split_of)
meta[["uid", "split"]].to_csv(OUT / "splits.csv", index=False)

Y = to_multihot(list(meta["groups"]))
counts = pd.DataFrame(Y, columns=GROUPS).groupby(meta["split"]).sum().T
counts["total"] = counts.sum(1)
print(meta["split"].value_counts().to_dict())
counts.sort_values("total", ascending=False)"""
    ),
    code(
        """# Exact duplicate images (same file bytes) across splits
dup = meta[meta.duplicated("md5", keep=False)].sort_values("md5")
print(f"{dup['md5'].nunique()} duplicated images across {len(dup)} studies")
train_md5 = set(meta.loc[meta.split == "train", "md5"])
leaky_test = set(meta.loc[(meta.split == "test") & meta["md5"].isin(train_md5), "uid"])
print(f"{len(leaky_test)} test studies have an exact duplicate in train -> removed from queries")

lib_idx = np.where(meta["split"] == "train")[0]
q_idx = np.where((meta["split"] == "test") & ~meta["uid"].isin(leaky_test))[0]
print(f"library = {len(lib_idx)} train X-rays, queries = {len(q_idx)} test X-rays")"""
    ),
    md("## 3 · Encoders"),
    code(
        """IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
RAD = ((0.5307,) * 3, (0.2583,) * 3)   # from microsoft/rad-dino preprocessor_config.json

def preprocess(img_l, resize_to, crop, mean, std):
    \"\"\"Grayscale PIL -> (3, crop, crop) float32. Written by hand (no HF processor) so the
    browser can reproduce it exactly: resize short side, center crop, scale, normalise.\"\"\"
    im = img_l if min(img_l.size) == resize_to else resize_short(img_l, resize_to)
    w, h = im.size
    l, t = (w - crop) // 2, (h - crop) // 2
    a = np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.float32) / 255.0
    a = np.repeat(a[None], 3, axis=0)
    return (a - np.array(mean, np.float32)[:, None, None]) / np.array(std, np.float32)[:, None, None]

class HFEmbedder(nn.Module):
    \"\"\"DINOv2-family model -> L2-normalised CLS embedding (pooler_output).\"\"\"
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name)
    def forward(self, pixel_values):
        return F.normalize(self.m(pixel_values=pixel_values).pooler_output, dim=-1)

class BiomedClipEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        import open_clip
        model, _ = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.v = model.visual
    def forward(self, pixel_values):
        return F.normalize(self.v(pixel_values), dim=-1)

SPECS = {
    "dinov2-base":  dict(build=lambda: HFEmbedder("facebook/dinov2-base"), resize=256, crop=224, norm=IMAGENET),
    "biomedclip":   dict(build=BiomedClipEmbedder,                         resize=224, crop=224, norm=CLIP),
    "rad-dino-224": dict(build=lambda: HFEmbedder("microsoft/rad-dino"),   resize=224, crop=224, norm=RAD),
    "rad-dino-518": dict(build=lambda: HFEmbedder("microsoft/rad-dino"),   resize=518, crop=518, norm=RAD),
}

def batch_pixels(spec, uids):
    return np.stack([preprocess(images[u], spec["resize"], spec["crop"], *spec["norm"]) for u in uids])

@torch.no_grad()
def embed_all(model, spec, uids, bs=32):
    out = []
    for i in tqdm(range(0, len(uids), bs), leave=False):
        x = torch.from_numpy(batch_pixels(spec, uids[i:i + bs])).to(DEVICE)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)

uids = meta["uid"].tolist()
np.save(OUT / "embeddings" / "uids.npy", np.array(uids))
EMB = {}
rng = np.random.default_rng(SEED)
r = rng.standard_normal((len(uids), 768)).astype(np.float32)
EMB["random"] = r / np.linalg.norm(r, axis=1, keepdims=True)

for name, spec in SPECS.items():
    t0 = time.time()
    model = spec["build"]().eval().to(DEVICE)
    EMB[name] = embed_all(model, spec, uids)
    np.save(OUT / "embeddings" / f"{name}.npy", EMB[name])
    print(f"{name:13s} dim={EMB[name].shape[1]}  {time.time() - t0:5.0f}s")
    del model; torch.cuda.empty_cache()"""
    ),
    md("## 4 · Retrieval benchmark (test queries vs. train library)"),
    code(
        """rel, gain = relevance(Y[q_idx], Y[lib_idx])
abnormal_q = ~meta["is_normal"].values[q_idx]
SLICES = {"all": np.ones(len(q_idx), bool), "abnormal": abnormal_q}
print(f"queries: {len(q_idx)} ({abnormal_q.sum()} abnormal); "
      f"avg relevant cases per query: {rel.sum(1).mean():.0f} of {len(lib_idx)}")

rows, per_query, topk = [], {}, {}
for name, E in EMB.items():
    sim = E[q_idx] @ E[lib_idx].T
    pq_ = retrieval_metrics(sim, rel, gain, KS)
    per_query[name] = pq_
    topk[name] = np.argsort(-sim, axis=1, kind="stable")[:, :10]
    for sl, mask in SLICES.items():
        for metric, (m, lo, hi) in summarize({k: v[mask] for k, v in pq_.items()}).items():
            rows.append(dict(model=name, slice=sl, metric=metric, mean=m, lo=lo, hi=hi))
res = pd.DataFrame(rows)
res.to_csv(OUT / "results_retrieval.csv", index=False)

KEY = ["P@1", "P@5", "Hit@5", "R@10", "mAP@10", "nDCG@10", "MRR"]
def table(sl, with_ci=True):
    d = res[(res.slice == sl) & res.metric.isin(KEY)].copy()
    d["v"] = d.apply(lambda r: f"{r['mean']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}]" if with_ci else f"{r['mean']:.3f}", axis=1)
    return d.pivot(index="model", columns="metric", values="v")[KEY].loc[list(EMB)]

print("ALL QUERIES"); display(table("all"))
print("ABNORMAL QUERIES ONLY"); display(table("abnormal"))"""
    ),
    code(
        """# Per-finding P@5: for queries that have finding c, the share of the top-5 cases that also have c.
# 'library base rate' is what a random ranking would score.
lib_Y = Y[lib_idx]
rows = []
for c, g in enumerate(GROUPS):
    has = Y[q_idx][:, c]
    if has.sum() < 5:
        continue
    row = {"finding": g, "n_queries": int(has.sum()), "library base rate": lib_Y[:, c].mean()}
    for name in EMB:
        row[name] = lib_Y[topk[name][has, :5], c].mean()
    rows.append(row)
per_class = pd.DataFrame(rows).set_index("finding").sort_values("n_queries", ascending=False)
per_class.to_csv(OUT / "results_per_finding.csv")
per_class.style.format("{:.3f}", subset=[c for c in per_class.columns if c != "n_queries"])"""
    ),
    md("## 5 · Near-duplicate audit\nIf a test X-ray is almost identical to a train X-ray (e.g. the same patient scanned twice), retrieval scores are inflated. Look at the most similar cross-split pairs below; if they look like the same person, tell me."),
    code(
        """import matplotlib.pyplot as plt

best = res[(res.slice == "all") & (res.metric == "mAP@10")].set_index("model")["mean"].drop("random").idxmax()
E = EMB[best]
cross = E[q_idx] @ E[lib_idx].T
mx = cross.max(1)
print(f"model: {best}; max test-to-train cosine percentiles:",
      {p: round(float(np.percentile(mx, p)), 4) for p in (50, 90, 99, 100)})
print("baseline: same percentiles of max train-to-train cosine (different studies) for comparison:")
tt = E[lib_idx[:800]] @ E[lib_idx].T
np.fill_diagonal(tt[:, :800], -1)
print({p: round(float(np.percentile(tt.max(1), p)), 4) for p in (50, 90, 99, 100)})

order = np.argsort(-mx)[:6]
fig, ax = plt.subplots(2, 6, figsize=(18, 6))
for j, qi in enumerate(order):
    li = cross[qi].argmax()
    for row, (idx, tag) in enumerate([(q_idx[qi], "test"), (lib_idx[li], "train")]):
        ax[row, j].imshow(images[uids[idx]], cmap="gray"); ax[row, j].axis("off")
        ax[row, j].set_title(f"{tag} uid {uids[idx]}" + (f"\\ncos {mx[qi]:.4f}" if row == 0 else ""), fontsize=9)
plt.tight_layout(); plt.show()"""
    ),
    md("## 6 · Browser check: ONNX export, int8 compression, size, CPU speed, quality cost"),
    code(
        """import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def files_mb(path):
    return sum(f.stat().st_size for f in path.parent.glob(path.name + "*")) / 2**20  # includes .data sidecars

def export_onnx(model, crop, path):
    dummy = torch.zeros(1, 3, crop, crop)
    kw = dict(input_names=["pixel_values"], output_names=["embedding"], opset_version=17)
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kw)
    except TypeError:            # older torch without the dynamo argument
        torch.onnx.export(model, (dummy,), str(path), **kw)
    except Exception as e:       # legacy exporter removed or failed
        print("  legacy exporter failed (", type(e).__name__, e, ") -> trying dynamo exporter")
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, input_names=["pixel_values"], output_names=["embedding"])

def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

def ort_embed(sess, x):
    e = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)

def median_ms(sess, x, n=15):
    for _ in range(3):
        ort_embed(sess, x)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); ort_embed(sess, x); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))

browser_rows = []
for name in EXPORT_MODELS:
    spec = SPECS[name]
    print(f"\\n== {name}")
    model = spec["build"]().eval().float().cpu()
    fp32, int8 = OUT / "onnx" / f"{name}.fp32.onnx", OUT / "onnx" / f"{name}.int8.onnx"
    export_onnx(model, spec["crop"], fp32)
    del model
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    s32, s8 = session(fp32, os.cpu_count()), session(int8, os.cpu_count())
    s8_1 = session(int8, 1)

    x1 = batch_pixels(spec, [uids[q_idx[0]]])
    row = dict(model=name, fp32_MB=files_mb(fp32), int8_MB=files_mb(int8),
               fp32_ms=median_ms(s32, x1), int8_ms=median_ms(s8, x1), int8_ms_1thread=median_ms(s8_1, x1, n=5))

    # Fidelity: torch (GPU) embedding vs ONNX fp32 / int8 on the same test images
    fid_idx = q_idx[:N_FIDELITY]
    e32 = np.concatenate([ort_embed(s32, batch_pixels(spec, [uids[i]])) for i in fid_idx])
    e8 = np.concatenate([ort_embed(s8, batch_pixels(spec, [uids[i]])) for i in fid_idx])
    ref = EMB[name][fid_idx]
    row["cos_torch_vs_onnx32_min"] = float((ref * e32).sum(1).min())
    row["cos_fp32_vs_int8_mean"] = float((ref * e8).sum(1).mean())
    row["cos_fp32_vs_int8_min"] = float((ref * e8).sum(1).min())

    # Retrieval with int8 queries against the fp32 library (the deployed setup:
    # library embedded offline at full precision, query embedded in the browser)
    n_q = INT8_MAX_QUERIES.get(name, len(q_idx))
    sub = np.arange(min(n_q, len(q_idx)))
    q8 = np.concatenate([ort_embed(s8, batch_pixels(spec, [uids[q_idx[i]]])) for i in tqdm(sub, leave=False, desc="int8 queries")])
    m8 = retrieval_metrics(q8 @ EMB[name][lib_idx].T, rel[sub], gain[sub], KS)
    m32 = {k: v[sub] for k, v in per_query[name].items()}
    row["n_queries"] = len(sub)
    for k in ["P@5", "mAP@10", "nDCG@10"]:
        row[f"{k}_fp32"], row[f"{k}_int8"] = float(np.nanmean(m32[k])), float(np.nanmean(m8[k]))
    browser_rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

browser = pd.DataFrame(browser_rows).set_index("model")
browser.to_csv(OUT / "results_browser.csv")
browser.T"""
    ),
    md("## 7 · Summary — copy this cell's output back to Claude"),
    code(
        """pd.set_option("display.width", 250)
print("DATA:", f"{len(meta)} studies, dropped {n_unusable} unusable; split", meta['split'].value_counts().to_dict(),
      f"| exact dup leaks removed: {len(leaky_test)} | queries {len(q_idx)} ({abnormal_q.sum()} abnormal)")
for sl in ("all", "abnormal"):
    print(f"\\n--- retrieval, {sl} queries (mean [95% CI]) ---")
    print(table(sl).to_string())
print("\\n--- per-finding P@5 ---")
print(per_class.round(3).to_string())
print("\\n--- near-duplicate audit ---")
print(f"{best}: max test->train cosine p99={np.percentile(mx, 99):.4f}, max={mx.max():.4f}")
print("\\n--- browser check ---")
print(browser.round(4).T.to_string())"""
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "01_encoder_benchmark.ipynb"
nbf.write(nb, out)
print("wrote", out)
