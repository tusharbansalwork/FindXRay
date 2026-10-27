"""Build the search library the API serves: train-split embeddings + case metadata.

Inputs
  data/splits.csv                          fixed split (notebook 01)
  artifacts/embeddings/uids.npy            row order of the embedding file
  artifacts/embeddings/rad-dino-224.npy    RAD-DINO 224 embeddings, one row per study
  Hugging Face dataset ykumards/open-i     report text + MeSH (only these columns are read, not images)

Outputs (artifacts/library/)
  embeddings.npy   float32 (N_train, 768), L2-normalised, same row order as cases.json
  cases.json       one record per library case: uid, groups, indication, findings, impression, mesh
  manifest.json    counts and source info, so the API can sanity-check what it loaded

Usage:  python scripts/build_library.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.labels import mesh_to_groups  # noqa: E402

REPO = "datasets/ykumards/open-i"
COLUMNS = ["uid", "MeSH", "indication", "findings", "impression"]
MODEL = "rad-dino-224"
OUT = ROOT / "artifacts" / "library"


def load_metadata() -> pd.DataFrame:
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{REPO}/data/*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards found in {REPO}")
    tables = [pq.read_table(s, columns=COLUMNS, filesystem=fs) for s in shards]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)


def clean(text) -> str | None:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    text = " ".join(str(text).split())
    return text or None


def main() -> None:
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    uids = np.load(ROOT / "artifacts" / "embeddings" / "uids.npy")
    emb = np.load(ROOT / "artifacts" / "embeddings" / f"{MODEL}.npy").astype(np.float32)
    if len(uids) != len(emb):
        raise SystemExit("uids.npy and embedding rows differ in length")

    meta = load_metadata().set_index("uid")
    train_uids = splits.loc[splits["split"] == "train", "uid"].to_numpy()
    row_of = {int(u): i for i, u in enumerate(uids)}
    missing = [u for u in train_uids if int(u) not in row_of or int(u) not in meta.index]
    if missing:
        raise SystemExit(f"{len(missing)} train uids missing from embeddings/metadata, e.g. {missing[:5]}")

    lib_emb = emb[[row_of[int(u)] for u in train_uids]]
    lib_emb /= np.linalg.norm(lib_emb, axis=1, keepdims=True)

    cases = []
    for u in train_uids:
        m = meta.loc[int(u)]
        groups = mesh_to_groups(m["MeSH"])
        if groups is None:  # cannot happen for split members, but fail loudly if it does
            raise SystemExit(f"uid {u} has no usable labels")
        cases.append({
            "uid": int(u),
            "groups": sorted(groups),
            "indication": clean(m["indication"]),
            "findings": clean(m["findings"]),
            "impression": clean(m["impression"]),
            "mesh": clean(m["MeSH"]),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "embeddings.npy", lib_emb)
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False))
    manifest = {
        "model": MODEL,
        "n_cases": len(cases),
        "dim": int(lib_emb.shape[1]),
        "split": "train",
        "source": "Indiana University chest X-ray collection (Open-i), via Hugging Face ykumards/open-i; CC BY-NC-ND 4.0",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(cases)} cases, embeddings {lib_emb.shape} -> {OUT}")


if __name__ == "__main__":
    main()
