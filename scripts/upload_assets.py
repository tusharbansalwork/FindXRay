"""Upload what the deployed site needs to Hugging Face (run once, and again if they change).

  - artifacts/onnx/rad-dino-224.int8.onnx -> model repo  <user>/findxray-encoder   (the browser downloads it)
  - artifacts/library/*                   -> dataset repo <user>/findxray-case-library, folder library/
                                             (the API server downloads it at start-up)

Needs a Hugging Face token with write access: run `hf auth login` once first.

Usage:
  python scripts/upload_assets.py                 # both
  python scripts/upload_assets.py --only model
  python scripts/upload_assets.py --only library
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "artifacts" / "onnx" / "rad-dino-224.int8.onnx"
LIBRARY_DIR = ROOT / "artifacts" / "library"

MODEL_CARD = """---
license: mit
base_model: microsoft/rad-dino
library_name: onnx
tags: [chest-x-ray, radiology, image-feature-extraction, onnx, int8]
---

# FindXRay encoder: RAD-DINO (224 px), ONNX int8

The image encoder of [FindXRay](https://github.com/tusharbansalwork/FindXRay), which finds similar past
chest X-rays. It runs in the visitor's browser with ONNX Runtime Web.

- Derived from [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) (MIT licence): exported to ONNX
  and compressed with ONNX Runtime dynamic int8 quantisation (330 MB -> 86 MB).
- Input `pixel_values` [1, 3, 224, 224]: grayscale X-ray, short side resized to 518 then 224 (bicubic),
  centre-cropped to 224, scaled to [0, 1], normalised with mean 0.5307 and std 0.2583, repeated over 3 channels.
- Output `embedding` [1, 768]: L2-normalised CLS embedding.
- Quality: cosine 0.996 vs the full-precision model; retrieval on IU X-ray unchanged (P@5 0.417).

Research demo only. Not a medical device and not for diagnosis.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="tush-2ar")
    ap.add_argument("--only", choices=["model", "library"])
    args = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami()["name"])

    if args.only in (None, "model"):
        repo = f"{args.user}/findxray-encoder"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md", repo_id=repo)
        api.upload_file(path_or_fileobj=str(MODEL_FILE), path_in_repo=MODEL_FILE.name, repo_id=repo)
        print(f"model:   https://huggingface.co/{repo}/resolve/main/{MODEL_FILE.name}")

    if args.only in (None, "library"):
        repo = f"{args.user}/findxray-case-library"
        files = ["embeddings.npy", "cases.json", "manifest.json"]
        missing = [f for f in files if not (LIBRARY_DIR / f).exists()]
        if missing:
            raise SystemExit(f"missing {missing} in {LIBRARY_DIR}: run scripts/build_library.py first")
        api.upload_folder(folder_path=str(LIBRARY_DIR), path_in_repo="library", repo_id=repo, repo_type="dataset",
                          allow_patterns=files, commit_message="Search library for the FindXRay API")
        print(f"library: https://huggingface.co/datasets/{repo}/tree/main/library")


if __name__ == "__main__":
    main()
