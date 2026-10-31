"""Build the AWS Lambda deployment package: build/findxray-lambda.zip.

Contents: the API's Python dependencies as Linux arm64 wheels (downloaded for Lambda's platform, so this
works the same on Windows, macOS or Linux), the casefinder package, the case library (embeddings + cases)
and run.sh, which the AWS Lambda Web Adapter layer starts. boto3 is not packaged: the Lambda runtime has it.

Usage:
  python scripts/build_lambda.py                  # library from artifacts/library (scripts/build_library.py)
  python scripts/build_lambda.py --library-from-hf  # library downloaded from the Hugging Face dataset (CI)
  python scripts/build_lambda.py --arch x86_64      # only to test the package on an x86 Linux machine
"""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PKG = BUILD / "lambda"
ZIP = BUILD / "findxray-lambda.zip"
LIBRARY_REPO = "tush-2ar/findxray-case-library"
LIBRARY_FILES = ["embeddings.npy", "cases.json", "manifest.json"]

PYTHON = "3.12"   # must match the Lambda runtime in deploy/aws/main.tf
PLATFORMS = {   # arm64 (Graviton) is what the Lambda uses; x86_64 is only for testing the package locally
    "arm64": ["manylinux_2_28_aarch64", "manylinux2014_aarch64"],
    "x86_64": ["manylinux_2_28_x86_64", "manylinux2014_x86_64"],
}
DEPENDENCIES = ["fastapi", "uvicorn", "numpy", "tzdata", "langgraph", "langchain-core",
                "langchain-google-genai", "langfuse", "python-dotenv"]

# Started by the Lambda Web Adapter (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap); it forwards HTTP to this port.
RUN_SH = """#!/bin/bash
exec python -m uvicorn casefinder.api:app --host 127.0.0.1 --port "${PORT:-8000}" --log-level warning
"""

# Direct-upload limit for a zip is 50 MB; unzipped code + layers must stay under 250 MB.
MAX_ZIP_MB, MAX_UNZIPPED_MB = 50, 250


def library_dir(from_hf: bool) -> Path:
    if from_hf:
        from huggingface_hub import snapshot_download
        root = snapshot_download(LIBRARY_REPO, repo_type="dataset", allow_patterns=["library/*"],
                                 local_dir=BUILD / "hf_library")
        return Path(root) / "library"
    path = ROOT / "artifacts" / "library"
    missing = [f for f in LIBRARY_FILES if not (path / f).exists()]
    if missing:
        sys.exit(f"missing {missing} in {path}: run scripts/build_library.py, or use --library-from-hf")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-from-hf", action="store_true")
    ap.add_argument("--arch", choices=sorted(PLATFORMS), default="arm64")
    args = ap.parse_args()

    shutil.rmtree(PKG, ignore_errors=True)
    PKG.mkdir(parents=True)
    pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
           "--target", str(PKG), "--implementation", "cp", "--python-version", PYTHON,
           "--only-binary=:all:", "-c", str(ROOT / "deploy" / "constraints.txt")]
    for p in PLATFORMS[args.arch]:
        pip += ["--platform", p]
    print(f"downloading Linux {args.arch} dependencies...")
    subprocess.run(pip + DEPENDENCIES, check=True)

    shutil.copytree(ROOT / "src" / "casefinder", PKG / "casefinder",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    lib = library_dir(args.library_from_hf)
    (PKG / "library").mkdir()
    for f in LIBRARY_FILES:
        shutil.copy2(lib / f, PKG / "library" / f)
    for d in list(PKG.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)

    # Deterministic zip (sorted names, fixed timestamps) so an unchanged build has the same hash.
    # run.sh must be executable inside the zip, also when built on Windows.
    ZIP.unlink(missing_ok=True)
    unzipped = 0
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        info = zipfile.ZipInfo("run.sh", date_time=(2020, 1, 1, 0, 0, 0))
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        z.writestr(info, RUN_SH, compress_type=zipfile.ZIP_DEFLATED)
        for f in sorted(p for p in PKG.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f.relative_to(PKG).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            data = f.read_bytes()
            unzipped += len(data)
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)

    zip_mb, raw_mb = ZIP.stat().st_size / 2**20, unzipped / 2**20
    print(f"wrote {ZIP.relative_to(ROOT)}: {zip_mb:.1f} MB zipped, {raw_mb:.0f} MB unzipped")
    if zip_mb > MAX_ZIP_MB or raw_mb > MAX_UNZIPPED_MB:
        sys.exit(f"too large for a direct Lambda upload (limits: {MAX_ZIP_MB} MB zipped, {MAX_UNZIPPED_MB} MB unzipped)")


if __name__ == "__main__":
    main()
