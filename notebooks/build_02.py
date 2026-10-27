"""Builds notebooks/02_upload_library_images.ipynb."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# 02 · Host the library X-rays on Hugging Face

Uploads the frontal X-ray of every **train-split** case (the 2,517 cases the search returns) to a public Hugging Face dataset, **byte-for-byte unmodified**, so the app can show them. The licence (CC BY-NC-ND 4.0) allows sharing unmodified copies non-commercially with credit; the dataset card added below gives that credit.

**Before running:**
1. Hugging Face → Settings → Access Tokens → create a token with **Write** access.
2. In this Kaggle notebook: Add-ons → Secrets → add it with the label `HF_TOKEN` and tick it for this notebook.
3. Settings: Accelerator **None** (no GPU needed), Internet **On**.
4. Set `HF_REPO` in the next cell. If your GitHub repo is private, also upload `data/splits.csv` as a Kaggle dataset and add it as input (the notebook finds it automatically).

Runtime: about 10–20 min, mostly the upload (~0.7 GB). If the upload is interrupted, run the upload cell again; it resumes.

**When it finishes:** send back the output of the last cell."""
    ),
    code(
        """HF_REPO = "YOUR-HF-USERNAME/findxray-case-library"   # <- change this
SPLITS_URL = "https://raw.githubusercontent.com/tusharbansalwork/FindXRay/main/data/splits.csv"
SOURCE_REPO = "ykumards/open-i"
EXPECTED_TRAIN = 2517

assert not HF_REPO.startswith("YOUR-"), "Set HF_REPO first"
"""
    ),
    code(
        """import glob, hashlib, json, random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_TOKEN")
api = HfApi(token=token)
print("logged in to Hugging Face as:", api.whoami()["name"])

local = glob.glob("/kaggle/input/**/splits.csv", recursive=True)
if local:
    splits = pd.read_csv(local[0]); print("splits from Kaggle input:", local[0])
else:
    splits = pd.read_csv(SPLITS_URL); print("splits from GitHub:", SPLITS_URL)
train_uids = set(splits.loc[splits["split"] == "train", "uid"].astype(int))
print(f"{len(splits)} studies in split file, {len(train_uids)} train")
assert len(train_uids) == EXPECTED_TRAIN, "split file does not match notebook 01"
"""
    ),
    md("## 1 · Extract the train images exactly as stored (no decoding or re-encoding)"),
    code(
        """OUT = Path("/kaggle/working/hf_upload")
(OUT / "images").mkdir(parents=True, exist_ok=True)

for old in (OUT / "images").glob("*"):   # clear leftovers from an earlier run
    old.unlink()

# The source stores JPEGs (despite its card saying PNG); detect the real format from the
# file signature and keep the bytes exactly as they are.
SIGNATURES = {b"\\xff\\xd8\\xff": "jpg", b"\\x89PNG\\r\\n\\x1a\\n": "png"}
def file_ext(b):
    for sig, ext in SIGNATURES.items():
        if b.startswith(sig):
            return ext
    raise ValueError(f"unknown image format, first bytes {b[:8].hex()}")

files = sorted(f for f in list_repo_files(SOURCE_REPO, repo_type="dataset") if f.endswith(".parquet"))
rows = []
for f in files:
    pf = pq.ParquetFile(hf_hub_download(SOURCE_REPO, f, repo_type="dataset", token=token))
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["uid", "img_frontal"]).to_pylist():
            uid, b = int(r["uid"]), r["img_frontal"]
            if uid not in train_uids:
                continue
            assert b, f"uid {uid}: no frontal image"
            ext = file_ext(b)
            (OUT / "images" / f"{uid}.{ext}").write_bytes(b)
            rows.append({"uid": uid, "file": f"images/{uid}.{ext}", "format": ext, "bytes": len(b),
                         "sha256": hashlib.sha256(b).hexdigest()})

manifest = pd.DataFrame(rows).sort_values("uid")
manifest.to_csv(OUT / "manifest.csv", index=False)
missing = train_uids - set(manifest["uid"])
FORMATS = sorted(manifest["format"].unique())
print(f"extracted {len(manifest)} images, {manifest['bytes'].sum() / 2**30:.2f} GB; formats {manifest['format'].value_counts().to_dict()}; missing: {len(missing)}")
assert not missing"""
    ),
    md("## 2 · Dataset card (licence and credit)"),
    code(
        """card = f\"\"\"---
license: cc-by-nc-nd-4.0
pretty_name: IU chest X-ray case library (unmodified frontal images)
tags:
- medical
- chest-x-ray
- radiology
size_categories:
- 1K<n<10K
---

# IU chest X-ray case library

Unmodified frontal chest X-ray images ({", ".join(f.upper() for f in FORMATS)}, exactly as stored in the
source) for the {len(manifest)} training-split cases used as the search library of the FindXRay
similar-case search demo. Files are named `images/<uid>.<ext>`, where `uid` is the source study ID.

- **Source:** Indiana University chest X-ray collection, distributed by the U.S. National Library
  of Medicine through Open-i (https://openi.nlm.nih.gov/). Files are taken byte-for-byte from the
  Hugging Face copy [`{SOURCE_REPO}`](https://huggingface.co/datasets/{SOURCE_REPO}).
- **Licence:** CC BY-NC-ND 4.0, inherited from the source. Non-commercial use only; the images
  are shared unmodified.
- **Reference:** Demner-Fushman D, et al. *Preparing a collection of radiology examinations for
  distribution and retrieval.* J Am Med Inform Assoc. 2016;23(2):304-310.
- `manifest.csv` lists each file with its size and SHA-256.

Research demo data. **Not for clinical use.**
\"\"\"
(OUT / "README.md").write_text(card)
print(card)"""
    ),
    md("## 3 · Upload (resumable)"),
    code(
        """api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)
api.upload_large_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(OUT))
print("uploaded to https://huggingface.co/datasets/" + HF_REPO)"""
    ),
    md("## 4 · Check: files arrived intact, and browsers on other sites may load them"),
    code(
        """BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
remote = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset") if f.startswith("images/")]
print(f"files on Hugging Face: {len(remote)} images (expected {len(manifest)})")

random.seed(0)
sample = manifest.sample(5, random_state=0)
intact = 0
for _, r in sample.iterrows():
    resp = requests.get(f"{BASE}/{r['file']}", timeout=60)
    ok = resp.status_code == 200 and hashlib.sha256(resp.content).hexdigest() == r["sha256"]
    intact += ok
    print(f"uid {r['uid']}: HTTP {resp.status_code}, sha256 {'matches' if ok else 'DIFFERS'}")

# What a browser sees when the app (another origin) loads an image with crossorigin="anonymous"
probe = requests.get(f"{BASE}/{sample.iloc[0]['file']}", headers={"Origin": "http://localhost:8000"},
                     allow_redirects=False, timeout=60)
hops = [("first response", probe)]
if probe.is_redirect or probe.status_code in (301, 302, 303, 307, 308):
    nxt = requests.get(requests.compat.urljoin(probe.url, probe.headers["Location"]), headers={"Origin": "http://localhost:8000"}, timeout=60)
    hops.append(("after redirect", nxt))
summary = {
    "hf_repo": HF_REPO,
    "images_extracted": int(len(manifest)),
    "images_on_hf": len(remote),
    "total_GB": round(manifest["bytes"].sum() / 2**30, 3),
    "sample_intact": f"{intact}/5",
    "formats": manifest["format"].value_counts().to_dict(),
    "url_template": BASE + "/images/{uid}." + (FORMATS[0] if len(FORMATS) == 1 else "<ext from manifest.csv>"),
}
for name, r in hops:
    summary[name] = {
        "status": r.status_code,
        "access-control-allow-origin": r.headers.get("access-control-allow-origin"),
        "cross-origin-resource-policy": r.headers.get("cross-origin-resource-policy"),
        "content-type": r.headers.get("content-type"),
        "host": requests.utils.urlparse(r.url).netloc,
    }
print(json.dumps(summary, indent=2))"""
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "02_upload_library_images.ipynb"
nbf.write(nb, out)
print("wrote", out)
