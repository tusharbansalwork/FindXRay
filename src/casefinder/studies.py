"""Fetch test studies (frontal image + radiologist report) from the source dataset, with a local cache."""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = "datasets/ykumards/open-i"
TEXT_COLUMNS = ["uid", "MeSH", "indication", "findings", "impression"]


def fetch_studies(uids: list[int], cache_dir: str | Path) -> dict[int, dict]:
    """Return {uid: {"image": bytes, "mime_type": str, "MeSH", "indication", "findings", "impression"}}.

    Reads each source row group at most once and only for row groups containing a wanted uid;
    results are cached under cache_dir (images as files, text in studies.json).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "studies.json"
    meta = {int(k): v for k, v in json.loads(meta_path.read_text()).items()} if meta_path.exists() else {}

    def image_path(uid: int, mime: str) -> Path:
        return cache / f"{uid}.{'jpg' if mime == 'image/jpeg' else 'png'}"

    wanted = {u for u in uids if u not in meta or not image_path(u, meta[u]["mime_type"]).exists()}
    if wanted:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        for shard in sorted(fs.glob(f"{SOURCE}/data/*.parquet")):
            if not wanted:
                break
            pf = pq.ParquetFile(shard, filesystem=fs)
            for rg in range(pf.num_row_groups):
                rg_uids = pf.read_row_group(rg, columns=["uid"]).column("uid").to_pylist()
                hits = wanted.intersection(rg_uids)
                if not hits:
                    continue
                rows = pf.read_row_group(rg, columns=TEXT_COLUMNS + ["img_frontal"]).to_pylist()
                for row in rows:
                    uid = int(row["uid"])
                    if uid not in hits:
                        continue
                    data = row.pop("img_frontal")
                    if not data:
                        raise ValueError(f"study {uid} has no frontal image")
                    mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
                    image_path(uid, mime).write_bytes(data)
                    meta[uid] = {k: row[k] for k in TEXT_COLUMNS if k != "uid"} | {"mime_type": mime}
                wanted -= hits
                meta_path.write_text(json.dumps({str(k): v for k, v in meta.items()}))
        if wanted:
            raise ValueError(f"uids not found in {SOURCE}: {sorted(wanted)[:10]}")

    return {u: {**meta[u], "image": image_path(u, meta[u]["mime_type"]).read_bytes()} for u in uids}


def fetch_texts() -> dict[int, dict]:
    """Report text and MeSH for every study (column projection only: a few MB, no images)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out: dict[int, dict] = {}
    for shard in sorted(fs.glob(f"{SOURCE}/data/*.parquet")):
        for row in pq.read_table(shard, columns=TEXT_COLUMNS, filesystem=fs).to_pylist():
            out[int(row.pop("uid"))] = row
    return out


def reference_text(study: dict) -> str:
    """The radiologist's report as one text: findings then impression (either may be missing)."""
    return " ".join(t.strip() for t in (study.get("findings"), study.get("impression")) if t and t.strip())
