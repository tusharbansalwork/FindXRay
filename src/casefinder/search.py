"""In-memory similar-case search over the library built by scripts/build_library.py.

The library is ~2.5k cases x 768 floats (~7.7 MB), so exact search with one
matrix-vector product is faster and simpler than an approximate index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hit:
    rank: int
    similarity: float
    case: dict


class CaseLibrary:
    def __init__(self, embeddings: np.ndarray, cases: list[dict], model: str = "unknown"):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(cases):
            raise ValueError("embeddings must be (N, D) with one row per case")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("library contains a zero vector")
        self.embeddings = embeddings / norms
        self.cases = cases
        self.model = model

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def load(cls, directory: str | Path) -> "CaseLibrary":
        d = Path(directory)
        manifest = json.loads((d / "manifest.json").read_text())
        lib = cls(np.load(d / "embeddings.npy"), json.loads((d / "cases.json").read_text()), manifest["model"])
        if len(lib) != manifest["n_cases"] or lib.dim != manifest["dim"]:
            raise ValueError(f"library files disagree with manifest.json in {d}")
        return lib

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has {q.shape[0]} values, expected {self.dim}")
        if not np.all(np.isfinite(q)):
            raise ValueError("query contains NaN or infinite values")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            raise ValueError("query is a zero vector")
        if not 1 <= k <= len(self):
            raise ValueError(f"k must be between 1 and {len(self)}")
        sims = self.embeddings @ (q / norm)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [Hit(rank=i + 1, similarity=float(sims[j]), case=self.cases[j]) for i, j in enumerate(top)]
