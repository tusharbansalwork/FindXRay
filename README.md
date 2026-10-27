# GoogleForXrays (working name)

Upload a chest X-ray and see the most similar past cases, what they were diagnosed with, and a draft report that cites them.

**Research demo. Not a medical device and not for diagnosis.**

## Status
- [x] Retrieval benchmark: encoder comparison on IU X-ray (notebook 01)
- [x] Browser speed test of the int8 encoder (~0.5 s per X-ray, multi-threaded WASM)
- [ ] Search API + LangGraph report agent + Langfuse tracing
- [ ] Report-quality evaluation
- [ ] Frontend + deployment

## Results so far
720 held-out test X-rays searched against a 2,517-case library (IU X-ray). Mean with 95% bootstrap CI.

| Encoder | P@5 | P@5 (abnormal only) | mAP@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| Random | 0.246 | 0.174 | 0.125 | 0.186 | 0.444 |
| DINOv2-base (general) | 0.334 | 0.208 | 0.206 | 0.262 | 0.505 |
| BiomedCLIP | 0.374 | 0.278 | 0.237 | 0.278 | 0.542 |
| **RAD-DINO 224, chosen** | **0.417** [0.396–0.443] | **0.326** [0.298–0.355] | **0.284** | **0.310** | **0.585** |
| RAD-DINO 518 | 0.428 [0.407–0.453] | 0.344 [0.317–0.373] | 0.292 | 0.318 | 0.617 |

Compressed to int8 for the browser: 86 MB (from 330 MB), no measurable loss in retrieval quality. Runs in the browser in ~0.5 s per X-ray (median 484 ms, multi-threaded WebAssembly), so the X-ray is encoded on the visitor's device. Full reasoning and caveats: [DECISIONS.md](DECISIONS.md).

## Layout
```
src/casefinder/      metrics.py (retrieval metrics), labels.py (MeSH -> finding groups)
tests/               unit tests for the above
notebooks/           Kaggle notebooks; build_*.py regenerates them from tested code
results/             metric CSVs from each notebook run
data/splits.csv      fixed train/val/test split
artifacts/           embeddings and ONNX models (not in git)
tools/browser_bench/ in-browser encoder speed test
```

## Run the tests
```
pip install -e ".[dev]"
pytest
```

## Data and model credits
- Indiana University chest X-ray collection via Open-i (U.S. National Library of Medicine), CC BY-NC-ND 4.0.
- RAD-DINO (Microsoft), MIT licence. BiomedCLIP (Microsoft). DINOv2 (Meta).
