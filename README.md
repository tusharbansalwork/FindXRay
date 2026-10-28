# FindXRay

Upload a chest X-ray and see the most similar past cases, what they were diagnosed with, and a draft report that cites them.

**Research demo. Not a medical device and not for diagnosis.**

## Status
- [x] Retrieval benchmark: encoder comparison on IU X-ray (notebook 01)
- [x] Browser speed test of the int8 encoder (~0.5 s per X-ray, multi-threaded WASM)
- [x] Search API (reproduces the benchmark exactly through HTTP; ~2 ms per search)
- [x] Case images for the results (unmodified copies on Hugging Face)
- [x] LangGraph report agent + Langfuse tracing (draft, second look, verify, revise; median 16 s per report on gemini-3.5-flash-lite)
- [x] Report-quality evaluation: 50 held-out studies scored against radiologists' reports (CheXbert)
- [ ] Frontend + deployment

## Results
### Finding similar cases
720 held-out test X-rays searched against a 2,517-case library (IU X-ray). Mean with 95% bootstrap CI.

| Encoder | P@5 | P@5 (abnormal only) | mAP@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| Random | 0.246 | 0.174 | 0.125 | 0.186 | 0.444 |
| DINOv2-base (general) | 0.334 | 0.208 | 0.206 | 0.262 | 0.505 |
| BiomedCLIP | 0.374 | 0.278 | 0.237 | 0.278 | 0.542 |
| **RAD-DINO 224, chosen** | **0.417** [0.396–0.443] | **0.326** [0.298–0.355] | **0.284** | **0.310** | **0.585** |
| RAD-DINO 518 | 0.428 [0.407–0.453] | 0.344 [0.317–0.373] | 0.292 | 0.318 | 0.617 |

Compressed to int8 for the browser: 86 MB (from 330 MB), no measurable loss in retrieval quality. Runs in the browser in ~0.5 s per X-ray (median 484 ms, multi-threaded WebAssembly), so the X-ray is encoded on the visitor's device. Full reasoning and caveats: [DECISIONS.md](DECISIONS.md).

### Drafting the report
50 held-out test X-rays (never used for tuning). Each system's report and the radiologist's report are labelled by
CheXbert (14 findings); F1 measures how well the findings agree. Mean with 95% bootstrap CI.

| System | micro-F1 | Normal/abnormal correct |
|---|---|---|
| Copy the most similar case's report | 0.389 [0.282–0.487] | 74% |
| Gemini, no similar cases | 0.374 [0.262–0.485] | 54% |
| **FindXRay agent** | **0.496** [0.381–0.589] | **78%** |

- The agent beats copying the nearest case by +0.107 F1 [+0.010, +0.204] and the same model without retrieval by +0.122 [+0.003, +0.241].
- The checking steps (second look, sentence verifier) did not measurably change accuracy (+0.005): the gain comes from drafting with similar cases. See D14.
- Median 16 s per report (p90 26 s) on `gemini-3.5-flash-lite`.

## Layout
```
src/casefinder/      metrics.py (retrieval metrics), labels.py (MeSH -> finding groups),
                     search.py (in-memory case library), api.py (FastAPI: /search, /report, /health)
src/casefinder/agent LangGraph report agent: draft -> look again -> amend -> verify -> revise, Gemini + Langfuse
scripts/             build_library.py (builds artifacts/library), check_api_against_benchmark.py,
                     draft_report.py (try the agent on a test X-ray), push_prompts.py (prompts -> Langfuse)
tests/               unit tests for the above
notebooks/           Kaggle notebooks; build_*.py regenerates them from tested code
results/             metric CSVs from each notebook run
data/splits.csv      fixed train/val/test split
artifacts/           embeddings and ONNX models (not in git)
tools/browser_bench/ in-browser encoder speed test
```

## Run the tests
```
pip install -e ".[dev]"                         # needs Python 3.11+ (Langfuse 5 requires it)
pytest
```

## Run the search API locally
```
python scripts/build_library.py                 # once: builds artifacts/library from data/splits.csv + embeddings
uvicorn casefinder.api:app --app-dir src        # http://127.0.0.1:8000/docs
python scripts/check_api_against_benchmark.py   # optional: all 720 test queries through HTTP vs notebook 01
```

## Try the report agent
```
copy .env.example .env                          # then fill in GEMINI_API_KEY and the LANGFUSE_* keys
python scripts/push_prompts.py                  # once, and after editing prompts: versions them in Langfuse
python scripts/draft_report.py                  # drafts a report for a random held-out test X-ray
uvicorn casefinder.api:app --app-dir src --env-file .env   # serves /search and /report
```
Drafting sends the uploaded X-ray to Google's Gemini API. Langfuse traces record prompts, outputs and timings,
but the image is replaced by a placeholder before upload. Use public or sample X-rays only, never real patient data.

## Evaluate report quality
```
pip install -e ".[eval]"                        # PyTorch + CheXbert scorer (CPU is fine)
python scripts/eval_generate.py --start 11 --n 60              # final set: sample positions 11-60 (1-10 = tuning)
python scripts/eval_score.py --start 11 --end 60 --push-langfuse # CheXbert F1 vs radiologists, CIs, Langfuse scores
```
Compares four systems on the same held-out studies: copying the nearest case's report, Gemini without
retrieval, the agent's first draft, and the agent's verified report. Results land in `results/02_report_eval/`.

## Data and model credits
- Indiana University chest X-ray collection via Open-i (U.S. National Library of Medicine), CC BY-NC-ND 4.0. Library images re-shared unmodified at [tush-2ar/findxray-case-library](https://huggingface.co/datasets/tush-2ar/findxray-case-library).
- RAD-DINO (Microsoft), MIT licence. BiomedCLIP (Microsoft). DINOv2 (Meta).
- CheXbert labeller (Smit et al., 2020; Stanford AIMI weights, MIT) via `f1chexbert` for report evaluation.
