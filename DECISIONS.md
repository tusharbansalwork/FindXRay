# Decision log

Each entry: what was decided, why, and the evidence. Newest last.

## D1 · Dataset: IU X-ray (Open-i), not NIH ChestX-ray14 or MIMIC-CXR — 2026-10-27
- **NIH ChestX-ray14 is out as the main benchmark.** RAD-DINO (the strongest open chest X-ray encoder) was pretrained on all 112,120 NIH images (its published `training_images.csv` lists them). Measuring retrieval on NIH would test images the model has already seen.
- **MIMIC-CXR is out.** PhysioNet's data use agreement forbids sending credentialed data to third-party APIs or online services, and the report-drafting step uses the Gemini API.
- **IU X-ray is in.** 3,851 studies with radiologist reports and manually assigned MeSH terms; licence CC BY-NC-ND 4.0 (non-commercial use with attribution, images shown unmodified). None of RAD-DINO's 882,775 training images come from IU (checked against `training_images.csv`: only BRAX, CheXpert, MIMIC-CXR, NIH, PadChest). Source used: Hugging Face `ykumards/open-i` (full-resolution PNGs).

## D2 · Labels: 18 finding groups from MeSH — 2026-10-27
- IU's ~120 MeSH terms mix findings with anatomy ("Lung/hypoinflation", "Aorta/tortuous"), so raw-term matching would call every "Lung" case similar. `src/casefinder/labels.py` maps term + qualifier to 18 groups.
- Judgement calls: pneumonia and consolidation merged into Lung Opacity (30–40 cases each); calcinosis grouped with granulomas; spine degeneration and curvature share a group.
- 92 studies with no usable annotation ("No Indexing" only) are dropped; 82 annotated-but-unmapped studies become "Other Abnormal".

## D3 · Split: fixed 70/10/20, stratified by normal/abnormal, seed 42 — 2026-10-27
- 3,597 usable studies with a frontal image → 2,517 train (search library) / 360 val / 720 test (queries). Saved as `data/splits.csv` and reused everywhere.
- IU releases no patient ID, so each study is treated as one patient. Leakage checks: 0 exact duplicate images between test and train; the most similar test→train pair (cosine 0.75) is in line with the different-patient baseline (train→train max 0.70, p99 0.662 vs test→train p99 0.655).

## D4 · Metrics and definitions — 2026-10-27
- A library case is relevant if it shares at least one finding group with the query ("No Finding" counts as a group). nDCG uses Jaccard label overlap as graded gain.
- Headline metrics: P@k, mAP@10, nDCG@10, MRR, each with a 95% bootstrap CI, on all queries and on abnormal queries only.
- R@k (share of all relevant cases found) is reported but not headlined: a typical query has ~25% of the 2.5k-case library relevant, so R@10 can never be high. Hit@k is also weak here (random already scores Hit@5 = 0.69).

## D5 · Encoder: RAD-DINO at 224 px, int8 ONNX — 2026-10-27
Evidence: notebook 01, 720 test queries.

| Model | P@5 all | P@5 abnormal | mAP@10 | MRR |
|---|---|---|---|---|
| random | 0.246 | 0.174 | 0.125 | 0.444 |
| DINOv2-base | 0.334 | 0.208 | 0.206 | 0.505 |
| BiomedCLIP | 0.374 | 0.278 | 0.237 | 0.542 |
| RAD-DINO 224 | 0.417 | 0.326 | 0.284 | 0.585 |
| RAD-DINO 518 | 0.428 | 0.344 | 0.292 | 0.617 |

- 224 px and 518 px have overlapping 95% CIs on every metric; 224 px is ~7× faster on CPU (int8: 172 ms vs 1,260 ms; 1 thread: 340 ms vs 2,726 ms).
- int8 compression: 330 MB → 86 MB; embedding cosine vs fp32 mean 0.996 (min 0.988); retrieval unchanged (P@5 0.4172 → 0.4175, mAP@10 0.2843 → 0.2843).
- Known limitation: whole-image embeddings capture global findings (cardiomegaly 3.9× over base rate, hyperinflation 3.8×, low lung volume 3.6×) but not small local ones (nodule/mass at base rate; devices and fractures weak). The report-drafting step must not infer local findings from neighbours.
- Caveat: the model was chosen by looking at test results. The gaps are large enough that this does not change the choice; later notebooks select on val and report test only for the final system.

## D6 · Hosting: encoder runs in the visitor's browser — 2026-10-27
- Hugging Face removed free Docker/Gradio Spaces (mid-2026). Render's free instance (512 MB RAM, 0.1 CPU) cannot run the encoder server-side, and its 750 free instance-hours per workspace are shared with SeeCite.
- So: the int8 encoder runs in the browser (ONNX Runtime Web); the Render backend only does the vector search over ~2.5k stored embeddings, the LangGraph agent and the Gemini calls.

## D7 · Browser runtime: int8 model on multi-threaded WASM — 2026-10-27
Evidence: `tools/browser_bench/` on Tushar's laptop, ONNX Runtime Web 1.30, RAD-DINO 224, 10 timed runs each.

| Setup | Download | Median | p90 |
|---|---|---|---|
| WASM, all threads, int8 | 86 MB | 484 ms | 526 ms |
| WASM, 1 thread, int8 | 86 MB | 1,609 ms | 1,626 ms |
| WebGPU, int8 | 86 MB | 677 ms | 689 ms |
| WebGPU, fp32 | 330 MB | 371 ms | 389 ms |

- Chosen: int8 on multi-threaded WASM. Half a second per X-ray, and it runs on any modern browser without needing a GPU.
- WebGPU fp32 is ~110 ms faster but needs a 4× larger download (330 MB), so it is not worth it. WebGPU int8 is *slower* than WASM because the int8 matrix ops are not GPU-native and partly fall back to the CPU.
- Threads matter (3.3× faster than single-threaded), and browsers only allow them on cross-origin-isolated pages. **Deployment requirement:** the frontend host must send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, and every cross-origin resource the page loads (e.g. X-ray images) must be CORS/CORP-enabled.
- Browser overhead vs native: the same model took 172 ms in native ONNX Runtime on Kaggle's 4-core CPU, about 2.8× faster than in the browser.
