# Decision log

Each entry: what was decided, why, and the evidence. Newest last.

## D1 · Dataset: IU X-ray (Open-i), not NIH ChestX-ray14 or MIMIC-CXR — 2026-10-27
- **NIH ChestX-ray14 is out as the main benchmark.** RAD-DINO (the strongest open chest X-ray encoder) was pretrained on all 112,120 NIH images (its published `training_images.csv` lists them). Measuring retrieval on NIH would test images the model has already seen.
- **MIMIC-CXR is out.** PhysioNet's data use agreement forbids sending credentialed data to third-party APIs or online services, and the report-drafting step uses the Gemini API.
- **IU X-ray is in.** 3,851 studies with radiologist reports and manually assigned MeSH terms; licence CC BY-NC-ND 4.0 (non-commercial use with attribution, images shown unmodified). None of RAD-DINO's 882,775 training images come from IU (checked against `training_images.csv`: only BRAX, CheXpert, MIMIC-CXR, NIH, PadChest). Source used: Hugging Face `ykumards/open-i` (full-resolution images, stored as JPEG even though its card says PNG).

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
Evidence: `tools/browser_bench/` on Tushar's laptop (Chrome 155, Windows, 16 logical cores, AMD GCN-5 integrated GPU), ONNX Runtime Web 1.30, RAD-DINO 224, 10 timed runs each.

| Setup | Download | Median | p90 |
|---|---|---|---|
| WASM, all threads, int8 | 86 MB | 484 ms | 526 ms |
| WASM, 1 thread, int8 | 86 MB | 1,609 ms | 1,626 ms |
| WebGPU, int8 | 86 MB | 677 ms | 689 ms |
| WebGPU, fp32 | 330 MB | 371 ms | 389 ms |

- Chosen: int8 on multi-threaded WASM. Half a second per X-ray, and it runs on any modern browser without needing a GPU.
- WebGPU fp32 is ~110 ms faster but needs a 4× larger download (330 MB), so it is not worth it. WebGPU int8 is *slower* than WASM because the int8 matrix ops are not GPU-native and partly fall back to the CPU.
- Threads matter (3.3× faster than single-threaded), and browsers only allow them on cross-origin-isolated pages. **Deployment requirement:** the frontend host must send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, and every cross-origin resource the page loads (e.g. X-ray images) must be CORS/CORP-enabled.
- One-time start-up cost (chosen setup): 1.7 s to create the session + 0.8 s for the first run ≈ 2.5 s, on top of downloading 86 MB (seconds on fast broadband, about a minute at 10 Mbps). Implication for the frontend: start loading the model as soon as the page opens, before the user picks an X-ray, and cache it so repeat visits skip the download.
- ONNX Runtime used 4 of the 16 available threads by default; more threads were not tested.
- Browser overhead vs native: the same model took 172 ms in native ONNX Runtime on Kaggle's 4-core CPU, about 2.8× faster than in the browser.

## D8 · Search backend: exact in-memory search, embedding-only requests — 2026-10-27
- The library is the 2,517 train-split cases (7.7 MB of embeddings + 1.2 MB of report text, built by `scripts/build_library.py`). At this size one matrix-vector product is exact and takes ~2 ms, so no approximate index (FAISS/pgvector) is needed.
- The browser sends only the 768-number embedding, never the image, to `/search`.
- Validation errors return location and message only. FastAPI's default 422 echoes the input back, which crashed on NaN (not valid JSON) and would return 768 numbers; a unit test caught this.
- CORS allows only configured origins (`CASEFINDER_ORIGINS`).
- Evidence: all 720 test embeddings sent through the running HTTP API reproduce notebook 01 exactly (P@5 0.4172, mAP@10 0.2843, abnormal P@5 0.3264, mAP@10 0.2045; differences < 1e-16). Server memory ~77 MB RSS, so it fits Render's free 512 MB.

## D9 · Case images: unmodified copies on a public Hugging Face dataset — 2026-10-27
- The 2,517 library X-rays are hosted at `huggingface.co/datasets/tush-2ar/findxray-case-library` (0.73 GB), uploaded by notebook 02 byte-for-byte from the source, with a dataset card carrying the CC BY-NC-ND 4.0 licence and credit. Linking to Open-i was rejected: it could not be checked from the build environment, and the dataset copy has no Open-i filenames.
- The source stores JPEGs, although its card says PNG; files keep their real format (`images/<uid>.jpg`). Verified: 2,517/2,517 files present, 5/5 random downloads identical by SHA-256.
- Browser access: Hugging Face answers with CORS headers (`Access-Control-Allow-Origin` reflected, then `*` from the CDN) but no `Cross-Origin-Resource-Policy`. Tested in headless Chromium on a page with the app's COOP/COEP headers: `<img crossorigin="anonymous">` loads (2048×2048); a plain `<img>` is blocked (`ERR_BLOCKED_BY_RESPONSE.NotSameOriginAfterDefaultedToSameOriginByCoep`). **Frontend rule: always load case images with `crossorigin="anonymous"`.**
- The API returns `image_url` per result from a configurable template (`CASEFINDER_IMAGE_URL`).

## D10 · Report agent: draft, verify, revise with citations — 2026-10-27
- Gemini sees the uploaded X-ray (option A), but only when the user asks for a draft; search stays on-device. Without the image the draft could only paraphrase neighbours, and D5 showed neighbours are unreliable for small local findings. The UI must say that drafting sends the image to Google, and to use public or sample X-rays only. (Langfuse does not receive it; see D11.)
- LangGraph flow: retrieve (k=5) → draft → check → revise (at most 2 rounds) → result. Every sentence carries sources: `image` and/or `case:<uid>`.
- A sentence passes only if (a) a deterministic check confirms every source is the image or a retrieved case, and (b) the model's sentence-by-sentence verifier approves it. Sentences still failing after 2 revisions are removed from the report and returned separately with the reason, so nothing unverified is presented as findings.
- The verifier is the same model family as the drafter (LLM-as-judge); its verdicts are not ground truth. The report-quality evaluation (next) measures against the radiologists' actual reports instead.
- Langfuse: one trace per report (`findxray-report`), every node, prompt, response, token count and latency; prompts versioned in Langfuse (`scripts/push_prompts.py`, label `production`) with local fallback, and each generation linked to its prompt version. Tracing failures never break a report: tested with Langfuse unreachable (report still returned, HTTP 200). That test found and fixed a case where building the trace link raised after a successful run.
- Tests: 19 new (53 total) with a scripted stand-in model covering every graph path; three deliberate code mutations were each caught. Not yet run against the real Gemini API.

## D11 · First real Gemini runs: model, latency settings, images kept out of traces — 2026-10-27
- `gemini-2.5-flash` is closed to new API projects (404, "no longer available to new users"); default is now `gemini-3.8-flash`, the model Google's error message names. Overridable with `GEMINI_MODEL`.
- With default settings the first call (draft) to `gemini-3.8-flash` hit a 504 DEADLINE_EXCEEDED at the 90 s deadline on a 2048×2048 X-ray. Now: `thinking_level=low` (LangChain documents it for lower latency), the image sent at `MEDIA_RESOLUTION_MEDIUM`, timeout 180 s. All overridable from `.env`. Verified offline in the request LangChain builds: thinking level LOW, per-image media resolution MEDIUM, timeout 180,000 ms. Lower thinking may cost quality; the report evaluation will show whether to raise it.
- Only LangChain's v1 image block (`{"type": "image", "base64": …}`) passes `media_resolution` to Gemini; the `image_url`/data-URI form silently dropped it, and a data URI in the v1 `url` field fails. Langfuse does not recognise that block as media, so it would store ~300 KB of base64 text per call. A Langfuse mask now replaces image bytes with a placeholder (size + hash prefix) before upload, so **X-rays are never stored in Langfuse**. Verified against a local capture server (decompressed OTLP upload): prompts and settings present, placeholder present for each of 4 calls, raw image bytes absent; trace 19 KB instead of ~1.6 MB.
- Retries: LangChain's `max_retries` is the total number of attempts (`max_retries=1` produced `attempts=1`, i.e. no retry; checked in the built request). Now 4 attempts; google-genai backs off exponentially on 429/5xx. Needed in practice: the next real run got 503 UNAVAILABLE ("model is currently experiencing high demand") from Google.
- Model choice, measured with `scripts/diagnose_gemini.py` on 2026-10-28 (00:56 IST): "reply OK" with a 30 s limit — `gemini-3.7-flash`, `gemini-3.8-flash`, `gemini-flash-latest` timed out (504); `gemini-3.6-flash` 2.2 s, `gemini-3.5-flash` 2.3 s, `gemini-3.5-flash-lite` 1.3 s. Full check on `gemini-3.6-flash`: text 2.8 s, text + image 2.8 s (called study 1212 normal; its label is No Finding), schema without image 3.3 s, schema + image 503 "high demand" after 3.7 s (transient; the agent retries, the diagnostic does not). Default is now `gemini-3.6-flash`; temperature is no longer sent because these models ignore it.
- Observation for the evaluation: with the schema but no image attached, the model still wrote five confident findings sentences instead of saying no image was provided. The agent always attaches the image, but this is the kind of unsupported claim the verifier and the report evaluation must catch.
