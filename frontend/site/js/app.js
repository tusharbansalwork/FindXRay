import { API_BASE, MODEL_URL, MODEL_MB, LIBRARY_SIZE, REPO_URL } from "./config.js";
import { Encoder } from "./encoder.js";
import { renderResults } from "./charts.js";

const $ = (sel, root = document) => root.querySelector(sel);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k);
  return e;
}

const HERO_UID = 549;
const NOT_FINDINGS = new Set(["No Finding", "Other Abnormal"]);
const STEP_NAMES = {
  retrieve: "Finding the similar cases",
  draft: "Drafting the report",
  look_again: "Second look for missed findings",
  amend: "Adding missed findings",
  check: "Checking every sentence",
  revise: "Revising unsupported sentences",
};
const NEXT_STEP = { retrieve: "draft", draft: "look_again", look_again: "check", amend: "check", check: "check-end", revise: "check" };

const state = { samples: null, encoder: null, upload: null, apiHealth: null };

/* ---------------- shared rendering ---------------- */

function cleanText(t) {
  return (t || "").replace(/\s+\./g, ".").replace(/\s+/g, " ").trim();
}
function caseTitle(groups) {
  const real = (groups || []).filter((g) => !NOT_FINDINGS.has(g));
  if (real.length) return real.join(", ");
  return groups?.includes("No Finding") ? "No finding" : "Other finding";
}
function openViewer(src, caption, cross) {
  const d = $("#viewer-dialog"), img = $("#dialog-img");
  if (cross) img.crossOrigin = "anonymous"; else img.removeAttribute("crossorigin");
  img.src = src;
  img.alt = caption;
  $("#dialog-caption").textContent = caption;
  d.showModal();
}

function renderCases(cases, { note } = {}) {
  const list = $("#cases");
  list.replaceChildren();
  $("#cases-sub").textContent = note || `From ${LIBRARY_SIZE.toLocaleString("en")} X-rays with radiologist reports. Select a case to read its report.`;
  for (const c of cases) {
    const detail = el("div", { class: "cdetail", id: `case-detail-${c.uid}` },
      el("img", { src: c.image_url, crossorigin: "anonymous", alt: `Case ${c.uid} X-ray`, loading: "lazy",
        onclick: () => openViewer(c.image_url, `Case ${c.uid}: ${caseTitle(c.groups)}`, true) }),
      c.findings && c.findings !== "None" ? el("p", {}, el("strong", { text: "Findings. " }), cleanText(c.findings)) : null,
      c.impression ? el("p", {}, el("strong", { text: "Impression. " }), cleanText(c.impression)) : null);
    const btn = el("button", { type: "button", "aria-expanded": "false", "aria-controls": `case-detail-${c.uid}` },
      el("span", { class: "rank", text: String(c.rank) }),
      el("span", { class: "cthumb" }, el("img", { src: c.image_url, crossorigin: "anonymous", alt: "", loading: "lazy" })),
      el("span", { class: "cbody" },
        el("span", { class: "chead" }, el("span", { class: "cid", text: `Case ${c.uid}` }),
          el("span", { class: "csim", text: `similarity ${c.similarity.toFixed(2)}` })),
        el("ul", { class: "tags", "aria-label": "Diagnoses" },
          ...(c.groups || []).map((g) => el("li", { class: `tag${g === "No Finding" ? " normal" : ""}`, text: g }))),
        el("p", { class: "cimp", text: cleanText(c.impression) || cleanText(c.findings) || "No report text." })));
    const li = el("li", { class: "case", id: `case-${c.uid}` }, btn, detail);
    btn.addEventListener("click", () => {
      const open = li.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
    });
    list.append(li);
  }
}

function focusCase(uid) {
  const li = document.getElementById(`case-${uid}`);
  if (!li) return;
  li.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  li.classList.add("flash");
  setTimeout(() => li.classList.remove("flash"), 1200);
  if (!li.classList.contains("open")) li.querySelector("button").click();
}

function sourceChips(sources) {
  return (sources || []).map((s) => {
    const m = /^case:(\d+)$/i.exec(s.trim());
    if (m) return el("button", { type: "button", class: "src", text: `Case ${m[1]}`, title: `Show case ${m[1]}`, onclick: () => focusCase(Number(m[1])) });
    if (s.trim().toLowerCase() === "image") return el("span", { class: "src", text: "This X-ray" });
    return el("span", { class: "src", text: s });
  });
}

function renderReport(r, { when }) {
  const body = $("#report-body");
  body.replaceChildren();
  $("#report-meta").textContent = when;
  for (const section of ["findings", "impression"]) {
    const items = r.report.filter((s) => s.section === section);
    if (!items.length) continue;
    const ol = el("ol", { class: "sentences" });
    for (const s of items) {
      const why = el("p", { class: "why", text: s.reason || "" });
      why.hidden = true;
      const chk = el("button", { type: "button", class: "src checked", text: "Checked", "aria-expanded": "false",
        title: "Show why the checker accepted this sentence" });
      chk.addEventListener("click", () => { why.hidden = !why.hidden; chk.setAttribute("aria-expanded", String(!why.hidden)); });
      ol.append(el("li", { class: "sentence" },
        el("p", { class: "stext", text: s.text }),
        el("div", { class: "sources" }, el("span", { class: "visually-muted", text: "Source:" }), ...sourceChips(s.sources), s.reason ? chk : null),
        why));
    }
    body.append(el("div", { class: "rsection" }, el("h4", { text: section === "findings" ? "Findings" : "Impression" }), ol));
  }
  if (!r.report.length) body.append(el("p", { class: "why", text: "No sentence passed the checks, so there is no report to show." }));
  if (r.missed_findings?.length) {
    body.append(el("p", { class: "why" }, el("span", { class: "added", text: "Second look: " }),
      `added ${r.missed_findings.map((m) => `${m.finding.toLowerCase()}${m.present === "uncertain" ? " (uncertain)" : ""}`).join(", ")} after the first draft.`));
  }
  if (r.removed?.length) {
    body.append(el("div", { class: "removed" }, el("h4", { text: `Removed by the checker (${r.removed.length})` }),
      ...r.removed.map((s) => el("p", {}, el("s", { text: s.text }), ` ${s.reason ? `(${s.reason})` : ""}`))));
  }
}

function agreementBlock(a) {
  const rad = new Set(a.radiologist), ai = new Set(a.ai);
  const both = [...rad].filter((x) => ai.has(x)), missed = [...rad].filter((x) => !ai.has(x)), extra = [...ai].filter((x) => !rad.has(x));
  const row = (cls, k, list) => el("li", {}, el("span", { class: `k ${cls}`, text: k }), el("span", { text: list.length ? list.join(", ") : "none" }));
  return el("div", { class: "agreement" },
    el("h4", { text: `Agreement with the radiologist: F1 ${a.f1.toFixed(2)}` }),
    el("p", { text: "Findings read from both reports by CheXbert, the automatic labeller used in the evaluation." }),
    el("ul", { class: "agree-list" }, row("both", "In both reports", both), row("missed", "Radiologist only", missed), row("extra", "AI only", extra)));
}

function renderReference(s) {
  const body = $("#reference-body");
  body.replaceChildren(el("div", { class: "ref-text" },
    s.radiologist.findings && s.radiologist.findings !== "None" ? [el("h4", { text: "Findings" }), el("p", { text: cleanText(s.radiologist.findings) })] : null,
    s.radiologist.impression ? [el("h4", { text: "Impression" }), el("p", { text: cleanText(s.radiologist.impression) })] : null),
    agreementBlock(s.agreement));
}

function setQueryImage(src, { tl, br, alt, cross }) {
  const img = $("#query-img");
  img.removeAttribute("crossorigin");
  if (cross) img.crossOrigin = "anonymous";
  img.alt = alt;
  $("#ov-bl").textContent = "";
  img.onload = () => { $("#ov-bl").textContent = `${img.naturalWidth} × ${img.naturalHeight} px`; };
  img.src = src;
  $("#ov-tl").textContent = tl;
  $("#ov-br").textContent = br;
  $(".viewer").onclick = () => openViewer(src, alt, cross);
}

function showResult({ reference }) {
  const r = $("#result");
  r.hidden = false;
  r.classList.toggle("no-reference", !reference);
}

/* ---------------- status line ---------------- */

function status(text, { progress, error } = {}) {
  const s = $("#status");
  if (!text) { s.hidden = true; return; }
  s.hidden = false;
  s.classList.toggle("error", !!error);
  s.replaceChildren(el("span", { text }));
  if (progress != null) s.append(el("div", { class: "bar" }, el("span", { style: `width:${Math.round(progress * 100)}%` })));
}

/* ---------------- samples ---------------- */

function selectSample(s, { scroll = true } = {}) {
  state.upload = null;
  document.querySelectorAll(".pick[data-uid]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.uid) === s.uid)));
  status(null);
  setQueryImage(s.image, { tl: `Study ${s.uid}`, br: "Held-out test X-ray", alt: `Chest X-ray: ${s.title}` });
  $("#query-meta").replaceChildren(el("strong", { text: s.title }), `. Radiologist's diagnosis: ${s.term.toLowerCase()}.`);
  renderCases(s.cases);
  renderReport(s, { when: `Recorded during evaluation. Gemini 3.5 Flash-Lite took ${s.seconds.toFixed(0)} s.` });
  renderReference(s);
  showResult({ reference: true });
  if (scroll) $("#try").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function buildPicker(samples) {
  const picker = $("#picker"), uploadTile = $("#upload-tile");
  for (const s of samples) {
    const b = el("button", { type: "button", class: "pick", role: "listitem", "aria-pressed": "false", "data-uid": String(s.uid),
      onclick: () => selectSample(s, { scroll: false }) },
      el("span", { class: "thumb" }, el("img", { src: s.image, alt: "", loading: "lazy" })),
      el("span", { class: "pick-title", text: s.title }),
      el("span", { class: "pick-term", text: s.term }));
    picker.insertBefore(b, uploadTile);
  }
}

function buildHero(sample) {
  const fig = $(".hero-figure");
  const q = $("#hero-query");
  if (!q.getAttribute("src")) q.src = sample.image;   // normally set in the HTML so it loads before any script runs
  $("#hero-term").textContent = sample.term;
  const ol = $("#hero-matches");
  const items = sample.cases.map((c) => {
    const img = el("img", { src: c.image_url, crossorigin: "anonymous", alt: `Case ${c.uid}: ${caseTitle(c.groups)}` });
    const li = el("li", { title: `Case ${c.uid}: ${caseTitle(c.groups)}` }, img, el("span", { class: "rank-ov", text: String(c.rank) }));
    ol.append(li);
    const loaded = img.complete ? Promise.resolve()
      : Promise.race([new Promise((r) => { img.onload = img.onerror = r; }), new Promise((r) => setTimeout(r, 5000))]);
    return { li, loaded };
  });
  // The scan starts as soon as the query X-ray is on screen (it is local and small). The matches then
  // appear strictly top to bottom, 150 ms apart, after the sweep; a match that is still downloading
  // holds back the ones below it, so the order never jumps (each waits at most 5 s for its image).
  let started = false;
  const start = async () => {
    if (started) return;
    started = true;
    fig.classList.add("run");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sleep = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms)));
    let next = performance.now() + (reduced ? 0 : 1250);
    for (const { li, loaded } of items) {
      await loaded;
      await sleep(next - performance.now());
      li.classList.add("in");
      next = performance.now() + (reduced ? 0 : 150);
    }
  };
  if (q.complete && q.naturalWidth) start();
  else { q.addEventListener("load", start, { once: true }); q.addEventListener("error", start, { once: true }); }
  setTimeout(start, 1200);   // never wait longer than this for the query image
}

function buildStruggle(s) {
  const sec = $("#struggle");
  const ai = s.report.find((x) => /pneumothorax/i.test(x.text));
  const rad = cleanText(s.radiologist.impression).split(/(?<=\.)\s+(?=\d\.)/)[0];
  $("#struggle-body").replaceChildren(
    el("div", { class: "film", role: "button", tabindex: "0", "aria-label": "Open the X-ray full size",
      onclick: () => openViewer(s.image, `Study ${s.uid}: ${s.title}`, false),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openViewer(s.image, `Study ${s.uid}: ${s.title}`, false); } } },
      el("img", { src: s.image, alt: `Chest X-ray, study ${s.uid}` }),
      el("span", { class: "ov tl", text: `Study ${s.uid}` }), el("span", { class: "ov br", text: "Held-out test X-ray" })),
    el("div", {},
      el("h3", { text: s.title }),
      el("p", { class: "term", text: s.term }),
      el("p", { text: s.what_happened }),
      el("div", { class: "contrast" },
        el("div", { class: "said-rad" }, el("h4", { text: "Radiologist" }), el("p", { text: rad })),
        el("div", { class: "said-ai" }, el("h4", { text: "FindXRay" }), el("p", { text: ai ? ai.text : "Did not mention a pneumothorax." }))),
      el("h4", { class: "list-title", text: "Why it missed" }),
      el("ul", {}, ...s.why.map((w) => el("li", { text: w }))),
      el("h4", { class: "list-title", text: "What fixes it next" }),
      el("ul", { class: "fix" }, ...s.fix.map((w) => el("li", { text: w })))));
  sec.hidden = false;
}

/* ---------------- upload: encode, search, draft ---------------- */

async function fetchJSON(url, opts = {}, timeoutMs = 20000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...opts, signal: ctl.signal });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`), { status: r.status });
    return body;
  } finally { clearTimeout(t); }
}

// The free server sleeps when idle; the first request can take about a minute while it starts.
async function apiCall(path, payload, onWaking) {
  const started = Date.now();
  let warned = false;
  for (;;) {
    try {
      return await fetchJSON(API_BASE + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }, 15000);
    } catch (e) {
      if (e.status && e.status < 500) throw e;
      if (Date.now() - started > 120000) throw new Error("The search server did not respond. Try again in a minute.");
      if (!warned) { onWaking?.(); warned = true; }
      await new Promise((r) => setTimeout(r, 4000));
    }
  }
}

function checkHealth() {
  fetchJSON(API_BASE + "/health", {}, 70000).then((h) => { state.apiHealth = h; }).catch(() => { state.apiHealth = null; });
}

function getEncoder() {
  state.encoder ??= new Encoder(MODEL_URL);
  return state.encoder;
}

async function handleFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    status("That file isn't a JPEG, PNG or WebP image. Choose a chest X-ray image file.", { error: true });
    return;
  }
  if (file.size > 25 * 2 ** 20) { status("That image is over 25 MB. Choose a smaller file.", { error: true }); return; }
  document.querySelectorAll(".pick[data-uid]").forEach((b) => b.setAttribute("aria-pressed", "false"));
  const url = URL.createObjectURL(file);
  state.upload = { file, url };
  checkHealth();
  setQueryImage(url, { tl: "Your upload", br: "Stays on this device for search", alt: "Your uploaded X-ray" });
  $("#query-meta").replaceChildren(el("strong", { text: file.name }));
  $("#cases").replaceChildren();
  $("#cases-sub").textContent = "Searching…";
  $("#report-body").replaceChildren();
  $("#report-meta").textContent = "";
  showResult({ reference: false });
  $("#try").scrollIntoView({ behavior: "smooth" });

  try {
    const enc = getEncoder();
    status(`Preparing the X-ray model (${MODEL_MB} MB, downloaded once, then kept by your browser)…`, { progress: 0 });
    await enc.load((p, cached) => status(cached ? "Loading the X-ray model from your browser's cache…"
      : `Preparing the X-ray model (${MODEL_MB} MB, downloaded once, then kept by your browser)…`, { progress: p }));
    status("Reading the X-ray on this device…");
    const { embedding, ms } = await enc.embed(file);
    if (state.upload?.file !== file) return;
    state.upload.embedding = embedding;
    status(`Searching ${LIBRARY_SIZE.toLocaleString("en")} cases…`);
    const res = await apiCall("/search", { embedding: Array.from(embedding), k: 5 },
      () => status("Waking up the search server. Free hosting sleeps when idle; this takes up to a minute the first time."));
    if (state.upload?.file !== file) return;
    status(null);
    $("#query-meta").replaceChildren(el("strong", { text: file.name }), `. Read on this device in ${(ms / 1000).toFixed(1)} s.`);
    renderCases(res.results);
    renderDraftPrompt();
  } catch (e) {
    console.error(e);
    status(e.message || "Something went wrong while reading the X-ray.", { error: true });
  }
}

function renderDraftPrompt() {
  const h = state.apiHealth;
  const body = $("#report-body");
  $("#report-meta").textContent = "";
  const disabledReason = h && h.reports_enabled === false ? "Report drafting isn't switched on for this server."
    : h && h.reports_left_today === 0 ? "Live report drafting is paused for today (daily limit reached). The sample X-rays above show full reports." : null;
  const btn = el("button", { type: "button", class: "btn primary", text: "Draft a report", disabled: !!disabledReason, onclick: draftReport });
  body.replaceChildren(el("div", { class: "draft-cta" },
    el("p", { text: disabledReason || "Takes about 15 to 30 seconds. This sends the image to Google's Gemini, which writes a report that cites the cases above; every sentence is then checked." }),
    btn));
}

async function imagePayload(file) {
  if ((file.type === "image/jpeg" || file.type === "image/png") && file.size <= 8 * 2 ** 20) {
    const buf = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return { image_base64: btoa(bin), mime_type: file.type };
  }
  // Other formats or very large files: re-encode as JPEG, longest side at most 2048 px.
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, 2048 / Math.max(bmp.width, bmp.height));
  const c = Object.assign(document.createElement("canvas"), { width: Math.round(bmp.width * s), height: Math.round(bmp.height * s) });
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  const dataUrl = c.toDataURL("image/jpeg", 0.92);
  return { image_base64: dataUrl.slice(dataUrl.indexOf(",") + 1), mime_type: "image/jpeg" };
}

async function draftReport() {
  const up = state.upload;
  if (!up?.embedding) return;
  const body = $("#report-body");
  const list = el("ol", { class: "progress", "aria-live": "polite" });
  body.replaceChildren(list);
  const t0 = performance.now();
  let last = t0, active = null;
  const addRow = (key, cls) => {
    const li = el("li", { class: cls }, el("span", { class: "dot", "aria-hidden": "true" }), el("span", { text: STEP_NAMES[key] }), el("span", { class: "t" }));
    li.dataset.key = key;
    list.append(li);
    return li;
  };
  const activate = (key) => { if (key === "check-end") { active = null; return; } active = addRow(key, "active"); };
  const finish = (key) => {
    const now = performance.now();
    let li = active && active.dataset.key === key ? active : null;
    if (!li) {   // a step we didn't predict (amend, revise): insert it before the active row
      li = addRow(key, "done");
      if (active) list.insertBefore(li, active);
    }
    li.className = "done";
    li.querySelector(".t").textContent = `${((now - last) / 1000).toFixed(1)} s`;
    last = now;
    if (li === active) active = null;
  };

  try {
    const payload = { embedding: Array.from(up.embedding), k: 5, ...(await imagePayload(up.file)) };
    activate("retrieve");
    const res = await fetch(API_BASE + "/report/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      throw new Error(typeof b.detail === "string" ? b.detail : `The report server answered HTTP ${res.status}.`);
    }
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "", result = null, model = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.event === "started") model = ev.model;
        else if (ev.event === "step") {
          finish(ev.step);
          if (!active) activate(ev.step === "check" ? "check-end" : NEXT_STEP[ev.step]);
        } else if (ev.event === "result") result = ev.result;
        else if (ev.event === "error") throw new Error(ev.message);
      }
    }
    if (!result) throw new Error("The report stream ended early. Try again.");
    if (state.upload !== up) return;
    renderReport(result, { when: `Drafted by ${model || result.model} in ${((performance.now() - t0) / 1000).toFixed(0)} s.` });
  } catch (e) {
    console.error(e);
    body.replaceChildren(el("div", { class: "draft-cta" }, el("p", { text: e.message || "Drafting failed." }),
      el("button", { type: "button", class: "btn", text: "Try again", onclick: draftReport })));
  }
}

function wireUpload() {
  const input = $("#file"), tile = $("#upload-tile");
  input.addEventListener("change", () => { handleFile(input.files[0]); input.value = ""; });
  tile.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  tile.tabIndex = 0;
  for (const t of [tile, $("#result")]) {
    t.addEventListener("dragover", (e) => { e.preventDefault(); tile.classList.add("drag"); });
    t.addEventListener("dragleave", () => tile.classList.remove("drag"));
    t.addEventListener("drop", (e) => { e.preventDefault(); tile.classList.remove("drag"); handleFile(e.dataTransfer.files[0]); });
  }
}

/* ---------------- start ---------------- */

async function main() {
  document.querySelectorAll("[data-repo]").forEach((a) => { a.href = REPO_URL; });
  wireUpload();
  const metricsP = fetch("data/metrics.json").then((r) => r.json());
  const samples = await fetch("data/samples.json").then((r) => r.json());
  state.samples = samples;
  const hero = samples.samples.find((s) => s.uid === HERO_UID) || samples.samples[0];
  buildHero(hero);   // first, so the five match images start downloading as early as possible
  const metrics = await metricsP;
  buildPicker(samples.samples);
  buildStruggle(samples.struggle);
  renderResults($("#results-body"), metrics);
  selectSample(samples.samples[0], { scroll: false });
  document.querySelector('[data-action="first-sample"]').addEventListener("click", (e) => {
    e.preventDefault(); selectSample(samples.samples[0]);
  });
  document.querySelector('[data-action="upload"]').addEventListener("click", (e) => {
    e.preventDefault(); $("#try").scrollIntoView({ behavior: "smooth" }); $("#file").click();
  });
}

main().catch((e) => {
  console.error(e);
  status("The page failed to load its data. Reload to try again.", { error: true });
});
