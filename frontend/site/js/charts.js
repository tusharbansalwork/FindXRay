// "How well it works": headline numbers, plain-English chapters, comparison tables and SVG charts,
// all built from data/metrics.json (written by scripts/build_site_data.py from the saved results).

const NS = "http://www.w3.org/2000/svg";
const C = { ours: "#E8F2FF", other: "#4E5A6A", grid: "#18202B", axis: "#6E7783", text: "#EDF0F4", muted: "#A1AAB6" };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k);
  return e;
}
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);
const pct = (x) => `${Math.round(x * 100)}%`;

function tooltip(chart) {
  const tip = el("div", { class: "tip", role: "presentation" });
  tip.hidden = true;
  chart.append(tip);
  return {
    show(evt, text) {
      const box = chart.getBoundingClientRect();
      const t = evt.target.getBoundingClientRect();
      tip.textContent = text;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max(t.left + t.width / 2 - box.left, 90), box.width - 90)}px`;
      tip.style.top = `${t.top - box.top + 8}px`;
    },
    hide() { tip.hidden = true; },
  };
}

// Horizontal bars with 95% CI whiskers; label above each bar, value after the whisker.
function ciBars(W, rows, { max, step, fmt = f3, metricName }) {
  const rowH = 58, top = 4, right = 62, barH = 12;
  const H = top + rows.length * rowH + 26;
  const x = (v) => (v / max) * (W - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": rows.map((r) => `${r.label}: ${fmt(r.v)}`).join("; ") });
  for (let t = 0; t <= max + 1e-9; t += step) {
    s.append(svg("line", { x1: x(t), x2: x(t), y1: top, y2: H - 24, stroke: C.grid, "stroke-width": 1 }));
    s.append(svg("text", { x: x(t), y: H - 6, fill: C.axis, "font-size": 12, "text-anchor": t === 0 ? "start" : "middle" }, f2(t)));
  }
  const marks = [];
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    s.append(svg("text", { x: 0, y: y + 15, fill: r.ours ? C.text : C.muted, "font-size": 14,
      "font-weight": r.ours ? 600 : 400 }, r.label));
    const by = y + 25, w = Math.max(x(r.v), 6);
    s.append(svg("path", { d: `M0,${by} h${w - 4} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`,
      fill: r.ours ? C.ours : C.other }));
    if (r.lo != null) {
      const wy = by + barH / 2, stroke = r.ours ? "#9FB4CC" : C.muted;
      s.append(svg("line", { x1: x(r.lo), x2: x(r.hi), y1: wy, y2: wy, stroke, "stroke-width": 1.5 }));
      for (const v of [r.lo, r.hi]) s.append(svg("line", { x1: x(v), x2: x(v), y1: wy - 6, y2: wy + 6, stroke, "stroke-width": 1.5 }));
    }
    s.append(svg("text", { x: x(r.hi ?? r.v) + 10, y: by + barH - 1, fill: C.text, "font-size": 14,
      "font-weight": r.ours ? 700 : 500 }, fmt(r.v)));
    const hit = svg("rect", { x: 0, y, width: W, height: rowH - 4, fill: "transparent" });
    hit.dataset.tip = `${r.label}: ${metricName} ${fmt(r.v)}` + (r.lo != null ? `, 95% CI ${fmt(r.lo)} to ${fmt(r.hi)}` : "");
    marks.push(hit);
    s.append(hit);
  });
  return { node: s, marks };
}

// Base rate (grey) vs rate among the five matches (light), with the lift.
function dumbbell(W, rows) {
  const rowH = 40, top = 4, right = 58, H = top + rows.length * rowH + 24, max = 0.5;
  const x = (v) => (v / max) * (W - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": rows.map((r) => `${r.finding}: ${pct(r.top5_rate)} of matches vs ${pct(r.base_rate)} of library`).join("; ") });
  for (let t = 0; t <= max + 1e-9; t += 0.1) {
    s.append(svg("line", { x1: x(t), x2: x(t), y1: top, y2: H - 22, stroke: C.grid, "stroke-width": 1 }));
    s.append(svg("text", { x: x(t), y: H - 4, fill: C.axis, "font-size": 12, "text-anchor": t === 0 ? "start" : "middle" }, pct(t)));
  }
  const marks = [];
  rows.forEach((r, i) => {
    const y = top + i * rowH, cy = y + 27;
    s.append(svg("text", { x: 0, y: y + 12, fill: C.muted, "font-size": 13.5 }, r.finding));
    s.append(svg("line", { x1: x(r.base_rate), x2: x(r.top5_rate), y1: cy, y2: cy, stroke: C.other, "stroke-width": 2 }));
    s.append(svg("circle", { cx: x(r.base_rate), cy, r: 4.5, fill: C.other, stroke: "#07090D", "stroke-width": 2 }));
    s.append(svg("circle", { cx: x(r.top5_rate), cy, r: 5.5, fill: C.ours, stroke: "#07090D", "stroke-width": 2 }));
    s.append(svg("text", { x: x(Math.max(r.top5_rate, r.base_rate)) + 11, y: cy + 5, fill: C.text, "font-size": 14, "font-weight": 600 },
      `${r.lift.toFixed(1)}×`));
    const hit = svg("rect", { x: 0, y, width: W, height: rowH - 2, fill: "transparent" });
    hit.dataset.tip = `${r.finding}: ${pct(r.top5_rate)} of matches vs ${pct(r.base_rate)} of the library`;
    marks.push(hit);
    s.append(hit);
  });
  return { node: s, marks };
}

function chartBox(title, sub, chartFn) {
  const box = el("div", { class: "chart" });
  box.append(el("p", { class: "chart-title", text: title }));
  if (sub) box.append(el("p", { class: "chart-sub", text: sub }));
  const holder = el("div");
  box.append(holder);
  const tip = tooltip(box);
  const width = () => Math.round(Math.max(300, Math.min(620, holder.getBoundingClientRect().width || 560)));
  let lastArgs = [], lastW = 0;
  const draw = (...args) => {
    lastArgs = args;
    if (!holder.isConnected) { requestAnimationFrame(() => draw(...args)); return; }
    lastW = width();   // drawn at the real width, so text keeps its size on phones
    holder.replaceChildren();
    const { node, marks } = chartFn(lastW, ...args);
    holder.append(node);
    for (const m of marks) {
      m.addEventListener("pointerenter", (e) => tip.show(e, m.dataset.tip));
      m.addEventListener("pointerleave", () => tip.hide());
    }
  };
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (width() !== lastW) draw(...lastArgs); }, 150); });
  return { box, draw };
}

function table(head, rows, oursIndex, caption) {
  return el("table", { class: "compare" },
    caption ? el("caption", { text: caption }) : null,
    el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { scope: "col", text: h })))),
    el("tbody", {}, ...rows.map((r, i) => el("tr", i === oursIndex ? { class: "ours" } : {}, ...r.map((c) => el("td", { text: c }))))));
}

function para(...parts) {
  const p = el("p");
  for (const x of parts) p.append(typeof x === "string" ? x : el("strong", { text: x.b }));
  return p;
}

function chapter(title, explain, extra, chart) {
  return el("div", { class: "chapter" },
    el("div", {}, el("h3", { text: title }), el("div", { class: "explain" }, ...explain), extra),
    chart);
}

export function renderResults(root, m) {
  const R = m.retrieval, byModel = Object.fromEntries(R.models.map((r) => [r.model, r]));
  const ours = byModel["rad-dino-224"], rnd = byModel.random, gen = byModel["dinov2-base"];
  const p5 = (r, sl = "all") => r[`${sl}:P@5`][0];
  const rep = m.report, sys = Object.fromEntries(rep.systems.map((s) => [s.system, s]));
  const a = sys.agent_final, io = sys.image_only, nc = sys.nearest_case;
  const pd = Object.fromEntries(rep.paired.map((p) => [p.comparison, p]));
  const vsIO = pd["agent vs Gemini without retrieval"], vsNC = pd["agent vs copying the nearest case"];
  const st = rep.agent, c = m.compression;

  // Headline numbers
  const lift = p5(ours) / p5(rnd), gain = a.micro_f1[0] / io.micro_f1[0] - 1;
  root.append(el("div", { class: "headline" },
    el("div", {}, el("span", { class: "big", text: `${lift.toFixed(1)}×` }),
      el("span", { class: "what", text: "more often than chance, the five matches share a diagnosis with the X-ray you searched" }),
      el("span", { class: "vs", text: `Precision@5 ${f2(p5(ours))} vs ${f2(p5(rnd))} for a random pick, on ${R.queries} test X-rays` })),
    el("div", {}, el("span", { class: "big", text: `+${Math.round(gain * 100)}%` }),
      el("span", { class: "what", text: "closer agreement with the radiologist's findings than the same AI without similar cases" }),
      el("span", { class: "vs", text: `F1 ${f2(a.micro_f1[0])} vs ${f2(io.micro_f1[0])}, on ${rep.studies} test X-rays` })),
    el("div", {}, el("span", { class: "big", text: pct(a.normal_abnormal_acc) }),
      el("span", { class: "what", text: "of X-rays correctly called normal or abnormal in the report" }),
      el("span", { class: "vs", text: `${pct(io.normal_abnormal_acc)} for the same AI without similar cases` }))));

  // 1 · Finding similar cases
  const order = ["random", "dinov2-base", "biomedclip", "rad-dino-224"];
  const rowsFor = (sl) => order.map((k) => {
    const [v, lo, hi] = byModel[k][`${sl}:P@5`];
    return { label: byModel[k].label, v, lo, hi, ours: k === "rad-dino-224" };
  });
  const retrieval = chartBox("Precision@5 by image model", "Bars: mean. Whiskers: 95% confidence interval.",
    (W, sl) => ciBars(W, rowsFor(sl), { max: 0.5, step: 0.1, metricName: "precision@5" }));
  const toggle = el("div", { class: "toggle", role: "group", "aria-label": "Which X-rays" });
  for (const [sl, name] of [["all", "All X-rays"], ["abnormal", "Abnormal X-rays only"]]) {
    const b = el("button", { type: "button", "aria-pressed": String(sl === "all"), text: name });
    b.addEventListener("click", () => {
      toggle.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      retrieval.draw(sl);
    });
    toggle.append(b);
  }
  retrieval.box.insertBefore(toggle, retrieval.box.children[2]);
  retrieval.draw("all");
  root.append(chapter("Finding similar cases",
    [para("Each of ", { b: `${R.queries} test X-rays` }, ` was searched against the ${R.library.toLocaleString("en")}-case library. `,
      "A match counts when the returned case shares at least one diagnosis with the X-ray searched; precision@5 is the share of the five matches that do."),
     para({ b: "A model trained on chest X-rays beats both a general-purpose image model and a medical image-text model" },
      `, by ${Math.round((p5(ours) / p5(gen) - 1) * 100)}% over the general one, and the lead holds on abnormal X-rays, where matching is hardest (${f2(p5(ours, "abnormal"))} vs ${f2(p5(rnd, "abnormal"))} for chance).`)],
    table(["Image model", "P@5", "P@5 abnormal", "mAP@10"],
      order.map((k) => [byModel[k].label, f3(p5(byModel[k])), f3(p5(byModel[k], "abnormal")), f3(byModel[k]["all:mAP@10"][0])]), 3),
    retrieval.box));

  // 2 · Which findings it matches
  const lifts = R.per_finding.filter((r) => r.n >= 40 && r.finding !== "No Finding")
    .map((r) => ({ ...r, lift: r.top5_rate / r.base_rate })).sort((x, y) => y.lift - x.lift);
  const card = lifts.find((r) => r.finding === "Cardiomegaly");
  const liftChart = chartBox("How often the five matches share the finding",
    "Grey dot: share of the whole library with the finding. Light dot: share among the five matches.", (W) => dumbbell(W, lifts));
  liftChart.draw();
  root.append(chapter("Strongest where the whole picture changes",
    [para(`When an X-ray shows an enlarged heart, ${pct(card.top5_rate)} of its five matches show one too, against ${pct(card.base_rate)} of the library. `,
      { b: "Heart size, lung volume and over-inflation are matched three to four times better than chance." }),
     para("Small, local findings such as nodules, fractures and devices are harder to match by overall appearance. That is why the report step also reads the image itself instead of copying its neighbours.")],
    null, liftChart.box));

  // 3 · Writing the report
  const reportChart = chartBox("Agreement with the radiologist's findings (F1)", "Bars: mean. Whiskers: 95% confidence interval.",
    (W) => ciBars(W, [nc, io, a].map((s) => ({ label: s.label, v: s.micro_f1[0], lo: s.micro_f1[1], hi: s.micro_f1[2], ours: s.system === "agent_final" })),
      { max: 0.7, step: 0.1, metricName: "F1" }));
  reportChart.draw();
  root.append(chapter("Writing the report",
    [para(`On ${rep.studies} test X-rays with real radiologist reports, `, { b: "CheXbert" },
      ", an automatic labeller from Stanford, lists 14 standard findings in the AI's report and in the radiologist's. F1 scores how well the two lists agree, from 0 to 1."),
     para({ b: `FindXRay beats copying the closest case's report by +${f2(vsNC.diff)} and the same Gemini model without similar cases by +${f2(vsIO.diff)}` },
      `. Both gains are statistically significant: their 95% confidence intervals sit above zero. It also cuts false findings, with precision up ${Math.round((a.precision / io.precision - 1) * 100)}%.`)],
    table(["System", "F1", "Precision", "Recall", "Normal/abnormal"],
      [nc, io, a].map((s) => [s.label, f3(s.micro_f1[0]), f2(s.precision), f2(s.recall), pct(s.normal_abnormal_acc)]), 2),
    reportChart.box));

  // 4 · Engineering numbers
  root.append(el("div", { class: "runs" },
    el("h3", { text: "Built to run anywhere" }),
    ...[
      [`${Math.round(c.int8_mb)} MB`, `image model, compressed from ${Math.round(c.fp32_mb)} MB with no loss in accuracy`],
      ["0.5 s", "to read an X-ray inside the browser; the image stays on the device for search"],
      [`${c.server_search_ms} ms`, `to search all ${R.library.toLocaleString("en")} cases on the server`],
      [`${Math.round(st.agent_seconds_median)} s`, `median for a full drafted and checked report (${Math.round(st.agent_seconds_p90)} s for 9 in 10)`],
    ].map(([n, d]) => el("div", {}, el("span", { class: "n", text: n }), el("span", { class: "d", text: d })))));

  root.append(el("p", { class: "fine", text:
    `What drives the gain: drafting with similar cases in view. The second look and sentence checks changed F1 by +${f3(pd["effect of look-again + verification"].diff)}, `
    + `because a checker from the same model family rarely disagrees with the writer (it approved ${(st.first_pass_supported_mean * 100).toFixed(1)}% of sentences); `
    + `an independent, stronger checker is the next step. With ${rep.studies} studies the intervals are wide, and CheXbert's labels are themselves automatic. `
    + `Report model: ${rep.model}.` }));
}
