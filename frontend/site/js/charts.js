// "How well it works": text + hand-built SVG charts from data/metrics.json.

const NS = "http://www.w3.org/2000/svg";
const C = { ours: "#D6EBFF", other: "#56677F", grid: "#1C2C41", axis: "#6A7E98", text: "#E3EBF5", muted: "#95A7BE" };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
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
      const t = evt.target.getBoundingClientRect?.() ?? box;
      tip.textContent = text;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max(t.left + t.width / 2 - box.left, 70), box.width - 70)}px`;
      tip.style.top = `${t.top - box.top}px`;
    },
    hide() { tip.hidden = true; },
  };
}

// Horizontal bars with 95% CI whiskers; label above each bar, value after the whisker.
function ciBars(W, rows, { max, step, fmt = f3, metricName }) {
  const rowH = 50, top = 6, left = 0, right = 56, barH = 10;
  const H = top + rows.length * rowH + 24;
  const x = (v) => left + (v / max) * (W - left - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": rows.map((r) => `${r.label}: ${fmt(r.v)}`).join("; ") });
  for (let t = 0; t <= max + 1e-9; t += step) {
    s.append(svg("line", { x1: x(t), x2: x(t), y1: top, y2: H - 22, stroke: C.grid, "stroke-width": 1 }));
    s.append(svg("text", { x: x(t), y: H - 6, fill: C.axis, "font-size": 11, "text-anchor": "middle" }, f2(t)));
  }
  const marks = [];
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const color = r.ours ? C.ours : C.other;
    s.append(svg("text", { x: left, y: y + 14, fill: r.ours ? C.text : C.muted, "font-size": 13,
      "font-weight": r.ours ? 600 : 400 }, r.label));
    const by = y + 22, w = Math.max(x(r.v) - left, 2);
    // bar anchored at the baseline, 4px rounded data end
    s.append(svg("path", { d: `M${left},${by} h${w - 4} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: color }));
    if (r.lo != null) {
      const wy = by + barH / 2;
      s.append(svg("line", { x1: x(r.lo), x2: x(r.hi), y1: wy, y2: wy, stroke: C.text, "stroke-width": 1.5, opacity: 0.85 }));
      for (const v of [r.lo, r.hi]) s.append(svg("line", { x1: x(v), x2: x(v), y1: wy - 5, y2: wy + 5, stroke: C.text, "stroke-width": 1.5, opacity: 0.85 }));
    }
    s.append(svg("text", { x: x(r.hi ?? r.v) + 8, y: by + barH - 1, fill: C.text, "font-size": 13,
      "font-weight": r.ours ? 600 : 400 }, fmt(r.v)));
    const hit = svg("rect", { x: 0, y, width: W, height: rowH - 4, fill: "transparent" });
    hit.dataset.tip = `${r.label}: ${metricName} ${fmt(r.v)}` + (r.lo != null ? ` (95% CI ${fmt(r.lo)} to ${fmt(r.hi)})` : "");
    marks.push(hit);
    s.append(hit);
  });
  return { node: s, marks };
}

function chartBox(title, sub, chartFn, tableRows, tableHead) {
  const box = el("div", { class: "chart" });
  box.append(el("p", { class: "chart-title", text: title }));
  if (sub) box.append(el("p", { class: "chart-sub", text: sub }));
  const holder = el("div");
  box.append(holder);
  const tip = tooltip(box);
  let lastArgs = [], lastW = 0;
  const draw = (...args) => {
    lastArgs = args;
    if (!holder.isConnected) { requestAnimationFrame(() => draw(...args)); return; }
    // Draw at the real width so text stays at its intended size on phones.
    const W = Math.round(Math.max(300, Math.min(600, holder.getBoundingClientRect().width || 560)));
    lastW = W;
    holder.replaceChildren();
    const { node, marks } = chartFn(W, ...args);
    holder.append(node);
    for (const m of marks) {
      m.addEventListener("pointerenter", (e) => tip.show(e, m.dataset.tip));
      m.addEventListener("pointerleave", () => tip.hide());
    }
  };
  let t;
  addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      const W = Math.round(Math.max(300, Math.min(600, holder.getBoundingClientRect().width || 560)));
      if (W !== lastW) draw(...lastArgs);
    }, 150);
  });
  if (tableRows) {
    const table = el("table", {},
      el("thead", {}, el("tr", {}, ...tableHead.map((h) => el("th", { text: h })))),
      el("tbody", {}, ...tableRows.map((r) => el("tr", {}, ...r.map((c) => el("td", { text: c }))))));
    box.append(el("details", { class: "table" }, el("summary", { text: "Show as a table" }), table));
  }
  return { box, draw };
}

function figs(items) {
  return el("ul", { class: "figs" }, ...items.map(([n, d]) => el("li", {}, el("span", { class: "n", text: n }), el("span", { class: "d", text: d }))));
}

function block(title, explainNodes, figItems, chart) {
  const left = el("div", {}, el("h3", { text: title }), el("div", { class: "explain" }, ...explainNodes), figItems ? figs(figItems) : null);
  return el("div", { class: "result-block" }, left, chart);
}
function para(...parts) {
  const p = el("p");
  for (const x of parts) p.append(typeof x === "string" ? x : el("strong", { text: x.b }));
  return p;
}

export function renderResults(root, m) {
  const R = m.retrieval, byModel = Object.fromEntries(R.models.map((r) => [r.model, r]));
  const ours = byModel["rad-dino-224"], rnd = byModel.random, gen = byModel["dinov2-base"];
  const p5 = (r, sl = "all") => r[`${sl}:P@5`][0];

  // 1 · Finding similar cases
  const order = ["random", "dinov2-base", "biomedclip", "rad-dino-224"];
  const rowsFor = (sl) => order.map((k) => {
    const [v, lo, hi] = byModel[k][`${sl}:P@5`];
    return { label: byModel[k].label, v, lo, hi, ours: k === "rad-dino-224" };
  });
  const retrievalChart = chartBox("Precision@5 by image model",
    `${R.queries} test X-rays searched against the ${R.library.toLocaleString("en")}-case library`,
    (W, sl) => ciBars(W, rowsFor(sl), { max: 0.5, step: 0.1, metricName: "precision@5" }),
    order.map((k) => [byModel[k].label, f3(p5(byModel[k])), f3(p5(byModel[k], "abnormal")), f3(byModel[k]["all:mAP@10"][0]), f3(byModel[k]["all:MRR"][0])]),
    ["Model", "P@5 all", "P@5 abnormal", "mAP@10", "MRR"]);
  const toggle = el("div", { class: "toggle", role: "group", "aria-label": "Which X-rays" });
  for (const [sl, name] of [["all", "All X-rays"], ["abnormal", "Abnormal X-rays only"]]) {
    const b = el("button", { type: "button", "aria-pressed": String(sl === "all"), text: name });
    b.addEventListener("click", () => {
      toggle.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      retrievalChart.draw(sl);
    });
    toggle.append(b);
  }
  retrievalChart.box.insertBefore(toggle, retrievalChart.box.children[2]);
  retrievalChart.draw("all");

  const c = m.compression;
  root.append(block(
    `Finds the right kind of case ${(p5(ours) / p5(rnd)).toFixed(1)}× more often than chance`,
    [para("Precision@5 is the share of the five returned cases that have at least one diagnosis in common with the X-ray you searched. ",
      { b: "A chest X-ray model beats both a general-purpose image model and a medical model trained on mixed images" },
      ", and it holds on abnormal X-rays, where matching is hardest.")],
    [
      [f2(p5(ours)), `precision@5, against ${f2(p5(rnd))} for a random pick`],
      [`+${Math.round((p5(ours) / p5(gen) - 1) * 100)}%`, `over a general-purpose vision AI (${f2(p5(gen))})`],
      [`${(p5(ours, "abnormal") / p5(rnd, "abnormal")).toFixed(1)}×`, `chance on abnormal X-rays (${f2(p5(ours, "abnormal"))} vs ${f2(p5(rnd, "abnormal"))})`],
      [`${Math.round(c.int8_mb)} MB`, `model size, compressed from ${Math.round(c.fp32_mb)} MB with no loss in accuracy, so it runs inside a web browser`],
    ],
    retrievalChart.box));

  // 2 · Which findings it matches
  const lifts = R.per_finding.filter((r) => r.n >= 40 && r.finding !== "No Finding")
    .map((r) => ({ ...r, lift: r.top5_rate / r.base_rate })).sort((a, b) => b.lift - a.lift);
  const liftChart = chartBox("Share of the five matches with the same finding",
    "Grey dot: share of the whole library with that finding. Light dot: share among the five matches.",
    (W) => dumbbell(W, lifts),
    lifts.map((r) => [r.finding, String(r.n), pct(r.base_rate), pct(r.top5_rate), `${r.lift.toFixed(1)}×`]),
    ["Finding", "Test X-rays", "Library", "Top 5", "Lift"]);
  liftChart.draw();
  const card = lifts.find((r) => r.finding === "Cardiomegaly");
  root.append(block(
    "Strongest on findings that change the whole picture",
    [para(`When the X-ray shows an enlarged heart, ${pct(card.top5_rate)} of the five matches show one too, against ${pct(card.base_rate)} of the library. `,
      { b: "Large patterns such as heart size, lung volume and over-inflation are matched three to four times better than chance." },
      " Small, local findings (nodules, fractures, devices) are harder to match by overall appearance, which is why the report step also looks at the image itself.")],
    null, liftChart.box));

  // 3 · Drafting the report
  const rep = m.report, sys = Object.fromEntries(rep.systems.map((s) => [s.system, s]));
  const a = sys.agent_final, io = sys.image_only, nc = sys.nearest_case;
  const pd = Object.fromEntries(rep.paired.map((p) => [p.comparison, p]));
  const vsIO = pd["agent vs Gemini without retrieval"], vsNC = pd["agent vs copying the nearest case"];
  const reportChart = chartBox("Agreement with the radiologist's findings (F1)",
    `${rep.studies} held-out test X-rays, each with a real radiologist's report`,
    (W) => ciBars(W, [nc, io, a].map((s) => ({ label: s.label, v: s.micro_f1[0], lo: s.micro_f1[1], hi: s.micro_f1[2], ours: s.system === "agent_final" })),
      { max: 0.7, step: 0.1, metricName: "F1" }),
    [nc, io, a].map((s) => [s.label, f3(s.micro_f1[0]), f3(s.precision), f3(s.recall), pct(s.normal_abnormal_acc)]),
    ["System", "F1", "Precision", "Recall", "Normal/abnormal correct"]);
  reportChart.draw();
  const st = rep.agent;
  root.append(block(
    `Agrees with radiologists ${Math.round((a.micro_f1[0] / io.micro_f1[0] - 1) * 100)}% more than the same AI working alone`,
    [para("An automatic labeller (CheXbert, from Stanford) reads the AI's report and the radiologist's report and lists 14 standard findings in each. ",
      "F1 measures how well the two lists match, from 0 to 1. ",
      { b: `FindXRay beats copying the closest case's report by +${f2(vsNC.diff)} and the same Gemini model without similar cases by +${f2(vsIO.diff)}` },
      ". Both gains are statistically significant: their 95% confidence intervals sit above zero.")],
    [
      [f2(a.micro_f1[0]), `F1, against ${f2(io.micro_f1[0])} without similar cases and ${f2(nc.micro_f1[0])} for copying`],
      [pct(a.normal_abnormal_acc), `normal vs abnormal called correctly, against ${pct(io.normal_abnormal_acc)} without similar cases`],
      [`+${Math.round((a.precision / io.precision - 1) * 100)}%`, `precision: fewer findings the radiologist did not report (${f2(a.precision)} vs ${f2(io.precision)})`],
      [`${Math.round(st.agent_seconds_median)} s`, `median time for a full report, checks included (${Math.round(st.agent_seconds_p90)} s for 9 in 10)`],
    ],
    reportChart.box));

  root.append(el("p", { class: "fine", text:
    `What drives the gain: drafting with similar cases in view. The second look and sentence checks changed F1 by only +${f3(pd["effect of look-again + verification"].diff)}, `
    + `because a checker from the same model family rarely disagrees with the writer (it approved ${(st.first_pass_supported_mean * 100).toFixed(1)}% of sentences); `
    + `a different, stronger checker is the next step. ${rep.studies} studies means wide intervals, and CheXbert's labels are themselves automatic. `
    + `Report model: ${rep.model}.` }));
}

function dumbbell(W, rows) {
  const rowH = 36, top = 4, labelW = 0, right = 52, H = top + rows.length * rowH + 22;
  const max = 0.5;
  const x = (v) => labelW + (v / max) * (W - labelW - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": rows.map((r) => `${r.finding}: ${pct(r.top5_rate)} of matches vs ${pct(r.base_rate)} of library`).join("; ") });
  for (let t = 0; t <= max + 1e-9; t += 0.1) {
    s.append(svg("line", { x1: x(t), x2: x(t), y1: top, y2: H - 20, stroke: C.grid, "stroke-width": 1 }));
    s.append(svg("text", { x: x(t), y: H - 4, fill: C.axis, "font-size": 11, "text-anchor": "middle" }, pct(t)));
  }
  const marks = [];
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const cy = y + 25;
    s.append(svg("text", { x: labelW, y: y + 11, fill: C.muted, "font-size": 12.5 }, r.finding));
    s.append(svg("line", { x1: x(r.base_rate), x2: x(r.top5_rate), y1: cy, y2: cy, stroke: C.other, "stroke-width": 2 }));
    s.append(svg("circle", { cx: x(r.base_rate), cy, r: 4.5, fill: C.other, stroke: "#0A111B", "stroke-width": 2 }));
    s.append(svg("circle", { cx: x(r.top5_rate), cy, r: 5, fill: C.ours, stroke: "#0A111B", "stroke-width": 2 }));
    s.append(svg("text", { x: x(Math.max(r.top5_rate, r.base_rate)) + 10, y: cy + 4, fill: C.text, "font-size": 12.5, "font-weight": 600 }, `${r.lift.toFixed(1)}×`));
    const hit = svg("rect", { x: 0, y, width: W, height: rowH - 2, fill: "transparent" });
    hit.dataset.tip = `${r.finding}: ${pct(r.top5_rate)} of matches vs ${pct(r.base_rate)} of the library (${r.n} test X-rays)`;
    marks.push(hit);
    s.append(hit);
  });
  return { node: s, marks };
}
