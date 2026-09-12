import {
  defaults,
  names,
  targets,
  evaluate,
  sweep,
  bestAt,
} from "./optimizer-v2.mjs";
import { predictRecordedTime } from "./model.mjs";
const evidence = await fetch("./measurements.json").then((r) => r.json());
const recallStudy=await fetch("./recall-validation.json").then(r=>r.json());
let state = { ...defaults },
  currentSweep,
  cacheKey = "";
const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");
const count = (n) =>
  n >= 1e12 ? n.toExponential(2) : nf.format(Math.round(n));
const short = (n) =>
  n >= 1e9
    ? `${(n / 1e9).toFixed(2)}B`
    : n >= 1e6
      ? `${(n / 1e6).toFixed(2)}M`
      : n >= 1e3
        ? `${(n / 1e3).toFixed(1)}k`
        : n.toFixed(n < 10 ? 1 : 0);
const pct = n => n===null ? "unmeasured" : `${(100*n).toFixed(2)}%`;
const ms = (n) =>
  n > 1e9
    ? `${(n / 86400000).toExponential(1)} days`
    : n >= 1000
      ? `${(n / 1000).toFixed(2)} s`
      : n >= 1
        ? `${n.toFixed(2)} ms`
        : `${(n * 1000).toFixed(1)} µs`;
const bytes = (n) => {
  const units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
  const i = Math.min(
    6,
    Math.max(0, Math.floor(Math.log10(Math.max(1, n)) / 3)),
  );
  return `${(n / 1000 ** i).toFixed(i ? 2 : 0)} ${units[i]}`;
};
const color = { A: "var(--a)", B: "var(--b)", C: "var(--c)" };
const memoryNames = {
  codes: "Packed document codes",
  documentIDs: "Document IDs / postings",
  clusterOffsets: "Cluster offsets",
  centroids: "Float centroids",
  bitplanes: "Bitplanes + padding bound",
  hashSlots: "Hash slots at 70% load",
};
const controls = {
  N: ["N", (n) => Math.round(10 ** n), (n) => Math.log10(n)],
  D: ["D", Number, Number],
  RAM: ["ram", (n) => 10 ** n, (n) => Math.log10(n)],
  Q: ["queries", Number, Number],
  k: ["k", Number, Number],
  P: ["probes", (n) => 2 ** n, (n) => Math.log2(n)],
  L: ["leaf", (n) => 2 ** n, (n) => Math.log2(n)],
  B: [
    "budget",
    (n) => (n >= 21 ? 0 : 2 ** n),
    (n) => (n === 0 ? 21 : Math.log2(n)),
  ],
  Dl: ["keyBits", Number, Number],
  T: ["candidates", (n) => Math.round(10 ** n), (n) => Math.log10(n)],
  H: ["hashNs", Number, Number],
};
function sync() {
  $("keyMode").value=state.keyMode;
  for (const [id, [property, , toControl]] of Object.entries(controls))
    $(id).value = toControl(state[property]);
  for (const button of document.querySelectorAll("#rt button"))
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.v === state.router),
    );
}
function labels(r) {
  $("Dl").disabled=state.keyMode==="full";
  $("T").disabled=state.keyMode==="full";
  $("vN").textContent = count(state.N);
  $("vD").textContent = `${state.D} bits`;
  $("vRAM").textContent = bytes(state.ram);
  $("vk").textContent =
    `${short(r.C)} clusters${state.router === "bits" ? ` · ${state.k} bits` : ""}`;
  $("vP").textContent = count(r.P);
  $("vL").textContent = count(state.leaf);
  $("vB").textContent = state.budget ? count(state.budget) : "no limit";
  $("vDl").textContent = `${state.keyMode==="full"?state.D:Math.min(state.keyBits,state.D)} bits`;
  $("vT").textContent = count(state.keyMode==="full"?100:Math.min(state.N,state.candidates));
  $("vH").textContent = `${state.hashNs} ns`;
  $("kWhy").textContent =
    state.router === "bits"
      ? `First ${state.k} sign bits assign documents to 2^${state.k} clusters. Example: prefix 101 selects cluster 5 when k=3.`
      : "Each cluster has a float centroid. Route by comparing the query with centroids; this is not binary lookup-table scoring.";
  $("rtHint").textContent =
    `Modeled selected pool: ${short(r.docs)} documents (${pct(r.fraction)} of N). Cluster imbalance is transferred from the supplied 4,096-cluster experiment; it is not measured at these settings.`;
}
function diagram(p) {
  const common = 'font-size="12" fill="var(--muted)"';
  if (p.method === "A") {
    const blocks = Array.from(
      { length: 16 },
      (_, i) =>
        `<rect x="${12 + i * 35}" y="12" width="28" height="26" rx="2" fill="${i === 5 || i === 6 ? "var(--a)" : "var(--sunk)"}" stroke="var(--rule)"/>`,
    ).join("");
    const docs = Array.from(
      { length: 24 },
      (_, i) => `<circle cx="${220 + i * 14}" cy="103" r="4" fill="var(--a)"/>`,
    ).join("");
    return `<svg viewBox="0 0 620 138" role="img" aria-label="Schematic cluster selection followed by scoring every selected document">${blocks}<text x="12" y="62" ${common}>${count(p.routing.P)} of ${short(p.routing.C)} clusters probed · schematic</text><text x="12" y="108" ${common}>${short(p.docs)} documents scored</text>${docs}</svg>`;
  }
  if (p.method === "B") {
    const node = (x, y) =>
      `<rect x="${x}" y="${y}" width="38" height="18" rx="2" fill="var(--sunk)" stroke="var(--b)"/><rect x="${x}" y="${y + 21}" width="80" height="4" fill="var(--b)" opacity=".6"/>`;
    return `<svg viewBox="0 0 620 155" role="img" aria-label="Branching reduces active documents but each node retains a full-width bitmap"><path d="M290 30L150 54M290 30L430 54M150 80L60 104M150 80L225 104M430 80L370 104M430 80L520 104" stroke="var(--rule)" fill="none"/>${node(272, 5)}${node(132, 54)}${node(412, 54)}${node(42, 104)}${node(207, 104)}${node(352, 104)}${node(502, 104)}<text x="365" y="17" ${common}>Active sets get smaller.</text><text x="365" y="35" ${common}>The bitmap stays full width.</text><text x="12" y="151" ${common}>${short(p.detail.W)} bitmap words per node in the load model</text></svg>`;
  }
  return `<svg viewBox="0 0 620 160" role="img" aria-label="Global query key and increasing weighted penalties; no cluster routing">${[20, 40, 60].map((radius) => `<circle cx="98" cy="76" r="${radius}" fill="none" stroke="var(--rule)" stroke-dasharray="4 4"/>`).join("")}<circle cx="98" cy="76" r="4" fill="var(--c)"/><text x="182" y="34" ${common}>Global ${p.detail.h}-bit key from sign(q)</text><text x="182" y="58" ${common}>Try flip sets by total weighted penalty.</text><text x="182" y="89" ${common}>First expected hit: penalty ${p.detail.firstPenalty}</text><text x="182" y="114" ${common}>Candidate target: penalty ${p.detail.penalty} · ${short(p.detail.keys)} keys</text><text x="182" y="140" ${common}>Query weights are quantized for this model.</text></svg>`;
}
function memoryStrip(p) {
  const m = p.memory;
  return `<div class="memory-strip"><div class="summary"><b>${m.feasible ? "Fits the RAM model" : m.staticOver ? "Index alone exceeds RAM" : "Peak memory bound exceeds RAM"}</b><span class="mono">${bytes(m.total)} / ${bytes(state.ram)}</span></div><div class="stack"><i style="width:${Math.min(100, (100 * m.index) / state.ram)}%;background:${color[p.method]}"></i><i style="width:${Math.min(100, (100 * m.scratch) / state.ram)}%;background:var(--b);opacity:.55"></i><i style="width:10%;background:var(--faint);opacity:.3"></i></div><div class="legend"><span>index ${bytes(m.index)}</span><span>query scratch bound ${bytes(m.scratch)}</span><span>runtime reserve ${bytes(m.reserve)}</span></div></div>`;
}
function card(p) {
  const descriptions = {
    A: "Probe clusters, then score every document inside them. Recall can be lost when a relevant document is in an unprobed cluster.",
    B: "Use the same clusters. Split by the highest-|q| bit and visit the lowest-penalty pending node first. Both branches remain available.",
    C: "One global hash table. Full-key exact search and short-key candidate search are different variants. Start at sign(q), then try flip combinations with larger weighted penalties. No separate clusters or probes.",
  };
  const notes = {
    A: () => `Scores ${short(p.docs)} documents. Selecting all clusters gives exact binary search; otherwise the displayed recall is modeled.`,
    B: () => `${short(p.detail.splits)} estimated splits and ${short(p.detail.leaves)} leaf visits. Both read full-width masks. Node counts come from the nearest archived pool (${short(p.detail.source)} documents), with a modeled leaf-size adjustment.`,
    C: () => `The candidate target is ${count(state.keyMode==="full"?100:Math.min(state.N,state.candidates))}, but completing the last penalty level yields about ${short(p.docs)}. The first-hit penalty is a separate calculation. Marginal bit mismatch rates do not prove independence or real recall.`,
  };
  const rows = p.rows
    .map(
      (r, i) =>
        `<tr><td class="step">${i + 1}</td><td>${r.name}<div class="why">${r.explanation}</div></td><td class="mono r">${short(r.count)}</td><td class="mono r">${ms(r.ns / 1e6)}</td><td class="mono r">${pct(r.ns / (p.ms * 1e6))}</td></tr>`,
    )
    .join("");
  const ledger = p.rows
    .map(
      (r, i) =>
        `<i style="width:${(100 * r.ns) / (p.ms * 1e6)}%;opacity:${0.35 + (0.65 * i) / Math.max(1, p.rows.length - 1)}"></i>`,
    )
    .join("");
  return `<section class="card ${p.method}"><div class="card-h"><div><h3 class="card-t">${names[p.method]} <span class="flag neu">estimate</span>${p.memory.feasible ? "" : ' <span class="flag bad">RAM excluded</span>'}</h3><p class="card-d">${descriptions[p.method]}</p></div><div class="big"><div class="n mono">${ms(p.ms)}</div><div class="u">per query · single thread</div><div class="rc mono">${p.method==="C"?(p.detail.exact?"Binary recall@100 = 100% if exact search completes":"Short-key recall: unmeasured at these settings"):`modeled recall@100 ≈ ${pct(p.recall)}`}</div><div class="rc mono">index ${bytes(p.memory.index)}</div></div></div><div class="dia">${diagram(p)}</div><div class="bar">${ledger}</div><div class="legend">${p.rows.map((r) => `<span>${r.name} · ${pct(r.ns / (p.ms * 1e6))}</span>`).join("")}</div>${memoryStrip(p)}<details><summary>Step by step, with your numbers</summary><div class="tw"><table><thead><tr><th>Step</th><th>What happens</th><th class="r">Work count</th><th class="r">Time</th><th class="r">Share</th></tr></thead><tbody>${rows}<tr class="tot"><td></td><td>Total</td><td></td><td class="mono r">${ms(p.ms)}</td><td class="mono r">100%</td></tr></tbody></table></div></details><p class="note">${notes[p.method]()}</p></section>`;
}
function memoryTable(points) {
  const lines = points.map((p) => {
    const m = p.memory;
    return `<h3 style="color:${color[p.method]};margin:16px 0 5px">${names[p.method]}</h3>${Object.entries(
      m.components,
    )
      .map(([name, value]) =>
        memoryRow(memoryNames[name], value, state.ram, color[p.method]),
      )
      .join(
        "",
      )}${memoryRow(`Query scratch bound × ${state.queries}`, m.scratch, state.ram, "var(--b)")}${memoryRow("Runtime reserve · 10%", m.reserve, state.ram, "var(--faint)")}${memoryRow("Total for RAM check", m.total, state.ram, m.feasible ? "var(--ok)" : "var(--crit)")}`;
  });
  return (
    lines.join("") +
    '<p class="why">For B, the frontier bound uses the entire selected pool as a largest-cluster bound and allows vector capacity growth. Exceeding this conservative estimate does not prove an implementation cannot fit; it excludes the configuration until peak memory is measured. Cluster loads are themselves modeled. These figures exclude index construction and original float embeddings.</p>'
  );
}
function memoryRow(label, value, budget, tint) {
  return `<div class="memrow"><div class="memlab">${label}</div><div class="membar"><i style="width:${Math.min(100, (100 * value) / budget)}%;background:${tint}"></i></div><div class="memval mono">${bytes(value)}</div></div>`;
}
function plot(sw) {
  const known=sw.all.filter(p=>Number.isFinite(p.recall));
  if (!known.length)
    return '<p class="why"><strong>No configuration passes the RAM check.</strong> The table has no winner. Increase RAM or shorten the document codes.</p>';
  const W = Math.max(400, $("pareto").clientWidth - 32),
    H = 330,
    L = 68,
    R = 18,
    T = 24,
    B = 48;
  const times = known.map((p) => p.ms),
    lo = Math.floor(Math.log10(Math.min(...times))),
    hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...times))));
  const x = (rec) => L + (W - L - R) * rec,
    y = (ms) => T + ((H - T - B) * (hi - Math.log10(ms))) / (hi - lo);
  let axes = "";
  for (let e = lo; e <= hi; e += Math.max(1, Math.ceil((hi - lo) / 6)))
    axes += `<line x1="${L}" x2="${W - R}" y1="${y(10 ** e)}" y2="${y(10 ** e)}" stroke="var(--rule2)"/><text x="${L - 8}" y="${y(10 ** e) + 4}" text-anchor="end" class="plot-label">${ms(10 ** e)}</text>`;
  for (const rec of [0, 0.2, 0.4, 0.6, 0.8, 1])
    axes += `<text x="${x(rec)}" y="${H - B + 20}" text-anchor="middle" class="plot-label">${Math.round(rec * 100)}%</text>`;
  // Bin separately for each method so overlapping methods are not erased by a faster one.
  const bins = new Map();
  for (const p of known) {
    const key = `${p.method}:${Math.round(x(p.recall) / 2)}:${Math.round(y(p.ms) / 2)}`;
    if (!bins.has(key)) bins.set(key, p);
  }
  const points = [...bins.values()]
    .map(
      (p) =>
        `<circle cx="${x(p.recall)}" cy="${y(p.ms)}" r="2" fill="${color[p.method]}" opacity=".35"><title>${names[p.method]} · ${pct(p.recall)} estimated · ${ms(p.ms)} · ${bytes(p.memory.total)}</title></circle>`,
    )
    .join("");
  const line = sw.front
    .map((p, i) => `${i ? "L" : "M"}${x(p.recall)} ${y(p.ms)}`)
    .join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Modeled recall versus estimated latency for configurations passing the RAM model">${axes}${points}<path d="${line}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>${targets.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" stroke="var(--rule)" stroke-dasharray="3 4"/>`).join("")}<text x="${W / 2}" y="${H - 7}" class="plot-label" text-anchor="middle">Modeled recall@100 · not measured recall</text><text x="${L}" y="14" class="plot-label">Estimated latency</text></svg><div class="legend">${["A", "B", "C"].map((m) => `<span><i class="dot" style="background:${color[m]}"></i>${names[m]}</span>`).join("")}<span>line = lowest estimates in the grid</span></div>`;
}
function configText(p) {
  const s = p.settings;
  return p.method === "C"
    ? `global · ${s.keyMode==="full"?"full-key exact":s.candidates>=s.N?"short-key exhaustive":"short-key candidates"} · ${s.keyMode==="full"?s.D:s.keyBits} bits · ${count(s.keyMode==="full"?100:s.candidates)} candidates`
    : `${s.router === "km" ? "k-means" : "sign-bit"} · ${count(2 ** s.k)} clusters · ${count(s.probes)} probes${p.method === "B" ? ` · leaf ${s.leaf} · budget ${s.budget ? count(s.budget) : "unlimited"}` : ""}`;
}
let choices = [];
function targetRows(sw) {
  choices = [];
  return targets
    .map((t) => {
      const per = ["A", "B", "C"].map((m) => bestAt(sw.all, t, m)),
        best = bestAt(sw.all, t);
      const cells = per.map((p) => {
        if (!p)
          return '<td class="r">No qualifying grid point<div class="target-help">RAM or modeled recall limit</div></td>';
        const id = choices.push(p) - 1;
        return `<td class="r"><strong class="mono">${ms(p.ms)}</strong><div class="target-help">estimated recall ${pct(p.recall)}<br>${configText(p)}<br>index ${bytes(p.memory.index)}<br>RAM check ${bytes(p.memory.total)}</div><button class="load" data-choice="${id}">Inspect</button></td>`;
      });
      return `<tr><td><b>${Math.round(t * 100)}%</b></td>${cells.join("")}<td>${best ? `<strong style="color:${color[best.method]}">${names[best.method]}</strong><div class="target-help">Model only · needs benchmark</div>` : "No feasible selection"}</td></tr>`;
    })
    .join("");
}
function render() {
  const current = ["A", "B", "C"].map((m) => evaluate(state, m, evidence));
  labels(current[0].routing);
  // Keep open explanations open while a parameter changes.
  const open = [...document.querySelectorAll("#cards details")].map(
    (d) => d.open,
  );
  $("cards").innerHTML = current.map(card).join("");
  document
    .querySelectorAll("#cards details")
    .forEach((d, i) => (d.open = Boolean(open[i])));
  $("mem").innerHTML = memoryTable(current);
  $("dense-memory").innerHTML =
    [16, 24, 32, 48, 64, 128, 256]
      .map((h) =>
        memoryRow(
          `${h}-bit dense key · 8-byte offsets`,
          2 ** h * 8,
          state.ram,
          "var(--c)",
        ),
      )
      .join("") +
    '<p class="why">This is a directory-size comparison, not a claim that a dense table stores the exact answer for every possible float query. Float query magnitudes still affect ranking.</p>';
  const key = [state.N, state.D, state.ram, state.hashNs, state.queries].join(
    "|",
  );
  if (key !== cacheKey) {
    currentSweep = sweep(state, evidence);
    cacheKey = key;
  }
  $("pareto").innerHTML = plot(currentSweep);
  $("targets").querySelector("tbody").innerHTML = targetRows(currentSweep);
  $("sweepNote").textContent =
    `${count(currentSweep.tested)} configurations evaluated; ${count(currentSweep.all.length)} pass the RAM model; ${count(currentSweep.rejected)} are excluded. N=${count(state.N)}, D=${state.D}, RAM=${bytes(state.ram)}. Runtime reserve is included. ${currentSweep.all.filter(p=>p.recall===null).length} RAM-feasible short-key settings have unknown recall and are not ranked. No claim of measured optimality at this scale.`;
  for (const b of document.querySelectorAll("[data-choice]"))
    b.addEventListener("click", () => {
      state = { ...choices[Number(b.dataset.choice)].settings };
      sync();
      render();
      $("cards").scrollIntoView({ block: "start" });
    });
}
$("keyMode").addEventListener("input",()=>{state.keyMode=$("keyMode").value;render();});
for (const [id, [property, fromControl]] of Object.entries(controls))
  $(id).addEventListener("input", () => {
    state[property] = fromControl(Number($(id).value));
    render();
  });
for (const b of document.querySelectorAll("#rt button"))
  b.addEventListener("click", () => {
    state.router = b.dataset.v;
    sync();
    render();
  });
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  $("theme").textContent = dark ? "Dark theme" : "Light theme";
});
$("reset").addEventListener("click", () => {
  state = { ...defaults };
  sync();
  render();
});
$("download").addEventListener("click", () => {
  const fields = [
    "method",
    "recall",
    "recall_basis",
    "key_mode",
    "estimated_ms",
    "index_bytes",
    "scratch_bound_bytes",
    "reserve_bytes",
    "total_bytes",
    "fits_ram",
    "N",
    "D",
    "clusters",
    "probes",
    "leaf",
    "node_budget",
    "key_bits",
    "candidates",
    "source",
  ];
  const csv = [
    fields.join(","),
    ...currentSweep.records.map((p) => {
      const s = p.settings,
        m = p.memory;
      return [
        p.method,
        p.recall===null?"":p.recall,
        p.method==="C"?(p.detail.exact?"exact_if_completed":"unmeasured"):"model",
        p.method==="C"?s.keyMode:"",
        p.ms,
        m.index,
        m.scratch,
        m.reserve,
        m.total,
        m.feasible,
        s.N,
        s.D,
        p.method === "C" ? "" : 2 ** s.k,
        p.method === "C" ? "" : s.probes,
        p.method === "B" ? s.leaf : "",
        p.method === "B" ? s.budget : "",
        p.method === "C" ? s.keyBits : "",
        p.method === "C" ? s.candidates : "",
        "model_not_benchmark",
      ].join(",");
    }),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "modeled-recall-latency-ram.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const testRows = evidence.rows.filter((r) => r.phase === "evaluation");
const error = Math.max(
  ...testRows.map((r) =>
    Math.abs(predictRecordedTime(r, evidence.calibration) / r.api_p50_ms - 1),
  ),
);
$("timing-check").innerHTML =
  `<p class="why">Largest relative timing error across ${testRows.length} held-out summaries: <strong>${(error * 100).toFixed(1)}%</strong>. This does not bound error for new structures or larger corpora.</p><table class="small-table"><thead><tr><th>1M test case</th><th>Measured</th><th>From its work counts</th></tr></thead><tbody>${testRows
    .filter((r) => r.pool_size === 1e6)
    .map(
      (r) =>
        `<tr><td>${r.method}${r.method === "branch" ? ` · ${r.node_budget || "unlimited"}` : ""}</td><td>${ms(r.api_p50_ms)}</td><td>${ms(predictRecordedTime(r, evidence.calibration))}</td></tr>`,
    )
    .join("")}</tbody></table>`;
$("assumptions").innerHTML =
  `<p><strong>Timing rates:</strong> ${evidence.calibration.scoreNs.toFixed(3)} ns per four-sign score term, ${evidence.calibration.wordNs.toFixed(3)} ns per bitmap word visit, ${evidence.calibration.nodeNs.toFixed(2)} ns per node, and ${ms(evidence.calibration.fixedMs)} fixed setup. These aggregate rates include average CPU/memory work. They are not instruction latencies.</p><p><strong>Unmeasured rates:</strong> float centroid term 0.5 ns; cluster-selection entry 2 ns; hash cost from the control plus 2 ns/key word; enumeration comparison 5 ns; posting ID read 2 ns; scattered scoring 1.2× scan. No GPU or production-throughput claim.</p><p><strong>RAM:</strong> 8-byte aligned packed codes, 64-bit IDs even for C, offsets, float centroids, bitplanes including a padding bound, hash entries at 70% load. B includes pending masks and vector growth; C includes posting IDs and an enumeration-queue bound. Scratch scales with concurrent queries. Ten percent of RAM is reserved. Construction-time copies and stored original float vectors are outside this index model.</p><p><strong>Recall:</strong> a hypothetical joint-normal full score and routing/key score. Average over the entire true top-100 tail, rather than conditioning on only the 100th score. Routing correlation (.348 or .808 at 4,096 clusters) and load curves come from the supplied model and remain unverified transfer assumptions. Branch conditional recall comes from archived binary runs and is transferred to selected clusters. Their product is an approximation, not a guaranteed recall.</p><p><strong>C:</strong> a global table; changing A/B clusters cannot change its candidate target. Quantized representative query magnitudes define weighted flip penalties. A dynamic count includes every combination at a penalty and the corresponding independent-bit probability. It is not a uniform-key model and not an unweighted Hamming-ball formula. A quantized representative query is still an assumption.</p><p><strong>Corrections:</strong> leaf mask work restored; actual split/leaf ratios used; centroid scoring separated from binary scoring; IDs and query scratch added to RAM; one memory total shared by cards, graph and table; C swept once; failed points retained in CSV; no “fastest” badge across unequal recall; float and binary evidence no longer conflated.</p>`;

$("recall-study").innerHTML=recallStudy.cases.map(c=>`<h3>${c.name} · ${count(c.documents)} documents</h3><div class="tw"><table><thead><tr><th>Method / work setting</th><th>Measured recall</th><th>Normal prediction</th><th>Key attempts</th></tr></thead><tbody><tr><td>A · full scan</td><td>100%</td><td>Exact reference</td><td>—</td></tr>${c.summaries.map(r=>`<tr><td>${r.method==="C"?`C · first 24 bits · ${count(r.target)} candidates`:`B · node budget ${count(r.budget)}`}</td><td>${pct(r.measured_recall)}</td><td>${r.method==="C"?`${pct(r.predicted_recall)} (${r.prediction_gap_points>=0?'+':''}${r.prediction_gap_points.toFixed(1)} points)`:'—'}</td><td>${r.method==="C"?count(r.median_key_attempts):'—'}</td></tr>`).join('')}</tbody></table></div>`).join('')+'<p class="why">96 native full scans agree with an independent exact score sort. C includes all prefix keys at the final score tie. Empty-key counts use a checked two-half enumeration count; they are not measured hash-search times. This study isolates search with no outer clustering.</p>';

window.addEventListener("resize", () => {
  $("pareto").innerHTML = plot(currentSweep);
});
sync();
render();
