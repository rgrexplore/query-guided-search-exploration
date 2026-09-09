import {
  names,
  colors,
  defaults,
  pool,
  calculate,
  branchCounts,
  loadMeasurement,
  predictMeasurement,
  toySearch,
  binomialCdf,
} from "./model.mjs";

const evidence = await fetch("./measurements.json").then((response) =>
  response.json(),
);
let state = defaults(evidence.calibration);
let selected = "dense";
let visible = new Set(Object.keys(names));
let currentToy;
const byId = (id) => document.getElementById(id);
const number = (value) => {
  if (value === 0) return "0";
  if (Math.abs(value) < 0.01 || Math.abs(value) >= 1e12)
    return value.toExponential(2);
  return value.toLocaleString("en-US", {
    maximumFractionDigits: value < 10 ? 2 : 0,
  });
};
const ms = (value) =>
  value < 0.001
    ? `${number(value * 1000)} µs`
    : `${Number(value.toPrecision(4)).toLocaleString("en-US", { maximumFractionDigits: 4 })} ms`;
const bytes = (value) => {
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  const index =
    value > 0 ? Math.min(5, Math.max(0, Math.floor(Math.log2(value) / 10))) : 0;
  return `${number(value / 1024 ** index)} ${units[index]}`;
};
const percent = (value) => `${(value * 100).toFixed(2)}%`;

// A definition is just a label, range, and explanation. No generated form framework.
const fields = {
  n: [
    "Documents N",
    1000,
    1e10,
    1000,
    "Total stored documents. The graph varies this value.",
    true,
  ],
  d: [
    "Dimensions D",
    64,
    2048,
    64,
    "One sign bit per document dimension. Queries keep their float values.",
  ],
  clusters: [
    "Clusters C",
    1,
    1e6,
    1,
    "Assume equally sized clusters. Their actual sizes can be uneven.",
    true,
  ],
  routeBits: [
    "Routing bits r",
    0,
    20,
    1,
    "r bits allow 2^r buckets. This is sign routing, not IVF.",
  ],
  probes: [
    "Clusters to visit P",
    1,
    1e6,
    1,
    "Capped at C. More visited clusters mean more documents to search.",
    true,
  ],
  routingMs: [
    "Routing time (ms)",
    0,
    100,
    0.01,
    "A separate assumption for selecting clusters. Not free, and not measured by the no-routing run.",
  ],
  nodes: [
    "Nodes per visited cluster B",
    1,
    1e6,
    1,
    "Total visits, including splits and leaves. This is a workload assumption, not a recall target.",
    true,
  ],
  splitFraction: [
    "Fraction of nodes that split",
    0,
    1,
    0.01,
    "0.69 means about 69% split; the rest score leaf documents.",
  ],
  scoredFraction: [
    "Fraction of documents scored",
    0,
    1,
    0.01,
    "Held constant when changing N. The model does not derive this from B. Load measured counts below.",
  ],
  leaf: [
    "Documents allowed per leaf",
    1,
    1024,
    1,
    "Caps scored documents at leaf visits × leaf size. Changing this does not predict a new tree.",
    true,
  ],
  active: [
    "Active documents per split",
    1,
    1e6,
    1,
    "Sparse model only: assumed average across visited split nodes. We have not measured this average yet.",
    true,
  ],
  filterBits: [
    "Filter dimensions K",
    1,
    2048,
    1,
    "Use the K largest query magnitudes, capped at D.",
  ],
  weightBits: [
    "Bits per query weight",
    1,
    8,
    1,
    "Round |query values| to integers from 0 to 2^b − 1 for the prefilter.",
  ],
  shortlist: [
    "Fraction surviving the filter",
    0,
    1,
    0.01,
    "Exact-scored afterward. This is an assumed output of the threshold, not a quality estimate.",
  ],
  hashProbes: [
    "Full-key lookup budget",
    1,
    1e12,
    1,
    "Count all keys checked, even empty ones. Keys are ordered by weighted mismatch penalty.",
    true,
  ],
  keyBits: [
    "Short-key bits h",
    1,
    24,
    1,
    "A smaller key gives more documents per key. Independent of the outer routing bits.",
  ],
  radius: [
    "Allowed flips per key",
    0,
    16,
    1,
    "Hamming radius: each flipped bit counts equally. Used by short-key and substring probing.",
  ],
  tables: [
    "Substring tables m",
    1,
    64,
    1,
    "Split D bits across m disjoint tables. More tables mean more stored IDs and duplicate candidates.",
  ],
  fixedMs: [
    "Fixed cost (ms)",
    0,
    10,
    0.001,
    "Fitted average setup and call overhead at D=256, top 100. Not a guarantee at another D.",
  ],
  scoreNs: [
    "Time per score term (ns)",
    0.001,
    100,
    0.001,
    "Fitted scan cost per four-sign lookup term, including average result selection.",
  ],
  wordNs: [
    "Dense bitmap word (ns)",
    0.001,
    100,
    0.001,
    "Fitted average per split-or-leaf word visit. Split and leaf instruction counts are shown separately.",
  ],
  nodeNs: [
    "Queue / allocation per node (ns)",
    0,
    10000,
    1,
    "Fitted average for the existing dense implementation. Sparse uses this only as an assumption.",
  ],
  sparseWordNs: [
    "Sparse word visit (ns)",
    0.01,
    1000,
    0.01,
    "Unmeasured. Includes ID access and building child lists; start higher than dense-word cost.",
  ],
  logicNs: [
    "Counter operation (ns)",
    0.01,
    100,
    0.01,
    "Unmeasured word-level logical operation with counter reads and writes.",
  ],
  hashNs: [
    "Key / dedup lookup (ns)",
    1,
    10000,
    1,
    "Unmeasured. Cache misses, collisions, and table layout can change this substantially.",
  ],
  queueNs: [
    "Key queue comparison (ns)",
    0.01,
    100,
    0.01,
    "Unmeasured. Full-key enumeration uses a log2(budget + 1) queue-work approximation.",
  ],
};
function field(key) {
  const [label, min, max, step, help, logarithmic] = fields[key];
  const slider = logarithmic
    ? `min="${Math.log10(Math.max(1, min))}" max="${Math.log10(max)}" step="0.02"`
    : `min="${min}" max="${max}" step="${step}"`;
  return `<div class="field"><div class="field-head"><label for="${key}">${label}</label><input id="${key}" data-key="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${state[key]}"></div><input aria-label="${label} slider" data-slider="${key}" type="range" ${slider}><p class="help">${help}</p></div>`;
}
function section(title, keys, open = false, description = "") {
  return `<details class="control-section" ${open ? "open" : ""}><summary>${title}${description ? `<span>${description}</span>` : ""}</summary>${keys.map(field).join("")}</details>`;
}
byId("shared-controls").innerHTML =
  `<section class="control-section"><h3>Shared inputs</h3>${field("n")}${field("d")}<div class="field"><label for="routing">First stage</label><select id="routing"><option value="none">No routing · search all documents</option><option value="ivf">Similarity clusters · IVF</option><option value="sign">Sign-bit buckets · 2^r</option></select></div><div id="cluster-input">${field("clusters")}</div><div id="sign-input">${field("routeBits")}</div><div id="routing-inputs">${field("probes")}${field("routingMs")}</div></section>`;
byId("method-controls").innerHTML =
  section(
    "Branch workload",
    ["nodes", "splitFraction", "scoredFraction", "leaf", "active"],
    true,
    "Measured preset; editable assumptions",
  ) +
  section("Prefilter", ["filterBits", "weightBits", "shortlist"]) +
  section("Key probing", ["hashProbes", "keyBits", "radius", "tables"]);
byId("rate-controls").innerHTML = [
  "fixedMs",
  "scoreNs",
  "wordNs",
  "nodeNs",
  "sparseWordNs",
  "logicNs",
  "hashNs",
  "queueNs",
]
  .map(field)
  .join("");

function syncInputs() {
  for (const key of Object.keys(fields)) {
    byId(key).value = state[key];
    const slider = document.querySelector(`[data-slider="${key}"]`);
    slider.value = fields[key][5]
      ? Math.log10(Math.max(1, state[key]))
      : state[key];
  }
  byId("routing").value = state.routing;
  byId("cluster-input").hidden = state.routing !== "ivf";
  byId("sign-input").hidden = state.routing !== "sign";
  byId("routing-inputs").hidden = state.routing === "none";
}
for (const input of document.querySelectorAll("[data-key], [data-slider]")) {
  input.addEventListener("change", () => {
    const key = input.dataset.key || input.dataset.slider;
    const [, min, max, step, , logarithmic] = fields[key];
    const value =
      input.dataset.slider && logarithmic
        ? Math.round(10 ** Number(input.value))
        : Number(input.value);
    state[key] = Math.min(
      max,
      Math.max(min, step >= 1 ? Math.round(value) : value),
    );
    syncInputs();
    render();
  });
  if (input.type === "range")
    input.addEventListener("input", () =>
      input.dispatchEvent(new Event("change")),
    );
  else
    input.addEventListener("input", () => {
      // Let someone finish typing before clamping an incomplete number.
      const key = input.dataset.key;
      const value = Number(input.value);
      if (
        input.value === "" ||
        !Number.isFinite(value) ||
        value < fields[key][1] ||
        value > fields[key][2]
      )
        return;
      state[key] = value;
      const slider = document.querySelector(`[data-slider="${key}"]`);
      slider.value = fields[key][5] ? Math.log10(Math.max(1, value)) : value;
      render();
    });
}
byId("routing").addEventListener("change", (event) => {
  state.routing = event.target.value;
  syncInputs();
  render();
});
byId("reset").addEventListener("click", () => {
  state = defaults(evidence.calibration);
  syncInputs();
  render();
});
byId("restore-rates").addEventListener("click", () => {
  Object.assign(state, evidence.calibration);
  syncInputs();
  render();
});
byId("chart-legend").innerHTML = Object.keys(names)
  .map(
    (id) =>
      `<label><input type="checkbox" checked data-series="${id}"><span class="swatch" style="background:${colors[id]}"></span>${names[id]}</label>`,
  )
  .join("");
for (const input of document.querySelectorAll("[data-series]"))
  input.addEventListener("change", () => {
    input.checked
      ? visible.add(input.dataset.series)
      : visible.delete(input.dataset.series);
    renderChart();
  });

function renderChart() {
  const width = Math.max(280, Math.round(byId("chart").clientWidth)),
    height = 350,
    left = 66,
    right = 22,
    top = 22,
    bottom = 48;
  const xs = Array.from({ length: 71 }, (_, i) => 10 ** (3 + i / 10));
  const series = [...visible].map((id) => ({
    id,
    points: xs
      .map((n) => ({ n, result: calculate({ ...state, n }, id) }))
      .filter((p) => p.result.feasible)
      .map((p) => ({ n: p.n, value: p.result.ms })),
  }));
  const values = series.flatMap((s) => s.points.map((p) => p.value));
  const low = values.length ? Math.floor(Math.log10(Math.min(...values))) : -2;
  const high = values.length
    ? Math.max(low + 1, Math.ceil(Math.log10(Math.max(...values))))
    : 2;
  const x = (n) => left + ((Math.log10(n) - 3) / 7) * (width - left - right);
  const y = (v) =>
    top + ((high - Math.log10(v)) / (high - low)) * (height - top - bottom);
  let grid = "";
  for (
    let exponent = low;
    exponent <= high;
    exponent += Math.max(1, Math.ceil((high - low) / 6))
  ) {
    const yy = y(10 ** exponent);
    grid += `<line x1="${left}" x2="${width - right}" y1="${yy}" y2="${yy}" stroke="#e7eae3"/><text x="${left - 10}" y="${yy + 4}" text-anchor="end">${number(10 ** exponent)}</text>`;
  }
  for (let exponent = 3; exponent <= 10; exponent += width < 500 ? 2 : 1) {
    const xx = x(10 ** exponent);
    grid += `<line x1="${xx}" x2="${xx}" y1="${top}" y2="${height - bottom}" stroke="#eef0ea"/><text x="${xx}" y="${height - bottom + 23}" text-anchor="middle">${["1K", "10K", "100K", "1M", "10M", "100M", "1B", "10B"][exponent - 3]}</text>`;
  }
  const lines = series
    .filter((s) => s.points.length)
    .map(
      ({ id, points }) =>
        `<path d="${points.map((p, i) => `${i ? "L" : "M"}${x(p.n).toFixed(2)},${y(p.value).toFixed(2)}`).join(" ")}" fill="none" stroke="${colors[id]}" stroke-width="${id === selected ? 3 : 1.8}" ${["full", "prefix", "mih"].includes(id) ? 'stroke-dasharray="5 4"' : ""}/><circle cx="${x(state.n)}" cy="${y(calculate(state, id).ms)}" r="${calculate(state, id).feasible ? 4 : 0}" fill="${colors[id]}"><title>${names[id]}: ${ms(calculate(state, id).ms)} at ${number(state.n)} documents</title></circle>`,
    )
    .join("");
  byId("chart").innerHTML =
    `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Estimated search time versus documents; logarithmic axes; not matched recall" viewBox="0 0 ${width} ${height}" style="font:11px -apple-system,BlinkMacSystemFont,Arial,sans-serif;fill:#63716f"><rect width="${width}" height="350" fill="white"/>${grid}<text x="${left}" y="12">Estimated ms</text><line x1="${x(state.n)}" x2="${x(state.n)}" y1="${top}" y2="${height - bottom}" stroke="#81907d" stroke-dasharray="3 4"/>${lines}<text x="${width / 2}" y="${height - 4}" text-anchor="middle">Total documents N · current selection ${number(state.n)}</text></svg>`;
  byId("chart-note").textContent =
    `The measured range was 1K–1M documents at 256 bits. Outside it, these are extrapolations. Branch visit counts and requested survivor fractions stay fixed; leaf capacity caps scored documents. Impossible branch workloads are omitted. Dashed curves assume uniform hash keys.`;
}
function renderInspector() {
  const result = calculate(state, selected),
    p = pool(state),
    b = branchCounts(state);
  byId("method-title").textContent = result.name;
  byId("method-quality").textContent = result.quality;
  byId("method-note").textContent =
    (result.feasible
      ? ""
      : "Conflicting workload: there are more nonempty leaf visits than documents scored. Reduce visits, increase the fraction that splits, or increase the scored fraction. The time below is only the cost of the entered counts. ") +
    result.note;
  const first =
    state.routing === "none"
      ? ["Start with all documents", `${number(state.n)} documents · C = 1`]
      : [
          "Choose clusters",
          `P = ${number(p.probes)} of C = ${number(p.clusters)}`,
        ];
  const steps = {
    scan: [
      first,
      [
        "Read packed codes",
        `${number(p.selected)} codes × ${Math.ceil(state.d / 64) * 8} bytes`,
      ],
      ["Score every code", `${Math.ceil(state.d / 4)} terms per document`],
      ["Keep top 100", "Same binary score throughout"],
    ],
    dense: [
      first,
      ["Choose next branch", `B = ${number(state.nodes)} visits per cluster`],
      [
        "Split or score a leaf",
        `${number(b.words)} words per node, even if sparse`,
      ],
      [
        "Score survivors",
        `${number(result.docs)} documents · leaf ≤ ${state.leaf}`,
      ],
    ],
    sparse: [
      first,
      ["Choose next branch", "Same penalty ordering as dense"],
      [
        "Read occupied words",
        `Assume ${number(state.active)} active documents per split`,
      ],
      ["Score survivors", `${number(result.docs)} documents`],
    ],
    prefilter: [
      first,
      [
        "Read K bitplanes",
        `K = ${Math.min(state.filterBits, state.d)} · ${state.weightBits}-bit weights`,
      ],
      ["Add / compare counters", "Includes carries and a threshold pass"],
      ["Score survivors", `${percent(state.shortlist)} assumed to survive`],
    ],
    full: [
      first,
      ["Construct ideal sign key", `${state.d} signs from the query`],
      [
        "Probe flip combinations",
        `${number(state.hashProbes)} keys per cluster, including empty keys`,
      ],
      [
        "Score found documents",
        `${number(result.docs)} expected under uniform keys`,
      ],
    ],
    prefix: [
      first,
      ["Use a short key", `h = ${Math.min(state.keyBits, state.d)} bits`],
      ["Probe nearby keys", `At most ${state.radius} flips per key`],
      ["Score complete codes", `${number(result.docs)} expected documents`],
    ],
    mih: [
      first,
      ["Split the key", `m = ${state.tables} substring tables`],
      [
        "Probe and combine IDs",
        `Radius ${state.radius} in each table; remove duplicates`,
      ],
      [
        "Score complete codes",
        `${number(result.docs)} expected unique documents`,
      ],
    ],
  };
  byId("flow").innerHTML = steps[selected]
    .map(
      ([title, description], i) =>
        `<div class="flow-step" style="border-color:${colors[selected]}"><small>0${i + 1}</small><strong>${title}</strong><span>${description}</span></div>`,
    )
    .join("");
  const labels = {
    fixed: "Fixed setup",
    routing: "Routing",
    scoring: "Scoring",
    bitmaps: "Bitmap / counters",
    queue: "Branch queue",
    hashing: "Keys / dedup / queue",
  };
  byId("time-parts").innerHTML = Object.entries(result.parts)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `<span>${labels[k]}<b>${ms(v)}</b></span>`)
    .join("");
  byId("formula-rows").innerHTML = result.rows
    .map(
      (row) =>
        `<tr><td>${row.label}</td><td>${row.unit === "bytes" ? bytes(row.value) : number(row.value)}${row.unit === "bytes" ? "" : ` <span class="subtext">${row.unit}</span>`}</td><td>${row.formula}</td></tr>`,
    )
    .join("");
  byId("memory-note").textContent =
    `Stored payload: ${bytes(result.indexBytes)}. ${["dense", "sparse"].includes(selected) ? `One full-width node can need ${bytes(result.temporary)} for its bitmap representation; many nodes can be queued at once. Sparse occupancy changes actual per-node size.` : `Additional modeled scratch or candidate storage: ${bytes(result.temporary)}.`} These are layout estimates, not process peak RAM. Dense traffic includes zero-initializing both child masks. Cache reuse means logical bytes are not DRAM bytes; their cost is already inside the time coefficients.`;
}
function render() {
  const p = pool(state);
  byId("total-docs").textContent = number(state.n);
  byId("bucket-docs").textContent = number(p.bucket);
  byId("pool-docs").textContent = number(p.selected);
  byId("routing-note").textContent =
    state.routing === "none"
      ? "No first-stage filter. Every method starts from the same full document pool."
      : `Balanced-cluster assumption: N / C documents per cluster, N × P / C selected. Routing adds ${ms(state.routingMs)}. Neither balance nor routing recall is guaranteed.${p.bucket < 100 ? " Fewer than 100 documents per cluster on average; empty and uneven buckets can matter a lot." : ""}`;
  byId("comparison").innerHTML = Object.keys(names)
    .map((id) => {
      const r = calculate(state, id);
      return `<tr class="${id === selected ? "selected" : ""}"><td><button class="method-button" data-method="${id}"><span class="swatch" style="background:${colors[id]}"></span>${r.name}</button><span class="subtext">${["scan", "dense"].includes(id) ? "Uses fitted time rates" : "Unmeasured approach"}</span></td><td>${r.feasible ? ms(r.ms) : "Conflicting counts"}${r.feasible ? "" : '<span class="subtext warning">inspect this method</span>'}</td><td>${number(r.docs)}${r.docs < 100 ? '<span class="subtext warning">fewer than 100 expected</span>' : ""}</td><td>${bytes(r.traffic)}</td><td>${bytes(r.indexBytes)}<span class="subtext">excludes runtime overhead</span></td></tr>`;
    })
    .join("");
  for (const button of document.querySelectorAll("[data-method]"))
    button.addEventListener("click", () => {
      selected = button.dataset.method;
      render();
    });
  renderChart();
  renderInspector();
  renderEvidence();
  renderQuality();
}

const pools = [...new Set(evidence.rows.map((row) => row.pool_size))].sort(
  (a, b) => a - b,
);
byId("evidence-pool").innerHTML = pools
  .map((n) => `<option value="${n}">${number(n)} documents</option>`)
  .join("");
byId("evidence-pool").value = 1e6;
byId("evidence-pool").addEventListener("change", renderEvidence);
byId("evidence-scope").textContent =
  `${evidence.hardware}. ${evidence.scope}. Validation: 100 queries; test: 200 queries. Three timed repeats. This is the local C++ implementation, not Exa's service.`;
function renderEvidence() {
  const rows = evidence.rows.filter((r) => r.phase === "evaluation");
  const scan = rows.find((r) => r.pool_size === 1e6 && r.method === "scan");
  const branch = rows.find(
    (r) =>
      r.pool_size === 1e6 && r.method === "branch" && r.node_budget === 32768,
  );
  const errors = rows.map((r) =>
    Math.abs(predictMeasurement(r, state) / r.api_p50_ms - 1),
  );
  byId("evidence-summary").innerHTML =
    `<div><span>1M · exact packed scan</span><strong>${ms(scan.api_p50_ms)}</strong><span>Measured median · binary top 100</span></div><div><span>1M · dense branch, budget 32,768</span><strong>${ms(branch.api_p50_ms)}</strong><span>Measured median · ${percent(branch.mean_recall)} recall</span></div><div><span>Largest held-out timing error</span><strong>${percent(Math.max(...errors))}</strong><span>At current coefficients · over all ${rows.length} test settings</span></div>`;
  const n = Number(byId("evidence-pool").value);
  const sweep = evidence.rows
    .filter((r) => r.phase === "development" && r.pool_size === n)
    .sort(
      (a, b) =>
        a.method.localeCompare(b.method) || a.node_budget - b.node_budget,
    );
  byId("budget-rows").innerHTML = sweep
    .map(
      (r, i) =>
        `<tr><td>${r.method === "scan" ? "Exact scan" : `Branch · ${r.node_budget ? number(r.node_budget) : "unlimited"}`}</td><td>${ms(r.api_p50_ms)}</td><td>${percent(r.mean_recall)}</td><td>${r.method === "branch" ? `<button data-measurement="${i}">Load work</button>` : ""}</td></tr>`,
    )
    .join("");
  for (const button of document.querySelectorAll("[data-measurement]"))
    button.addEventListener("click", () => {
      state = loadMeasurement(state, sweep[Number(button.dataset.measurement)]);
      syncInputs();
      render();
      byId("status").textContent =
        "Loaded observed branch counts. Time remains a model; compare it with the measured time in the evidence table.";
      byId("total-docs").scrollIntoView({
        block: "center",
        behavior: "instant",
      });
    });
  byId("check-rows").innerHTML = rows
    .filter((r) => r.pool_size === n)
    .map((r) => {
      const predicted = predictMeasurement(r, state),
        error = (predicted / r.api_p50_ms - 1) * 100;
      return `<tr><td>${number(n)} / ${r.method}${r.method === "branch" ? `<span class="subtext">budget ${r.node_budget || "unlimited"}</span>` : ""}</td><td>${ms(r.api_p50_ms)}</td><td>${ms(predicted)}</td><td>${error >= 0 ? "+" : ""}${error.toFixed(1)}%</td></tr>`;
    })
    .join("");
}

function renderQuality() {
  const p = Number(byId("mismatch").value);
  byId("mismatch-value").textContent = percent(p);
  const prefix = binomialCdf(Math.min(state.keyBits, state.d), state.radius, p);
  const tables = Math.min(state.tables, state.d),
    short = Math.floor(state.d / tables),
    longer = state.d % tables;
  let miss = 1;
  for (let i = 0; i < tables; i++)
    miss *= 1 - binomialCdf(short + Number(i < longer), state.radius, p);
  byId("quality-estimates").innerHTML =
    `<div><span>Short-key inclusion · toy model</span><strong>${percent(prefix)}</strong></div><div><span>Substring union inclusion · toy model</span><strong>${percent(1 - miss)}</strong></div>`;
}
byId("mismatch").addEventListener("input", renderQuality);

function renderToy(recompute = true) {
  if (recompute) {
    const probability = Number(byId("toy-probability").value),
      budget = Number(byId("toy-budget").value);
    currentToy = toySearch({
      probability,
      budget,
      seed: Number(byId("toy-seed").value),
    });
    byId("toy-probability-value").textContent = percent(probability);
    byId("toy-budget-value").textContent = budget;
    byId("toy-step").max = currentToy.trace.length - 1;
    byId("toy-step").value = Math.min(
      Number(byId("toy-step").value),
      currentToy.trace.length - 1,
    );
  }
  const step = Number(byId("toy-step").value),
    node = currentToy.trace[step];
  byId("toy-step-value").textContent =
    `${step + 1} / ${currentToy.trace.length}`;
  byId("previous-step").disabled = step === 0;
  byId("next-step").disabled = step === currentToy.trace.length - 1;
  byId("query-values").innerHTML = currentToy.query
    .map(
      (q, i) =>
        `<div class="${!node.leaf && node.bit === i ? "current" : ""}"><span>bit ${i + 1}</span><strong>${q > 0 ? "+" : ""}${q}</strong></div>`,
    )
    .join("");
  byId("toy-recall").textContent =
    `Toy recall@10: ${percent(currentToy.recall)}`;
  byId("toy-count").textContent =
    `${currentToy.trace.filter((n) => n.leaf).reduce((s, n) => s + n.rows.length, 0)} documents scored`;
  byId("toy-step-description").textContent =
    `${node.explore ? "Random pending branch" : "Lowest-penalty pending branch"} · penalty ${number(node.penalty)}. ${node.leaf ? `Score these ${node.rows.length} documents.` : `Split ${node.rows.length} documents on bit ${node.bit + 1}; prefer ${currentToy.query[node.bit] >= 0 ? 1 : 0}.`}`;
  byId("document-dots").innerHTML = currentToy.codes
    .map(
      (_, i) =>
        `<span class="${node.rows.includes(i) ? "active " : ""}${currentToy.exact.includes(i) ? "relevant" : ""}" title="Document ${i}${node.rows.includes(i) ? ", active" : ""}${currentToy.exact.includes(i) ? ", exact top 10" : ""}"></span>`,
    )
    .join("");
  byId("ideal-key").textContent = currentToy.query
    .map((q) => (q >= 0 ? "1" : "0"))
    .join("");
  byId("flip-list").innerHTML = currentToy.flips
    .slice(0, 8)
    .map((f) => {
      const bits = currentToy.query
        .map((_, i) => ((f.mask >> i) & 1 ? i + 1 : null))
        .filter(Boolean);
      const key = currentToy.ideal ^ f.mask;
      const hits = currentToy.codes.filter((code) => code === key).length;
      return `<div>${bits.length ? `Flip ${bits.join(", ")}` : "No flips"} · ${number(f.penalty)}<small>${hits} matching documents</small></div>`;
    })
    .join("");
}
for (const id of ["toy-probability", "toy-budget", "toy-seed"])
  byId(id).addEventListener("input", () => renderToy());
byId("toy-step").addEventListener("input", () => renderToy(false));
byId("previous-step").addEventListener("click", () => {
  byId("toy-step").value = Number(byId("toy-step").value) - 1;
  renderToy(false);
});
byId("next-step").addEventListener("click", () => {
  byId("toy-step").value = Number(byId("toy-step").value) + 1;
  renderToy(false);
});

function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
byId("export").addEventListener("click", () =>
  download(
    JSON.stringify(
      {
        assumptions: state,
        estimates: Object.keys(names).map((id) => calculate(state, id)),
        evidenceSource: evidence.source_sha256,
      },
      null,
      2,
    ),
    "search-cost-scenario.json",
    "application/json",
  ),
);
byId("save-chart").addEventListener("click", () =>
  download(byId("chart").innerHTML, "search-cost-curves.svg", "image/svg+xml"),
);
window.addEventListener("resize", renderChart);
syncInputs();
render();
renderToy();
