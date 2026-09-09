import {
  defaults,
  searchPool,
  compare,
  predictRecordedTime,
  toySearch,
} from "./model.mjs";

const evidence = await fetch("./measurements.json").then((response) =>
  response.json(),
);
let settings = { ...defaults };
const example = toySearch({ documentCount: 16, leafSize: 2, budget: 64 });
let branchStep = 0;
let keyStep = 0;
const element = (id) => document.getElementById(id);
const count = (value) =>
  value.toLocaleString("en-US", { maximumFractionDigits: 0 });
const decimal = (value) =>
  Number(value.toPrecision(3)).toLocaleString("en-US", {
    maximumFractionDigits: 3,
  });
const time = (value) => `${decimal(value)} ms`;
const size = (value) =>
  value >= 1e9
    ? `${decimal(value / 1e9)} GB`
    : value >= 1e6
      ? `${decimal(value / 1e6)} MB`
      : `${decimal(value / 1000)} KB`;
const workRow = (label, value) =>
  `<div><span>${label}</span><strong>${value}</strong></div>`;
const colors = { scan: "#246bce", branch: "#b9503c", backwards: "#9b731f" };

function render() {
  const result = compare(settings, evidence);
  const { pool, scan, branch, backwards } = result;
  element("pool-flow").innerHTML =
    `<span>All documents<strong>${count(settings.documents)}</strong></span><i>→</i><span>Per group<strong>${count(pool.perGroup)}</strong></span><i>→</i><span>In opened groups<strong>${count(pool.selected)}</strong></span>`;
  element("pool-sentence").textContent =
    `Example: ${count(settings.documents)} documents ÷ ${count(pool.groups)} groups = about ${count(pool.perGroup)} per group. Open ${count(pool.opened)} ${pool.opened === 1 ? "group" : "groups"}, so the search starts with ${count(pool.selected)} documents.`;
  element("scan-time").textContent = time(scan.ms);
  element("branch-time").textContent = time(branch.ms);
  element("backwards-time").textContent = backwards.enoughExpected
    ? time(backwards.ms)
    : "Almost no matches";
  element("backwards-status").textContent =
    `${time(backwards.ms)} spent on ${count(backwards.keyLookups)} key attempts · assumed distribution`;

  element("scan-work").innerHTML =
    workRow("Documents scored", count(scan.documentsScored)) +
    workRow("Code data read", size(scan.bytesTouched)) +
    workRow("Extra stored data", "None");
  element("branch-work").innerHTML =
    workRow("Documents scored", `≈ ${count(branch.documentsScored)}`) +
    workRow("Group words read per split", count(result.wordsPerGroup)) +
    workRow("Extra data, whole database", size(branch.extraBytes));
  element("backwards-work").innerHTML =
    workRow("Keys tried per opened group", count(backwards.keysPerGroup)) +
    workRow(
      "Expected documents found",
      backwards.documentsScored < 0.01
        ? "Less than 0.01"
        : decimal(backwards.documentsScored),
    ) +
    workRow("Extra data, whole database", `≈ ${size(backwards.extraBytes)}`);
  element("split-price").textContent =
    `One split reads ${count(result.wordsPerGroup)} words, each holding bits for up to 64 documents. Its estimated cost is about ${decimal(branch.splitInDocuments)} full document scores. Splitting only helps if enough later work is avoided.`;

  const rates = evidence.calibration;
  element("scan-math").innerHTML =
    `<p><strong>${count(pool.selected)} documents × ${result.termsPerScore} small score lookups per document.</strong></p><p>Each lookup handles 4 bits. That is ${count(scan.scoreTerms)} lookups in total.</p><p>Estimated time = ${count(scan.scoreTerms)} × ${decimal(rates.scoreNs)} ns, plus ${time(result.setupMs)} setup and grouping. A million ns is 1 ms.</p><p>Each packed document takes ${result.bytesPerCode} bytes. Stored codes and document IDs take ${size(result.storedBase)} in the whole database.</p>`;
  element("branch-math").innerHTML =
    `<p>Every split walks <strong>ceil(${count(pool.perGroup)} / 64) = ${count(result.wordsPerGroup)} words</strong>. The word array does not shrink when fewer documents remain.</p><p>This example uses about ${count(branch.splits)} splits and ${count(branch.leaves)} leaf visits across the opened groups. A leaf is a small group whose remaining documents are scored.</p><p>That means ${count(branch.splitWords)} split-word visits + ${count(branch.leafWords)} leaf-word visits. A split word needs two AND operations, a bit count, and reads and writes of both child arrays.</p><p>Time = document scoring + word visits × ${decimal(rates.wordNs)} ns + node visits × ${decimal(rates.nodeNs)} ns + setup. The fitted costs already include average memory and queue effects.</p><p>The work counts come from our ${count(branch.recorded.pool_size)}-document validation run, scaled to this group size. They are an example, not a predicted recall.</p><p>Extra bitplanes: groups × bits × words per group × 8 bytes = ${size(branch.extraBytes)}. One queued group mask takes ${size(branch.nodeBytes)}; many can exist at once. Approximate repeated data traffic: ${size(branch.bytesTouched)}.</p>`;
  element("backwards-math").innerHTML =
    `<p><strong>There are 2^${settings.bits} possible full keys.</strong> Under independent, equally likely bits, one key matches an average of ${count(pool.perGroup)} / 2^${settings.bits} documents in a group.</p><p>After ${count(backwards.keysPerGroup)} different attempts per group, expected hits = ${count(pool.selected)} × ${count(backwards.keysPerGroup)} / 2^${settings.bits}.</p><p>With a long full key, almost all of those attempts can be empty. The displayed time is the cost of the attempts, not the time needed to find 100 candidates.</p><p>A flip costs |query value|. Try combinations in increasing total cost. Flipping bit 8, then bit 7, then both is different from only accumulating flips.</p><p>Time assumes 100 ns per key lookup, 2 ns per key word, and 5 ns per queue comparison, with about log₂(65,537) queue comparisons per attempted key. These costs have not been measured. The index allows 32 extra bytes per document for key metadata; queue memory is additional.</p>`;
  renderChart();
}

function renderChart() {
  const pool = searchPool(settings);
  const recordedSizes = [1000, 3000, 10000, 30000, 100000, 300000, 1000000];
  const sizes = recordedSizes.filter(
    (n) => n <= settings.documents / pool.opened,
  );
  if (!sizes.length) sizes.push(pool.perGroup);
  if (pool.perGroup > sizes[sizes.length - 1]) sizes.push(pool.perGroup);
  const points = sizes.map((n) => ({
    n,
    result: compare({ ...settings, groups: settings.documents / n }, evidence),
  }));
  const width = Math.max(280, Math.round(element("chart").clientWidth));
  const height = 310,
    left = 60,
    right = 20,
    top = 20,
    bottom = 50;
  const xLow = Math.floor(Math.log10(Math.min(...sizes))),
    xHigh = Math.max(xLow + 1, Math.ceil(Math.log10(Math.max(...sizes))));
  const values = points.flatMap((p) =>
    Object.keys(colors).map((id) => p.result[id].ms),
  );
  const yLow = Math.floor(Math.log10(Math.min(...values))),
    yHigh = Math.max(yLow + 1, Math.ceil(Math.log10(Math.max(...values))));
  const x = (n) =>
    left + ((Math.log10(n) - xLow) / (xHigh - xLow)) * (width - left - right);
  const y = (ms) =>
    top + ((yHigh - Math.log10(ms)) / (yHigh - yLow)) * (height - top - bottom);
  let content = "";
  const compact = (n) =>
    n >= 1e9
      ? `${decimal(n / 1e9)}B`
      : n >= 1e6
        ? `${decimal(n / 1e6)}M`
        : n >= 1000
          ? `${decimal(n / 1000)}K`
          : count(n);
  for (
    let exponent = xLow;
    exponent <= xHigh;
    exponent += width < 500 ? 2 : 1
  ) {
    content += `<line x1="${x(10 ** exponent)}" x2="${x(10 ** exponent)}" y1="${top}" y2="${height - bottom}" stroke="#e8ece3"/><text x="${x(10 ** exponent)}" y="${height - bottom + 22}" text-anchor="middle">${compact(10 ** exponent)}</text>`;
  }
  for (
    let exponent = yLow;
    exponent <= yHigh;
    exponent += Math.max(1, Math.ceil((yHigh - yLow) / 5))
  ) {
    content += `<line x1="${left}" x2="${width - right}" y1="${y(10 ** exponent)}" y2="${y(10 ** exponent)}" stroke="#e8ece3"/><text x="${left - 8}" y="${y(10 ** exponent) + 4}" text-anchor="end">${decimal(10 ** exponent)}</text>`;
  }
  for (const id of Object.keys(colors)) {
    const path = points
      .map((p, i) => `${i ? "L" : "M"}${x(p.n)},${y(p.result[id].ms)}`)
      .join(" ");
    content += `<path d="${path}" fill="none" stroke="${colors[id]}" stroke-width="2.5" ${id === "backwards" ? 'stroke-dasharray="6 5"' : ""}/>`;
    content += points
      .map(
        (p) =>
          `<circle cx="${x(p.n)}" cy="${y(p.result[id].ms)}" r="4" fill="${colors[id]}"><title>${compact(p.n)} per group: ${time(p.result[id].ms)}</title></circle>`,
      )
      .join("");
  }
  element("chart").innerHTML =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Estimated milliseconds against documents per group. Both axes use powers of ten. Walk-backwards is attempt cost only." style="font:11px Arial,sans-serif;fill:#66736e"><rect width="${width}" height="${height}" fill="white"/><text x="${left}" y="12">Milliseconds</text>${content}<text x="${width / 2}" y="${height - 4}" text-anchor="middle">Documents per group · 1K = 1,000</text></svg>`;
}

function renderSteps() {
  const node = example.trace[branchStep];
  element("scan-dots").innerHTML = Array.from(
    { length: 16 },
    () => '<span class="active"></span>',
  ).join("");
  element("branch-dots").innerHTML = example.codes
    .map(
      (_, row) =>
        `<span class="${node.rows.includes(row) ? "active" : ""}" title="Document ${row + 1}"></span>`,
    )
    .join("");
  element("branch-step").textContent = node.leaf
    ? `Step ${branchStep + 1}: score the ${node.rows.length} remaining documents.`
    : `Step ${branchStep + 1}: ${node.rows.length} documents. Split on bit ${node.bit + 1}; prefer ${example.query[node.bit] >= 0 ? 1 : 0}.`;
  const flip = example.flips[keyStep],
    key = example.ideal ^ flip.mask;
  element("key-bits").innerHTML = example.query
    .map(
      (_, bit) =>
        `<span class="${(flip.mask >> bit) & 1 ? "flipped" : ""}">${(key >> bit) & 1}</span>`,
    )
    .join("");
  const changed = example.query
    .map((_, bit) => ((flip.mask >> bit) & 1 ? bit + 1 : null))
    .filter(Boolean);
  const hits = example.codes.filter((code) => code === key).length;
  element("key-step").textContent =
    `${changed.length ? `Flip bit${changed.length > 1 ? "s" : ""} ${changed.join(" + ")}` : "Start with the query’s own signs"}. Total penalty ${decimal(flip.penalty)}. Found ${hits} ${hits === 1 ? "document" : "documents"}.`;
}
element("branch-next").addEventListener("click", () => {
  branchStep = (branchStep + 1) % example.trace.length;
  renderSteps();
});
element("key-next").addEventListener("click", () => {
  keyStep = (keyStep + 1) % example.flips.length;
  renderSteps();
});
for (const key of ["documents", "groups", "opened", "bits"]) {
  const input = element(key);
  input.addEventListener("input", () => {
    const value = Number(input.value);
    const min = input.min ? Number(input.min) : 1,
      max = input.max ? Number(input.max) : Infinity;
    if (!input.value || !Number.isFinite(value) || value < min || value > max)
      return;
    settings[key] = Math.round(value);
    render();
  });
  input.addEventListener("change", () => {
    input.value = settings[key];
    render();
  });
}
element("reset").addEventListener("click", () => {
  settings = { ...defaults };
  branchStep = 0;
  keyStep = 0;
  for (const key of Object.keys(settings)) element(key).value = settings[key];
  render();
  renderSteps();
});
element("save-chart").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob([element("chart").innerHTML], { type: "image/svg+xml" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "three-search-methods.svg";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const checks = evidence.rows.filter((row) => row.phase === "evaluation");
const error = Math.max(
  ...checks.map((row) =>
    Math.abs(
      predictRecordedTime(row, evidence.calibration) / row.api_p50_ms - 1,
    ),
  ),
);
element("calibration-check").textContent =
  `Our timing formula differs from the recorded test times by up to ${(error * 100).toFixed(1)}%. It is a rough estimate. The saved run did not establish a branch speedup at the tested recall targets.`;
window.addEventListener("resize", renderChart);
render();
renderSteps();
