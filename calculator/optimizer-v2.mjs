// This is an exploratory cost model. Estimates are not benchmark results.
export const defaults = {
  N: 1e9,
  D: 256,
  ram: 1e12,
  k: 12,
  probes: 32,
  router: "bits",
  leaf: 32,
  budget: 32768,
  keyMode: "full",
  keyBits: 24,
  candidates: 1000,
  hashNs: 100,
  queries: 1,
};
export const names = {
  A: "Cluster, then full scan",
  B: "Cluster, then bitplane branching",
  C: "Walk backwards · global hash",
};
export const targets = [0.8, 0.9, 0.95, 0.99];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function normalCdf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return clamp(0.5 * (1 + sign * erf), 0, 1);
}
export function normalQuantile(p) {
  let lo = -9,
    hi = 9;
  for (let i = 0; i < 55; i++) {
    const mid = (lo + hi) / 2;
    if (normalCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// Average over the whole true top-100 tail, not just the score at rank 100.
// Conditional normal scores are an assumption, not a model validated on this corpus.
const tailCache = new Map();
export function recallModel(rho, N, fraction) {
  if (fraction >= 1) return 1;
  if (fraction <= 0) return 0;
  const top = Math.min(1, 100 / N);
  if (rho === 0) return fraction;
  if (rho >= 1) return Math.min(1, fraction / top);
  let tail = tailCache.get(N);
  if (!tail) {
    tail = Array.from({ length: 48 }, (_, i) =>
      normalQuantile(1 - (top * (i + 0.5)) / 48),
    );
    tailCache.set(N, tail);
  }
  const threshold = normalQuantile(1 - fraction),
    deviation = Math.sqrt(1 - rho * rho);
  return (
    tail.reduce(
      (sum, z) => sum + normalCdf((rho * z - threshold) / deviation),
      0,
    ) / tail.length
  );
}

// Reported marginal mismatch rates from the imported notebook. Independence is NOT measured.
const mismatches = [
  [16, 0.3769],
  [24, 0.3857],
  [32, 0.3912],
  [48, 0.3933],
  [64, 0.3774],
  [128, 0.3674],
  [256, 0.3653],
  [768, 0.3696],
];
function mismatchRate(bits) {
  if (bits <= 16) return mismatches[0][1];
  for (let i = 1; i < mismatches.length; i++)
    if (bits <= mismatches[i][0]) {
      const [a, p] = mismatches[i - 1],
        [b, q] = mismatches[i];
      return p + ((q - p) * (bits - a)) / (b - a);
    }
  return mismatches.at(-1)[1];
}
const keyCache = new Map();
export function weightedKeys(bits) {
  if (keyCache.has(bits)) return keyCache.get(bits);
  // A representative half-normal magnitude profile, quantized to small positive integers.
  // Count every flip combination, including tied penalties, without enumerating 2^bits keys.
  const weights = Array.from({ length: bits }, (_, i) =>
    Math.max(1, Math.round(6 * normalQuantile(0.5 + (0.5 * (i + 0.5)) / bits))),
  );
  const p = mismatchRate(bits);
  let counts = [1],
    probability = [1];
  for (const w of weights) {
    const next = new Array(counts.length + w).fill(0),
      mass = new Array(counts.length + w).fill(0);
    for (let i = 0; i < counts.length; i++) {
      next[i] += counts[i];
      next[i + w] += counts[i];
      mass[i] += probability[i] * (1 - p);
      mass[i + w] += probability[i] * p;
    }
    counts = next;
    probability = mass;
  }
  let keys = 0,
    cdf = 0;
  const levels = counts.map((n, penalty) => {
    keys += n;
    cdf += probability[penalty];
    return { penalty, keys, cdf: Math.min(1, cdf) };
  });
  levels.at(-1).cdf = 1;
  const result = { weights, levels, p };
  keyCache.set(bits, result);
  return result;
}
export function collectKeys(N, bits, target) {
  const distribution = weightedKeys(bits);
  const wanted = Math.min(N, Math.max(100, target));
  const level =
    wanted === N
      ? distribution.levels.at(-1)
      : distribution.levels.find((v) => N * v.cdf >= wanted);
  const first = distribution.levels.find((v) => N * v.cdf >= 1);
  return {
    ...level,
    found: Math.min(N, N * level.cdf),
    firstPenalty: first.penalty,
    weights: distribution.weights,
  };
}

export function routing(s, rates) {
  const C = Math.min(s.N, 2 ** s.k),
    P = Math.min(C, s.probes);
  if (C === 1 || P === C)
    return {
      C,
      P,
      fraction: 1,
      docs: s.N,
      recall: 1,
      ns: 0,
      routeTerms: 0,
      rho: 1,
    };
  // Original load curves are only reported at C=4096, N=300k. Transfer is an assumption.
  const load = s.router === "bits" ? 18.034 * P ** -0.231 : 0.149 * P ** 0.223;
  const fraction = clamp((P / C) * load, 1 / s.N, 1);
  const rho = clamp(
    (s.router === "bits" ? 0.348 : 0.808) * Math.sqrt(s.k / 12),
    0,
    0.995,
  );
  // Centroids are floats: count float multiply-add terms, not binary lookup-table terms.
  const routeTerms = C * (s.router === "km" ? s.D : Math.ceil(s.k / 4));
  const scoring = routeTerms * (s.router === "km" ? 0.5 : rates.scoreNs);
  // Flat routing and top-P selection are both charged. Selection rate is an assumption.
  return {
    C,
    P,
    fraction,
    docs: s.N * fraction,
    recall: recallModel(rho, s.N, fraction),
    ns: scoring + C * 2,
    routeTerms,
    rho,
  };
}

export function branchWork(s, r, evidence) {
  const n = r.docs / r.P;
  if (n <= s.leaf) {
    const visited = s.budget ? Math.min(r.P, s.budget) : r.P;
    return {
      nodes: visited,
      splits: 0,
      leaves: visited,
      scored: n * visited,
      recall: visited / r.P,
      source: n,
    };
  }
  const all = evidence.rows.filter(
    (v) => v.phase === "development" && v.method === "branch",
  );
  const sizes = [...new Set(all.map((v) => v.pool_size))];
  const size = sizes.reduce((a, b) =>
    Math.abs(Math.log(b / n)) < Math.abs(Math.log(a / n)) ? b : a,
  );
  const scale = n / size,
    leafScale = 32 / s.leaf;
  const rows = all
    .filter((v) => v.pool_size === size)
    .sort((a, b) => a.mean_nodes - b.mean_nodes);
  const request = s.budget ? s.budget / r.P / scale / leafScale : Infinity;
  let low = {
      mean_nodes: 0,
      mean_bitplane_words: 0,
      mean_documents_scored: 0,
      mean_recall: 0,
    },
    high = rows.at(-1);
  for (const row of rows) {
    if (row.mean_nodes >= request) {
      high = row;
      break;
    }
    low = row;
  }
  const mix =
    request >= high.mean_nodes
      ? 1
      : (request - low.mean_nodes) / (high.mean_nodes - low.mean_nodes);
  const interpolate = (key) => low[key] + mix * (high[key] - low[key]);
  const nodes = interpolate("mean_nodes") * scale * leafScale * r.P;
  const splits =
    (interpolate("mean_bitplane_words") / Math.ceil(size / 64)) *
    scale *
    leafScale *
    r.P;
  return {
    nodes,
    splits,
    leaves: Math.max(0, nodes - splits),
    scored: Math.min(
      r.docs,
      interpolate("mean_documents_scored") * scale * r.P,
    ),
    recall: interpolate("mean_recall"),
    source: size,
  };
}

function finishMemory(components, queryBytes, s) {
  const index = Object.values(components).reduce((sum, v) => sum + v, 0);
  const scratch = queryBytes * s.queries,
    reserve = s.ram * 0.1;
  return {
    components,
    index,
    scratch,
    reserve,
    total: index + scratch + reserve,
    feasible: index + scratch + reserve <= s.ram,
    staticOver: index > s.ram,
  };
}

export function evaluate(s, method, evidence) {
  const rates = evidence.calibration;
  const r = routing(s, rates),
    codeBytes = Math.ceil(s.D / 64) * 8,
    G = Math.ceil(s.D / 4);
  const setup = rates.fixedMs * 1e6,
    score = (docs) => docs * G * rates.scoreNs;
  const topHeap = 100 * 16,
    query = s.D * 32 + G * 16 * 8 + topHeap;
  const common = {
    codes: s.N * codeBytes,
    documentIDs: s.N * 8,
    clusterOffsets: r.C > 1 ? (r.C + 1) * 8 : 0,
    centroids: s.router === "km" && r.C > 1 ? r.C * s.D * 4 : 0,
  };
  let rows = [],
    recall,
    docs,
    memory,
    detail = {};
  const row = (name, count, ns, explanation) =>
    rows.push({ name, count, ns, explanation });
  if (method === "A" || method === "B") {
    row(
      "Query setup",
      1,
      setup,
      "Build score tables and query ordering; fitted fixed cost.",
    );
    row(
      "Route and select probes",
      r.routeTerms,
      r.ns,
      `${r.C} cluster scores plus selection; float centroids use float arithmetic.`,
    );
    if (method === "A") {
      docs = r.docs;
      recall = r.recall;
      row(
        "Score documents and keep top 100",
        docs * G,
        score(docs),
        "Four signs per lookup; average top-100 selection cost is included in the fitted rate.",
      );
      memory = finishMemory(
        common,
        query + (r.P === r.C ? 0 : r.C * 16 + r.P * 8),
        s,
      );
    } else {
      const w = branchWork(s, r, evidence),
        W = Math.ceil(r.docs / r.P / 64);
      docs = w.scored;
      recall = r.recall * w.recall;
      row(
        "Split bitmap words",
        w.splits * W,
        w.splits * W * rates.wordNs,
        "Two ANDs, bit count, mask reads/writes and child initialization.",
      );
      row(
        "Leaf bitmap words",
        w.leaves * W,
        w.leaves * W * rates.wordNs,
        "Leaf masks are still full width; zero words are visited too.",
      );
      row(
        "Nodes and frontier",
        w.nodes,
        w.nodes * rates.nodeNs,
        "Pending-node selection and allocations, fitted average rate.",
      );
      row(
        "Score leaf documents",
        docs * G,
        score(docs),
        "Use the same binary score as A.",
      );
      // Sum ceil(cluster size/64) is bounded by floor((N+63C)/64), including padding.
      const planes = Math.floor((s.N + 63 * r.C) / 64) * s.D * 8;
      // Pending nodes are disjoint nonempty sets. At most P + visited nodes remain.
      // Use the whole selected pool as a largest-cluster bound, not the average bucket.
      const pending = Math.min(r.docs, r.P + (s.budget || r.docs));
      const largestMask = Math.ceil(r.docs / 64) * 8;
      const frontier = 2 * pending * (largestMask + 64) + 2 * largestMask;
      memory = finishMemory(
        { ...common, bitplanes: planes },
        query + r.C * 16 + frontier,
        s,
      );
      detail = {
        ...w,
        W,
        pending,
        largestMask,
        frontier,
        words: (w.splits + w.leaves) * W,
        logicalBytes: w.splits * W * 48 + w.leaves * W * 8 + docs * codeBytes,
      };
    }
  } else {
    const fullKey = s.keyMode === "full";
    const h = fullKey ? s.D : Math.min(s.D, s.keyBits);
    const kn = collectKeys(s.N, h, fullKey ? 100 : s.candidates);
    docs = kn.found;
    // Full-key search is exact when every flip set is visited in true penalty order
    // through the final score tie. The representative weights estimate COST only.
    // A short prefix plus a candidate quota has no justified recall estimate here.
    const exact = fullKey || s.candidates >= s.N;
    recall = exact ? 1 : null;
    row(
      "Query key and setup",
      s.D,
      setup + s.D * 0.3,
      fullKey ? "Full D-bit keys; exact query-weighted ordering through the top-100 tie." : "Global short-key candidates; no separate cluster routing.",
    );
    row(
      "Hash lookups, including empty keys",
      kn.keys,
      kn.keys * (s.hashNs + Math.ceil(h / 64) * 2),
      "A key can find no documents. Hash/key-word costs are assumptions.",
    );
    row(
      "Enumerate weighted flip sets",
      kn.keys,
      kn.keys * Math.log2(kn.keys + 1) * 5,
      "Queue allowance; enumerate all tied penalty combinations, not a single chain of flips.",
    );
    row(
      "Read postings",
      docs,
      docs * 2,
      "Read one 64-bit document ID per hit; 2 ns/ID is assumed.",
    );
    row(
      "Score candidates and keep top 100",
      docs * G,
      score(docs) * 1.2,
      "Scattered scoring uses a visible 1.2× allowance over the fitted scan rate.",
    );
    const occupied = Math.min(s.N, 2 ** h);
    // Sorted posting IDs are required even when many documents share one key.
    const hashSlots = Math.ceil(occupied / 0.7) * 32;
    const enumQueue = 2 * (kn.keys + 1) * 24;
    memory = finishMemory(
      { codes: s.N * codeBytes, documentIDs: s.N * 8, hashSlots },
      query + enumQueue,
      s,
    );
    detail = { ...kn, h, enumQueue, occupied, fullKey, exact };
  }
  const ns = rows.reduce((sum, v) => sum + v.ns, 0);
  return {
    method,
    settings: { ...s },
    ms: ns / 1e6,
    recall: recall === null ? null : clamp(Math.min(recall, docs / 100), 0, 1),
    docs,
    rows,
    memory,
    detail,
    routing: r,
  };
}

export function pareto(points) {
  const sorted = points.filter(p => Number.isFinite(p.recall)).sort((a, b) => b.recall - a.recall || a.ms - b.ms);
  const result = [];
  let fastest = Infinity;
  for (const p of sorted)
    if (p.ms < fastest) {
      result.push(p);
      fastest = p.ms;
    }
  return result.reverse();
}
export function bestAt(points, target, method) {
  return points
    .filter(
      (p) =>
        p.memory.feasible &&
        Number.isFinite(p.recall) &&
        p.recall >= target &&
        (!method || p.method === method),
    )
    .reduce((best, p) => (!best || p.ms < best.ms ? p : best), null);
}
export function sweep(base, evidence) {
  const all = [],
    records = [];
  let rejected = 0;
  const add = (s, m) => {
    const result = evaluate(s, m, evidence);
    records.push(result);
    if (result.memory.feasible) all.push(result);
    else rejected++;
  };
  const powers = [0, 6, 8, 10, 12, 14, 16, 18, 20].filter(
    (k) => 2 ** k <= base.N,
  );
  for (const router of ["bits", "km"])
    for (const k of powers) {
      if (k === 0 && router === "km") continue;
      for (let p = 0; p <= k; p++) {
        const s = { ...base, router, k, probes: 2 ** p };
        add(s, "A");
        for (const leaf of [32, 128, 512])
          for (const budget of [128, 2048, 32768, 0])
            add({ ...s, leaf, budget }, "B");
      }
    }
  // C has one full-key exact configuration and a separate short-key candidate grid.
  add({ ...base, keyMode: "full", keyBits: base.D, candidates: 100 }, "C");
  // Cluster parameters cannot change or duplicate global C results.
  for (const keyBits of [8, 12, 16, 24, 32, 48, 64].filter((h) => h <= base.D))
    for (const candidates of [
      ...new Set(
        [100, 1000, 10000, 100000, 0.01 * base.N, 0.1 * base.N, base.N].map(
          (n) => Math.min(base.N, n),
        ),
      ),
    ])
      add({ ...base, keyMode: "short", keyBits, candidates }, "C");
  return {
    all,
    records,
    rejected,
    front: pareto(all),
    tested: all.length + rejected,
  };
}
