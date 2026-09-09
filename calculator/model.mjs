// Work counts live here; the page only formats them. Times are estimates, not a CPU simulator.
export const names = {
  scan: "Packed scan",
  dense: "Dense bitplanes",
  sparse: "Sparse bitplanes",
  prefilter: "Bitplane prefilter",
  full: "Full-key probing",
  prefix: "Short-key probing",
  mih: "Multi-index hashing",
};
export const colors = {
  scan: "#246bce",
  dense: "#bf4e39",
  sparse: "#16836c",
  prefilter: "#8d65aa",
  full: "#947025",
  prefix: "#d07424",
  mih: "#677881",
};

export function defaults(calibration) {
  return {
    n: 1e6,
    d: 256,
    routing: "none",
    clusters: 64,
    routeBits: 6,
    probes: 1,
    routingMs: 0.05,
    nodes: 32768,
    splitFraction: 22557.88 / 32768,
    scoredFraction: 0.163766425,
    leaf: 32,
    active: 256,
    filterBits: 64,
    weightBits: 4,
    shortlist: 0.03,
    hashProbes: 65536,
    keyBits: 16,
    radius: 1,
    tables: 16,
    sparseWordNs: 2,
    logicNs: 0.5,
    hashNs: 100,
    queueNs: 5,
    ...calibration,
  };
}

export function pool(s) {
  const clusters =
    s.routing === "none"
      ? 1
      : s.routing === "sign"
        ? 2 ** s.routeBits
        : s.clusters;
  const probes = s.routing === "none" ? 1 : Math.min(s.probes, clusters);
  return {
    clusters,
    probes,
    bucket: s.n / clusters,
    selected: (s.n * probes) / clusters,
  };
}

// Number of bitstrings within r flips. D <= 2048 is fine in probability space.
// Keeping volume in log2 avoids trying to represent 2^2048 as a JS number.
export function logAdd2(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const high = Math.max(a, b);
  return high + Math.log2(1 + 2 ** (Math.min(a, b) - high));
}
export function logBall(bits, radius) {
  let term = 0;
  let sum = 0;
  for (let i = 1; i <= Math.min(bits, radius); i++) {
    term += Math.log2((bits - i + 1) / i);
    sum = logAdd2(sum, term);
  }
  return Math.min(bits, sum);
}
export function binomialCdf(bits, radius, probability) {
  if (radius >= bits || probability === 0) return 1;
  if (probability === 1) return 0;
  let term = bits * Math.log2(1 - probability);
  let sum = term;
  for (let i = 1; i <= radius; i++) {
    term +=
      Math.log2((bits - i + 1) / i) +
      Math.log2(probability / (1 - probability));
    sum = logAdd2(sum, term);
  }
  return Math.min(1, 2 ** sum);
}

export function occupiedWords(documents, active) {
  // Assume surviving IDs are scattered uniformly, not neatly packed together.
  // Count the short final word separately.
  const count = Math.min(documents, Math.max(0, active));
  if (documents <= 0 || count === 0) return 0;
  const full = Math.floor(documents / 64);
  const tail = documents - full * 64;
  const chanceEmpty = Math.log1p(-count / documents);
  const occupied = (size) => (size === 0 ? 0 : -Math.expm1(size * chanceEmpty));
  return full * occupied(64) + occupied(tail);
}

export function branchCounts(s) {
  const p = pool(s);
  const splits = s.nodes * s.splitFraction * p.probes;
  const leaves = s.nodes * (1 - s.splitFraction) * p.probes;
  const words = Math.ceil(p.bucket / 64);
  const docs = Math.min(p.selected * s.scoredFraction, leaves * s.leaf);
  return { splits, leaves, words, docs, nodes: splits + leaves };
}

export function calculate(s, id) {
  const p = pool(s);
  const codeBytes = Math.ceil(s.d / 64) * 8;
  const groups = Math.ceil(s.d / 4);
  const b = branchCounts(s);
  const words = Math.ceil(p.bucket / 64);
  const baseIndex = s.n * (codeBytes + 8); // Packed rows plus 64-bit document IDs.
  const planes = p.clusters * s.d * words * 8;
  const routing = s.routing === "none" ? 0 : s.routingMs;
  const rows = [];
  const add = (label, value, formula, unit = "visits") =>
    rows.push({ label, value, formula, unit });
  let docs = p.selected,
    wordWork = 0,
    wordCost = 0,
    nodeWork = 0,
    extraMs = 0;
  let traffic = 0,
    indexBytes = baseIndex,
    temporary = 0,
    note = "",
    quality = "Not predicted";
  let keyCount = 0;

  if (id === "scan") {
    traffic = docs * codeBytes;
    quality =
      s.routing === "none"
        ? "Exact for this binary score"
        : "Exact inside selected clusters";
    note =
      "Read each selected code once, score four signs per lookup, then keep the best 100.";
  }
  if (id === "dense" || id === "sparse") {
    docs = b.docs;
    nodeWork = b.nodes;
    let splitWords = b.splits * words;
    let leafWords = b.leaves * words;
    if (id === "sparse") {
      splitWords = b.splits * occupiedWords(p.bucket, s.active);
      const activeAtLeaf = b.leaves ? docs / b.leaves : 0;
      leafWords = b.leaves * occupiedWords(p.bucket, activeAtLeaf);
    }
    wordWork = splitWords + leafWords;
    wordCost = id === "dense" ? s.wordNs : s.sparseWordNs;
    add(
      "Split bitmap words",
      splitWords,
      id === "dense"
        ? "split nodes × ceil(documents per cluster / 64)"
        : "split nodes × expected occupied words",
    );
    add(
      "Leaf bitmap words",
      leafWords,
      id === "dense"
        ? "leaf nodes × ceil(documents per cluster / 64)"
        : "leaf nodes × expected occupied words",
    );
    add(
      "AND operations",
      splitWords * 2,
      "2 × split bitmap words",
      "operations",
    );
    add(
      "Population counts",
      splitWords,
      "1 × split bitmap words",
      "operations",
    );
    add(
      "Nodes removed from queue",
      nodeWork,
      "nodes per cluster × probed clusters",
    );
    // Dense: active + plane reads, two child writes, and two zero-initialized vectors.
    // Sparse: index/mask pairs add work; 64 B per split word is a layout assumption.
    traffic =
      splitWords * (id === "dense" ? 48 : 64) +
      leafWords * (id === "dense" ? 8 : 16) +
      docs * codeBytes;
    indexBytes += planes;
    temporary = words * (id === "dense" ? 8 : 16);
    note =
      id === "dense"
        ? "Each node still has a full-cluster bitmap. Zero words cost time too. Node cost includes an average allowance for the queue and allocations."
        : "Store only nonempty (word ID, mask) pairs. Fewer words, but extra IDs, less sequential access, and list construction. Active documents per split is an assumption to measure.";
  }
  if (id === "prefilter") {
    const bits = Math.min(s.filterBits, s.d);
    const counterBits = Math.ceil(
      Math.log2(bits * (2 ** s.weightBits - 1) + 1),
    );
    const planeWords = bits * words * p.probes;
    const addOperations = planeWords * s.weightBits * counterBits * 5;
    const compareOperations = words * p.probes * counterBits * 3;
    wordWork = addOperations + compareOperations;
    wordCost = s.logicNs;
    docs = p.selected * s.shortlist;
    add(
      "Input plane words",
      planeWords,
      "filter dimensions × cluster words × probes",
    );
    add(
      "Counter bits",
      counterBits,
      "ceil(log2(K × (2^weight bits − 1) + 1))",
      "bits",
    );
    add(
      "Counter / threshold operations",
      wordWork,
      "5 × plane words × weight bits × counter bits + 3 × cluster words × probes × counter bits",
      "operations",
    );
    indexBytes += planes;
    temporary = counterBits * words * 8;
    // One plausible full-width ripple-carry implementation, not one instruction per 64 scores.
    traffic = planeWords * 8 + wordWork * 24 + docs * codeBytes;
    note =
      "Add quantized query-weight penalties in bit-sliced counters, filter by a threshold, then score survivors with the original query. The survivor fraction is assumed; K alone cannot tell us how many survive. This counts a simple carry-based schedule, not an optimized kernel.";
  }
  if (id === "full" || id === "prefix" || id === "mih") {
    let postings = 0;
    if (id === "full") {
      const logKeys = Math.min(s.d, Math.log2(s.hashProbes));
      keyCount = 2 ** logKeys * p.probes;
      docs = p.selected * 2 ** (logKeys - s.d);
      postings = docs;
      indexBytes += s.n * 32; // Assumed hash entry overhead, not an allocator measurement.
      extraMs += (keyCount * Math.log2(s.hashProbes + 1) * s.queueNs) / 1e6;
      note =
        "Start with the ideal sign key, then enumerate distinct flip sets by total |query| penalty. Most full keys may be empty. Expected hits assume independent, equally likely document signs; this is not a real-data recall estimate.";
    } else if (id === "prefix") {
      const bits = Math.min(s.keyBits, s.d);
      const volume = 2 ** logBall(bits, s.radius);
      keyCount = volume * p.probes;
      docs = (p.selected * volume) / 2 ** bits;
      postings = docs;
      indexBytes += p.clusters * (2 ** bits + 1) * 8;
      note =
        "Probe a short key and nearby keys, gather their document IDs, then score complete codes. The radius here counts flips equally; query-weighted probing needs a different key schedule. Uniform key occupancy is an assumption.";
    } else {
      const tables = Math.min(s.tables, s.d);
      const short = Math.floor(s.d / tables);
      const longer = s.d % tables;
      let logMissChance = 0;
      for (let table = 0; table < tables; table++) {
        const bits = short + Number(table < longer);
        const logVolume = logBall(bits, s.radius);
        const probability = 2 ** (logVolume - bits);
        keyCount += 2 ** logVolume * p.probes;
        postings += p.selected * probability;
        logMissChance += Math.log1p(-probability);
        indexBytes += s.n * 8 + p.clusters * Math.min(p.bucket, 2 ** bits) * 32;
      }
      // expm1 keeps tiny hit probabilities that 1 - product would round to zero.
      docs = p.selected * -Math.expm1(logMissChance);
      extraMs += (postings * s.hashNs) / 1e6;
      note =
        "Split a code into disjoint substrings. Search each table, combine IDs, remove duplicates, and score complete codes. The union size assumes independent signs. Hamming-distance guarantees do not automatically apply to a query-weighted score.";
    }
    add(
      "Key lookups, including empty keys",
      keyCount,
      id === "full"
        ? "key budget × probes"
        : "sum of binomial key volumes × probes",
      "lookups",
    );
    add(
      "Posting IDs read",
      postings,
      "expected table hits before removing duplicates",
      "IDs",
    );
    // Hashing a long key costs more than hashing one machine word.
    const hashWords =
      id === "full"
        ? Math.ceil(s.d / 64)
        : id === "prefix"
          ? 1
          : Math.ceil(Math.ceil(s.d / s.tables) / 64);
    extraMs += (keyCount * (s.hashNs + hashWords * 2)) / 1e6;
    traffic = keyCount * (32 + hashWords * 8) + postings * 8 + docs * codeBytes;
    temporary = docs * 8;
  }
  add(
    "Full document scores",
    docs,
    id === "scan" ? "N × probes / clusters" : "surviving documents",
    "documents",
  );
  add(
    "Four-sign lookup terms",
    docs * groups,
    "documents scored × ceil(D / 4)",
    "terms",
  );
  add(
    "Logical bytes touched",
    traffic,
    "code reads + method-specific reads and writes; see README",
    "bytes",
  );
  const scoreMs = (docs * groups * s.scoreNs) / 1e6;
  const bitmapMs = (wordWork * wordCost) / 1e6;
  const queueMs = (nodeWork * s.nodeNs) / 1e6;
  const ms = s.fixedMs + routing + scoreMs + bitmapMs + queueMs + extraMs;
  const feasible = !["dense", "sparse"].includes(id) || b.leaves <= docs + 1e-8;
  return {
    id,
    name: names[id],
    feasible,
    ms,
    docs,
    indexBytes,
    temporary,
    traffic,
    keyCount,
    note,
    quality,
    rows,
    parts: {
      fixed: s.fixedMs,
      routing,
      scoring: scoreMs,
      bitmaps: bitmapMs,
      queue: queueMs,
      hashing: extraMs,
    },
  };
}

// This is the measured workload, without trying to guess visits from recall or a normal distribution.
export function predictMeasurement(row, rates) {
  const scoring = (row.mean_documents_scored * 64 * rates.scoreNs) / 1e6;
  const branching =
    row.method === "branch"
      ? (row.mean_nodes * Math.ceil(row.pool_size / 64) * rates.wordNs) / 1e6 +
        (row.mean_nodes * rates.nodeNs) / 1e6
      : 0;
  return rates.fixedMs + scoring + branching;
}

export function loadMeasurement(s, row) {
  const words = Math.ceil(row.pool_size / 64);
  const splits = row.mean_bitplane_words / words;
  return {
    ...s,
    n: row.pool_size,
    d: 256,
    routing: "none",
    nodes: row.mean_nodes,
    leaf: 32,
    splitFraction: splits / row.mean_nodes,
    scoredFraction: row.mean_documents_scored / row.pool_size,
  };
}

// A small, complete query example. Seeded data makes replaying a step useful.
export function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 2 ** 32;
  };
}
export function toySearch({ probability = 0, seed = 7, budget = 16 } = {}) {
  const random = randomGenerator(seed);
  const query = [0.7, 0.5, -0.4, -0.6, 0.3, -0.12, 0.08, -0.04];
  const weights = query.map(Math.abs);
  const ideal = query.reduce(
    (mask, q, bit) => mask | (Number(q >= 0) << bit),
    0,
  );
  const codes = Array.from({ length: 128 }, () => Math.floor(random() * 256));
  const score = (code) =>
    query.reduce((sum, q, bit) => sum + q * ((code >> bit) & 1 ? 1 : -1), 0);
  const order = weights
    .map((_, i) => i)
    .sort((a, b) => weights[b] - weights[a]);
  const all = codes.map((_, i) => i);
  const exact = [...all]
    .sort((a, b) => score(codes[b]) - score(codes[a]) || a - b)
    .slice(0, 10);
  const frontier = [{ rows: all, depth: 0, penalty: 0, order: 0 }];
  let next = 1;
  const found = [];
  const trace = [];
  for (let visit = 0; visit < budget && frontier.length; visit++) {
    frontier.sort((a, b) => a.penalty - b.penalty || a.order - b.order);
    const explore = frontier.length > 1 && random() < probability;
    const chosen = explore
      ? 1 + Math.floor(random() * (frontier.length - 1))
      : 0;
    const node = frontier.splice(chosen, 1)[0];
    const bit = order[node.depth];
    const leaf = node.rows.length <= 8 || node.depth === 8;
    trace.push({ ...node, bit, explore, leaf });
    if (leaf) {
      found.push(...node.rows);
      continue;
    }
    const match = [],
      opposite = [];
    for (const row of node.rows) {
      (((codes[row] >> bit) & 1) === ((ideal >> bit) & 1)
        ? match
        : opposite
      ).push(row);
    }
    if (match.length)
      frontier.push({
        rows: match,
        depth: node.depth + 1,
        penalty: node.penalty,
        order: next++,
      });
    if (opposite.length)
      frontier.push({
        rows: opposite,
        depth: node.depth + 1,
        penalty: node.penalty + weights[bit],
        order: next++,
      });
  }
  const result = found
    .sort((a, b) => score(codes[b]) - score(codes[a]) || a - b)
    .slice(0, 10);
  const recall = result.filter((row) => exact.includes(row)).length / 10;
  const flips = Array.from({ length: 256 }, (_, mask) => ({
    mask,
    penalty: weights.reduce(
      (sum, w, bit) => sum + ((mask >> bit) & 1 ? w : 0),
      0,
    ),
  })).sort((a, b) => a.penalty - b.penalty || a.mask - b.mask);
  return { query, ideal, codes, exact, result, recall, trace, flips };
}
