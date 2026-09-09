// Three methods, one shared workload. The saved runs supply the branch-work estimates.
export const defaults = {
  documents: 1_000_000,
  groups: 100,
  opened: 1,
  bits: 256,
};

export function searchPool(settings) {
  const groups = Math.min(settings.groups, settings.documents);
  const opened = Math.min(settings.opened, groups);
  return {
    groups,
    opened,
    perGroup: settings.documents / groups,
    selected: (settings.documents * opened) / groups,
  };
}

export function branchExample(perGroup, evidence) {
  const rows = evidence.rows.filter(
    (row) => row.phase === "development" && row.method === "branch",
  );
  const sizes = [...new Set(rows.map((row) => row.pool_size))];
  const closest = sizes.reduce((best, n) =>
    Math.abs(Math.log(n / perGroup)) < Math.abs(Math.log(best / perGroup))
      ? n
      : best,
  );
  // Pick the fastest saved setting that reached 95% recall in that validation pool.
  // This does not predict recall in new clusters; it only supplies a concrete work example.
  return rows
    .filter((row) => row.pool_size === closest && row.mean_recall >= 0.95)
    .sort((a, b) => a.api_p50_ms - b.api_p50_ms)[0];
}

export function predictRecordedTime(row, rates) {
  const scoring = (row.mean_documents_scored * 64 * rates.scoreNs) / 1e6;
  const branching =
    row.method === "branch"
      ? (row.mean_nodes * Math.ceil(row.pool_size / 64) * rates.wordNs) / 1e6 +
        (row.mean_nodes * rates.nodeNs) / 1e6
      : 0;
  return rates.fixedMs + scoring + branching;
}

export function compare(settings, evidence) {
  const pool = searchPool(settings);
  const rates = evidence.calibration;
  const bytesPerCode = Math.ceil(settings.bits / 64) * 8;
  const termsPerScore = Math.ceil(settings.bits / 4);
  const wordsPerGroup = Math.ceil(pool.perGroup / 64);
  // Routing was not timed in the no-routing study. Keep its assumed cost visible in the math.
  const routingMs = pool.groups > 1 ? 0.05 : 0;
  const setupMs = rates.fixedMs + routingMs;
  const scoreMs = (documents) =>
    (documents * termsPerScore * rates.scoreNs) / 1e6;
  const storedBase = settings.documents * (bytesPerCode + 8);

  const scan = {
    ms: setupMs + scoreMs(pool.selected),
    documentsScored: pool.selected,
    scoreTerms: pool.selected * termsPerScore,
    bytesTouched: pool.selected * bytesPerCode,
    extraBytes: 0,
  };

  const recorded = branchExample(pool.perGroup, evidence);
  const scale = pool.perGroup / recorded.pool_size;
  const nodes = recorded.mean_nodes * scale * pool.opened;
  const recordedSplits =
    recorded.mean_bitplane_words / Math.ceil(recorded.pool_size / 64);
  const splits = recordedSplits * scale * pool.opened;
  const leaves = nodes - splits;
  const scored = Math.min(
    pool.selected,
    recorded.mean_documents_scored * scale * pool.opened,
  );
  const splitWords = splits * wordsPerGroup;
  const leafWords = leaves * wordsPerGroup;
  const branch = {
    ms:
      setupMs +
      scoreMs(scored) +
      ((splitWords + leafWords) * rates.wordNs) / 1e6 +
      (nodes * rates.nodeNs) / 1e6,
    documentsScored: scored,
    nodes,
    splits,
    leaves,
    splitWords,
    leafWords,
    bytesTouched: splitWords * 48 + leafWords * 8 + scored * bytesPerCode,
    extraBytes: pool.groups * settings.bits * wordsPerGroup * 8,
    nodeBytes: wordsPerGroup * 8,
    // A rough comparison for one visit; it doesn't mean the split actually discards documents.
    splitInDocuments:
      (wordsPerGroup * rates.wordNs + rates.nodeNs) /
      (termsPerScore * rates.scoreNs),
    recorded,
  };

  // Full keys, not short prefixes. Most of this budget can be spent on empty keys.
  const keysPerGroup = 65_536;
  const keyLookups = keysPerGroup * pool.opened;
  const expectedDocuments =
    pool.selected * 2 ** (Math.log2(keysPerGroup) - settings.bits);
  const lookupNs = 100 + Math.ceil(settings.bits / 64) * 2;
  const enumerationNs = Math.log2(keysPerGroup + 1) * 5;
  const backwards = {
    ms:
      setupMs +
      (keyLookups * (lookupNs + enumerationNs)) / 1e6 +
      scoreMs(expectedDocuments),
    documentsScored: expectedDocuments,
    keyLookups,
    keysPerGroup,
    bytesTouched:
      keyLookups * (32 + bytesPerCode) + expectedDocuments * (8 + bytesPerCode),
    extraBytes: settings.documents * 32,
    // This is an occupancy assumption, not a measured outcome or a recall guarantee.
    enoughExpected: expectedDocuments >= 100,
  };
  return {
    pool,
    scan,
    branch,
    backwards,
    wordsPerGroup,
    bytesPerCode,
    termsPerScore,
    storedBase,
    routingMs,
    setupMs,
  };
}

// Eight-bit example only. It is small enough to check every document for the exact answer.
export function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 2 ** 32;
  };
}
export function toySearch({
  probability = 0,
  seed = 7,
  budget = 16,
  documentCount = 128,
  leafSize = 8,
} = {}) {
  const random = randomGenerator(seed);
  const query = [0.7, 0.5, -0.4, -0.6, 0.3, -0.12, 0.08, -0.04];
  const weights = query.map(Math.abs);
  const ideal = query.reduce(
    (mask, q, bit) => mask | (Number(q >= 0) << bit),
    0,
  );
  const codes = Array.from({ length: documentCount }, () =>
    Math.floor(random() * 256),
  );
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
    const leaf = node.rows.length <= leafSize || node.depth === 8;
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
