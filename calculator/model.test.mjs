import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import {
  defaults,
  pool,
  calculate,
  logBall,
  binomialCdf,
  occupiedWords,
  loadMeasurement,
  predictMeasurement,
  toySearch,
  names,
} from "./model.mjs";
const evidence = JSON.parse(
  fs.readFileSync(new URL("./measurements.json", import.meta.url)),
);
const initial = defaults(evidence.calibration);
const near = (actual, expected, tolerance = 1e-8) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`,
  );

test("exported measurements match the original CSV bytes", () => {
  const bytes = fs.readFileSync(
    new URL(evidence.source, new URL("./", import.meta.url)),
  );
  assert.equal(
    crypto.createHash("sha256").update(bytes).digest("hex"),
    evidence.source_sha256,
  );
});

test("routing bits mean 2^r buckets, while no routing visits the whole pool", () => {
  assert.deepEqual(
    pool({ ...initial, n: 65536, routing: "sign", routeBits: 4, probes: 2 }),
    { clusters: 16, probes: 2, bucket: 4096, selected: 8192 },
  );
  assert.equal(pool({ ...initial, probes: 500 }).selected, initial.n);
  assert.equal(
    pool({ ...initial, routing: "ivf", clusters: 8, probes: 99 }).selected,
    initial.n,
  );
});

test("1M measured branch counts include leaf bitmap work, not only split counters", () => {
  const row = evidence.rows.find(
    (r) =>
      r.phase === "evaluation" &&
      r.pool_size === 1e6 &&
      r.method === "branch" &&
      r.node_budget === 32768,
  );
  const s = loadMeasurement(initial, row),
    result = calculate(s, "dense");
  const splits = result.rows.find(
    (r) => r.label === "Split bitmap words",
  ).value;
  const leaves = result.rows.find((r) => r.label === "Leaf bitmap words").value;
  near(splits, 352466875);
  near(splits + leaves, 512e6);
  near(result.docs, 163766.425);
  near(result.ms, predictMeasurement(row, s));
  assert.ok(result.traffic > 16e9); // Repeated masks, not just the 32 MB document codes.
  assert.ok(result.ms > calculate(s, "scan").ms);
});

test("small Hamming volumes agree with exhaustive bitstring enumeration", () => {
  for (let bits = 1; bits <= 10; bits++)
    for (let radius = 0; radius <= bits; radius++) {
      const matches = Array.from(
        { length: 2 ** bits },
        (_, v) => v.toString(2).replaceAll("0", "").length,
      ).filter((n) => n <= radius).length;
      near(2 ** logBall(bits, radius), matches);
      near(binomialCdf(bits, radius, 0.5), matches / 2 ** bits);
    }
  near(binomialCdf(4, 1, 0.1), 0.9 ** 4 + 4 * 0.1 * 0.9 ** 3);
  assert.ok(Number.isFinite(logBall(2048, 1024)));
  assert.ok(binomialCdf(2048, 1024, 0.5) > 0.5);
});

test("sparse occupancy includes the partial word and respects empty/full populations", () => {
  near(occupiedWords(65, 65), 2);
  near(occupiedWords(1, 1), 1);
  near(occupiedWords(1000, 0), 0);
  const scattered = occupiedWords(1e6, 256);
  assert.ok(scattered > 250 && scattered < 256);
});

test("a one-table hash union preserves very small expected hit counts", () => {
  const s = { ...initial, tables: 1, radius: 0, hashProbes: 1 };
  const full = calculate(s, "full"),
    union = calculate(s, "mih");
  assert.ok(full.docs > 0);
  assert.ok(union.docs > 0);
  near(union.docs / full.docs, 1);
});

test("prefilter pays for counter carries, not one instruction for 64 weighted scores", () => {
  const result = calculate(initial, "prefilter");
  const planes = result.rows.find((r) => r.label === "Input plane words").value;
  const operations = result.rows.find(
    (r) => r.label === "Counter / threshold operations",
  ).value;
  assert.ok(operations > planes * 100);
  assert.ok(
    calculate({ ...initial, weightBits: 1 }, "prefilter").ms < result.ms,
  );
});

test("all methods return finite nonnegative costs across current control ranges", () => {
  for (const n of [1000, 1e6, 1e10])
    for (const d of [64, 256, 2048])
      for (const routing of ["none", "ivf", "sign"]) {
        const s = {
          ...initial,
          n,
          d,
          routing,
          clusters: 1e6,
          routeBits: 20,
          probes: 100,
        };
        for (const id of Object.keys(names)) {
          const result = calculate(s, id);
          for (const key of [
            "ms",
            "docs",
            "traffic",
            "indexBytes",
            "temporary",
          ])
            assert.ok(
              Number.isFinite(result[key]) && result[key] >= 0,
              `${id}: ${key}`,
            );
          assert.ok(result.docs <= pool(s).selected + 1e-6);
        }
      }
});

test("seeded toy search reaches exact top 10 when its entire frontier is visited", () => {
  for (const seed of [7, 29, 100]) {
    const full = toySearch({ seed, budget: 256 });
    assert.equal(full.recall, 1);
    assert.deepEqual(full.result, full.exact);
    const low = toySearch({ seed, budget: 1 });
    assert.equal(low.recall, 0);
  }
  assert.deepEqual(toySearch(), toySearch());
  const greedy = toySearch({ budget: 8, probability: 0 });
  const random = toySearch({ budget: 8, probability: 1 });
  assert.equal(greedy.trace.length, random.trace.length);
  assert.equal(
    greedy.trace.some((node) => node.explore),
    false,
  );
  assert.ok(random.trace.some((node) => node.explore));
});

test("an entered workload cannot score fewer than one document per nonempty leaf", () => {
  assert.equal(calculate(initial, "dense").feasible, true);
  assert.equal(calculate({ ...initial, n: 1000 }, "dense").feasible, false);
  const small = evidence.rows.find(
    (r) =>
      r.phase === "evaluation" && r.pool_size === 1000 && r.method === "branch",
  );
  assert.equal(
    calculate(loadMeasurement(initial, small), "dense").feasible,
    true,
  );
});
