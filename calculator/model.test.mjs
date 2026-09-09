import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import {
  defaults,
  searchPool,
  compare,
  predictRecordedTime,
  toySearch,
} from "./model.mjs";
const evidence = JSON.parse(
  fs.readFileSync(new URL("./measurements.json", import.meta.url)),
);
const near = (actual, expected) =>
  assert.ok(
    Math.abs(actual - expected) < 1e-8 * Math.max(1, Math.abs(expected)),
  );

test("measurements still match their source CSV", () => {
  const source = fs.readFileSync(
    new URL(evidence.source, new URL("./", import.meta.url)),
  );
  assert.equal(
    crypto.createHash("sha256").update(source).digest("hex"),
    evidence.source_sha256,
  );
});

test("one million documents in 100 groups gives 10,000 documents per group", () => {
  assert.deepEqual(searchPool(defaults), {
    groups: 100,
    opened: 1,
    perGroup: 10000,
    selected: 10000,
  });
  assert.equal(searchPool({ ...defaults, opened: 3 }).selected, 30000);
  assert.equal(searchPool({ ...defaults, groups: 1, opened: 3 }).selected, 1e6);
});

test("a known branch measurement includes both split and leaf word visits", () => {
  const row = evidence.rows.find(
    (r) =>
      r.phase === "evaluation" &&
      r.pool_size === 1e6 &&
      r.node_budget === 32768 &&
      r.method === "branch",
  );
  const words = 15625;
  const leafWords = row.mean_nodes * words - row.mean_bitplane_words;
  near(row.mean_bitplane_words, 352466875);
  near(leafWords, 159533125);
  near(row.mean_bitplane_words + leafWords, 512e6);
  const rates = evidence.calibration;
  near(
    predictRecordedTime(row, rates),
    rates.fixedMs +
      (row.mean_documents_scored * 64 * rates.scoreNs) / 1e6 +
      (512e6 * rates.wordNs) / 1e6 +
      (row.mean_nodes * rates.nodeNs) / 1e6,
  );
});

test("the data turned into bitplanes adds 320 GB at ten billion 256-bit documents", () => {
  const result = compare({ ...defaults, documents: 1e10, groups: 1 }, evidence);
  near(result.branch.extraBytes, 320e9);
  const odd = compare({ ...defaults, documents: 1000, groups: 1 }, evidence);
  assert.equal(odd.wordsPerGroup, 16);
  assert.equal(odd.branch.extraBytes, 256 * 16 * 8);
});

test("full-key attempts include empty lookups and do not claim enough documents", () => {
  const result = compare({ ...defaults, groups: 1 }, evidence);
  assert.equal(result.backwards.keyLookups, 65536);
  near(result.backwards.documentsScored / ((1e6 * 65536) / 2 ** 256), 1);
  assert.equal(result.backwards.enoughExpected, false);
  assert.ok(result.backwards.ms > 1);
});

test("all three methods have finite costs across the available page controls", () => {
  for (const documents of [1000, 1e6, 1e10])
    for (const bits of [128, 256, 512, 1024])
      for (const groups of [1, 100, 1e6]) {
        const result = compare(
          { ...defaults, documents, bits, groups, opened: 10 },
          evidence,
        );
        for (const method of ["scan", "branch", "backwards"]) {
          assert.ok(
            Number.isFinite(result[method].ms) && result[method].ms > 0,
          );
          assert.ok(result[method].documentsScored <= result.pool.selected);
        }
      }
});

test("the visible 16-document branch example agrees with an exhaustive score sort", () => {
  const example = toySearch({ documentCount: 16, leafSize: 2, budget: 64 });
  assert.equal(example.codes.length, 16);
  assert.equal(example.recall, 1);
  assert.deepEqual(example.result, example.exact);
  assert.deepEqual(
    example.trace[0].rows,
    Array.from({ length: 16 }, (_, i) => i),
  );
  assert.ok(example.trace[1].rows.length < 16);
  for (let i = 1; i < example.flips.length; i++)
    assert.ok(example.flips[i].penalty >= example.flips[i - 1].penalty);
  assert.equal(new Set(example.flips.map((f) => f.mask)).size, 256);
});
