import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  defaults,
  evaluate,
  routing,
  weightedKeys,
  collectKeys,
  recallModel,
  normalCdf,
  normalQuantile,
  sweep,
  bestAt,
  pareto,
} from "./optimizer-v2.mjs";
const evidence = JSON.parse(
  fs.readFileSync(new URL("./measurements.json", import.meta.url)),
);
const near = (a, b, tolerance = 1e-8) =>
  assert.ok(
    Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)),
    `${a} versus ${b}`,
  );

test("global C has the same target, work and memory when cluster controls change", () => {
  const a = evaluate({ ...defaults, keyMode: "short", candidates: 10000, k: 12 }, "C", evidence);
  const b = evaluate(
    { ...defaults, candidates: 10000, k: 24, probes: 65536, router: "km" },
    "C",
    evidence,
  );
  near(a.ms, b.ms);
  near(a.docs, b.docs);
  assert.equal(a.recall, b.recall);
  assert.deepEqual(a.memory, b.memory);
  const changed = evaluate({ ...defaults, keyMode: "short", candidates: 1e8 }, "C", evidence);
  assert.ok(changed.docs > a.docs);
  assert.ok(changed.ms > a.ms);
});

test("the RAM gate counts document IDs, scratch and reserve, not only code/hash bytes", () => {
  const blocked = evaluate({ ...defaults, keyMode: "short", ram: 35e9 }, "C", evidence);
  assert.equal(blocked.memory.components.codes, 32e9);
  assert.equal(blocked.memory.components.documentIDs, 8e9);
  assert.ok(blocked.memory.scratch > 0);
  assert.equal(blocked.memory.feasible, false);
  const healthy = evaluate({ ...defaults, keyMode: "short", ram: 1e12 }, "C", evidence);
  assert.equal(healthy.memory.feasible, true);
  for (const method of ["A", "B", "C"]) {
    const m = evaluate(defaults, method, evidence).memory;
    near(
      m.total,
      Object.values(m.components).reduce((a, b) => a + b, 0) +
        m.scratch +
        m.reserve,
    );
  }
});

test("concurrent queries multiply scratch but not the stored index", () => {
  for (const method of ["A", "B", "C"]) {
    const a = evaluate(defaults, method, evidence).memory,
      b = evaluate({ ...defaults, queries: 4 }, method, evidence).memory;
    near(a.index, b.index);
    near(b.scratch, 4 * a.scratch);
    assert.ok(b.total > a.total);
  }
});

test("1M branch work reproduces saved validation counters, including leaf words", () => {
  const row = evidence.rows.find(
    (r) =>
      r.phase === "development" &&
      r.pool_size === 1e6 &&
      r.method === "branch" &&
      r.node_budget === 32768,
  );
  const b = evaluate(
    { ...defaults, N: 1e6, k: 0, probes: 1, budget: 32768, leaf: 32 },
    "B",
    evidence,
  );
  near(b.detail.splits * 15625, row.mean_bitplane_words);
  near(b.detail.words, row.mean_nodes * 15625);
  near(b.docs, row.mean_documents_scored);
  assert.ok(b.detail.leaves > 0);
  assert.ok(b.detail.words > row.mean_bitplane_words);
  // Finite-budget traversal is not necessarily half splits and half leaves.
  assert.ok(Math.abs(b.detail.splits / b.detail.nodes - 0.5) > 0.1);
});

test("opening all clusters always selects all documents and gives exact routing recall", () => {
  for (const router of ["bits", "km"]) {
    const r = routing(
      { ...defaults, k: 12, probes: 4096, router },
      evidence.calibration,
    );
    assert.equal(r.docs, defaults.N);
    assert.equal(r.recall, 1);
    assert.equal(r.ns, 0);
  }
  const partial = routing(
    { ...defaults, k: 12, probes: 1 },
    evidence.calibration,
  );
  assert.ok(partial.docs < defaults.N && partial.recall < 1);
});

test("weighted key counts agree with exhaustive enumeration of all eight-bit flip sets", () => {
  const { weights, levels, p } = weightedKeys(8);
  for (const level of levels) {
    let keys = 0,
      mass = 0;
    for (let mask = 0; mask < 256; mask++) {
      let cost = 0,
        flips = 0;
      for (let i = 0; i < 8; i++)
        if ((mask >> i) & 1) {
          cost += weights[i];
          flips++;
        }
      if (cost <= level.penalty) {
        keys++;
        mass += p ** flips * (1 - p) ** (8 - flips);
      }
    }
    near(level.keys, keys);
    near(level.cdf, mass);
  }
  near(levels.at(-1).keys, 256);
  near(levels.at(-1).cdf, 1);
  assert.ok(new Set(weights).size > 1);
});

test("the candidate threshold differs from the first-hit threshold and includes final ties", () => {
  const result = collectKeys(1e6, 24, 1e5);
  assert.ok(result.found >= 1e5);
  assert.ok(result.penalty > result.firstPenalty);
  const previous = weightedKeys(24).levels[result.penalty - 1];
  assert.ok(previous.cdf * 1e6 < 1e5);
  assert.equal(collectKeys(1000, 8, 1000).keys, 256);
});

test("normal recall handles independence and averages beyond the rank-100 boundary", () => {
  near(recallModel(0, 1e6, 0.02), 0.02);
  near(recallModel(1, 1000, 0.05), 0.5);
  near(recallModel(0.5, 1e6, 1), 1);
  near(recallModel(0.5, 1e6, 0), 0);
  const boundary = normalCdf(
    (0.5 * normalQuantile(1 - 100 / 1e6) - normalQuantile(0.99)) /
      Math.sqrt(0.75),
  );
  assert.ok(recallModel(0.5, 1e6, 0.01) > boundary);
});

test("sweep retains rejected rows, has unique C settings, and never selects an over-RAM point", () => {
  const sw = sweep(defaults, evidence);
  assert.equal(sw.records.length, sw.tested);
  assert.equal(sw.all.length + sw.rejected, sw.tested);
  const c = sw.records.filter((p) => p.method === "C");
  assert.equal(
    new Set(c.map((p) => `${p.settings.keyMode}:${p.settings.keyBits}:${p.settings.candidates}`))
      .size,
    c.length,
  );
  for (const target of [0.8, 0.9, 0.95, 0.99])
    for (const method of ["A", "B", "C"]) {
      const best = bestAt(sw.all, target, method);
      if (best) {
        assert.ok(best.memory.feasible && best.recall >= target);
        assert.equal(
          best.ms,
          Math.min(
            ...sw.all
              .filter((p) => p.method === method && p.recall >= target)
              .map((p) => p.ms),
          ),
        );
      }
    }
  const none = sweep({ ...defaults, ram: 32e9 }, evidence);
  assert.equal(none.all.length, 0);
  assert.equal(bestAt(none.all, 0.8), null);
});

test("Pareto selection rejects dominated and duplicate points", () => {
  const p = pareto([
    { recall: 0.9, ms: 2 },
    { recall: 0.8, ms: 3 },
    { recall: 0.9, ms: 2 },
    { recall: 1, ms: 4 },
    { recall: 0.5, ms: 1 },
  ]);
  assert.deepEqual(p, [
    { recall: 0.5, ms: 1 },
    { recall: 0.9, ms: 2 },
    { recall: 1, ms: 4 },
  ]);
});

test('full-key exact recall and short-key unknown recall are different contracts',()=>{
 const exact=evaluate({...defaults,keyMode:'full'},'C',evidence);
 assert.equal(exact.recall,1);assert.equal(exact.detail.h,256);
 const short=evaluate({...defaults,keyMode:'short',keyBits:24,candidates:1000},'C',evidence);
 assert.equal(short.recall,null);assert.equal(short.detail.h,24);
 assert.equal(bestAt([short],.8),null);assert.deepEqual(pareto([short]),[]);
 const exhaustive=evaluate({...defaults,keyMode:'short',candidates:defaults.N},'C',evidence);
 assert.equal(exhaustive.recall,1);
});

test('complete weighted key order, including boundary ties, agrees with exact binary top 100',()=>{
 const q=[7,5,-4,-6,3,-1,1,-2],ideal=q.reduce((m,v,i)=>m|((v>=0?1:0)<<i),0);
 const docs=Array.from({length:256},(_,row)=>({row,score:q.reduce((sum,v,i)=>sum+v*((row>>i&1)?1:-1),0)}));
 const ordered=Array.from({length:256},(_,mask)=>({key:ideal^mask,penalty:q.reduce((sum,v,i)=>sum+Math.abs(v)*(mask>>i&1),0)})).sort((a,b)=>a.penalty-b.penalty);
 const chosen=[];let boundary=Infinity;
 for(const key of ordered){if(chosen.length>=100&&key.penalty>boundary)break;chosen.push(docs[key.key]);if(chosen.length===100)boundary=key.penalty;}
 const rank=(a,b)=>b.score-a.score||a.row-b.row;
 assert.deepEqual(chosen.sort(rank).slice(0,100),docs.sort(rank).slice(0,100));
});
