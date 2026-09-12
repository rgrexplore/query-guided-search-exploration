"""One fixed dataset, many index settings: scan (A), bitplanes (B) and backward walk (C).

The inputs never change: 1,000,000 MS MARCO passages stored as 256-bit codes, 200 queries,
and the exact top-100 answer for each query. Only index parameters vary: the number of
clusters, how many clusters are opened, the leaf size and node budget of B, and the key width
of C. Every setting reads the same arrays, and every method is measured on the same (clusters,
opened) pairs, so the methods are compared on identical work-to-do.

Stages:
    run      verify the inputs, derive the opened-cluster counts from the routing curve,
             measure every declared setting in its own process, then analyze.
    analyze  summarize a finished run: settings.csv, selected.csv, checks.json, summary.json.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import bitplane_index
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.analysis import load_cases
from experiments.isolated import ROOT, execute_cases
from experiments.models import mean_recall
from experiments.probe_cutoffs import routing_ranks

POOL = ROOT / "data/real-evaluation-2026-09-10/n1000000"
ROUTERS = ROOT / "data/real-scale-2026-09-10/n1000000/routers"
CLUSTER_COUNTS = (64, 256, 1024, 4096, 8192)
TARGETS = (0.8, 0.9, 0.95, 0.99)
ARRAYS = ("codes.npy", "queries.npy", "reference.npy")
DOCUMENTS, DIMENSIONS, QUERIES, TOP_K = 1_000_000, 256, 200, 100
WORD_BITS = 64
METHODS = ("scan", "branch", "keys")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_table(path, rows):
    """CSV whose columns are the union of every row's keys, in first-seen order."""
    names = []
    for row in rows:
        names.extend(key for key in row if key not in names)
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


# ----- Inputs -----

def verify_inputs():
    """Check that the arrays on disk are the declared benchmark, and record their hashes."""
    metadata = json.loads((POOL / "pool.json").read_text())
    identity = metadata["identity"]
    if (identity["documents"], identity["dimensions"], identity["queries"], identity["top_k"]) != (
            DOCUMENTS, DIMENSIONS, QUERIES, TOP_K):
        raise ValueError("the pool is not the declared 1M-document, 200-query benchmark")
    hashes = {}
    for name in ARRAYS:
        hashes[name] = file_hash(POOL / name)
        if hashes[name] != metadata["hashes"][name]:
            raise ValueError(f"benchmark array changed on disk: {name}")
    routers = {}
    for clusters in CLUSTER_COUNTS:
        folder = ROUTERS / f"ivf-{clusters}-seed42"
        info = json.loads((folder / "router.json").read_text())
        if (info["kind"], info["clusters"]) != ("ivf", clusters):
            raise ValueError(f"unexpected router at {folder}")
        current = {name: file_hash(folder / name) for name in ("assignments.npy", "centroids.npy")}
        if current["assignments.npy"] != info["assignments_sha256"]:
            raise ValueError(f"cluster assignments changed: {folder}")
        if current["centroids.npy"] != info["centroids_sha256"]:
            raise ValueError(f"cluster centroids changed: {folder}")
        routers[clusters] = dict(path=str(folder), hashes=current)
    return dict(pool=str(POOL), arrays=hashes, routers=routers, query_ids=metadata["query_ids"],
                scope="Documents, queries and reference answers are read in place and never modified.")


def routing_facts(clusters, query_rows):
    """Recall and documents opened as a function of P, from the actual router.

    Returns the per-P curve, the opened-cluster counts for each recall target, and per-query
    data (ordered cluster labels and cluster sizes) that the analysis uses to predict work.
    """
    folder = ROUTERS / f"ivf-{clusters}-seed42"
    case = dict(pool=str(POOL), router=str(folder), query_rows=query_rows)
    ranks, _, quantizer, route_queries, (labels, _) = routing_ranks(case)
    sizes = np.bincount(np.load(folder / "assignments.npy"), minlength=clusters)
    # The same router call the worker makes, asking for every cluster, best first.
    _, positions = quantizer.search(route_queries, clusters)
    opened_sizes = sizes[labels[positions]]
    opened_docs = np.cumsum(opened_sizes, axis=1)  # documents in the first P clusters
    histogram = np.bincount(ranks.ravel(), minlength=clusters + 1)
    recall_curve = np.cumsum(histogram)[1:] / ranks.size
    words = np.cumsum((opened_sizes + WORD_BITS - 1) // WORD_BITS, axis=1)
    probes = {}
    for target in TARGETS:
        probe = int(np.searchsorted(recall_curve, target) + 1)
        probes[target] = probe
    curve = [dict(P=int(p), recall=float(recall_curve[p - 1]), docs=float(opened_docs[:, p - 1].mean()))
             for p in sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, *probes.values()})
             if p <= clusters]
    return dict(clusters=clusters, probes=probes, curve=curve,
                cluster_sizes=dict(min=int(sizes.min()), median=float(np.median(sizes)), max=int(sizes.max())),
                recall_curve=recall_curve, opened_docs=opened_docs, opened_words=words)


# ----- Settings -----

def layouts_for(facts_by_clusters):
    """Every method is measured on exactly these (clusters, opened) pairs."""
    layouts = [dict(clusters=1, router=None, probes=1, target=None)]
    for clusters, facts in facts_by_clusters.items():
        for target, probe in facts["probes"].items():
            layouts.append(dict(clusters=clusters, router=str(ROUTERS / f"ivf-{clusters}-seed42"),
                                probes=probe, target=target))
    return layouts


def build_cases(layouts, query_rows, repetitions, ram_budget):
    cases = []
    for layout in layouts:
        common = dict(pool=str(POOL), documents=DOCUMENTS, dimensions=DIMENSIONS, query_rows=query_rows,
                      router=layout["router"], router_kind="none" if layout["router"] is None else "ivf",
                      clusters=layout["clusters"], probes=layout["probes"], target=layout["target"],
                      top_k=TOP_K, repetitions=repetitions, ram_budget_bytes=ram_budget)
        cases.append(dict(common, method="scan"))
        # Leaf 1,000,000 never splits: B does A's work plus its own bookkeeping.
        # Leaf 128 splits until a group is small; budget 0 means finish the search.
        for leaf, budget in ((DOCUMENTS, 0), (128, 0), (128, 64), (128, 512), (128, 4096)):
            if layout["clusters"] == 1 and leaf == 128 and budget == 0:
                # Predicted at about one second per query; still measured, with fewer repeats.
                cases.append(dict(common, method="branch", leaf_size=leaf, node_budget=budget, repetitions=1))
            else:
                cases.append(dict(common, method="branch", leaf_size=leaf, node_budget=budget))
        for bits in (4, 8, 12):
            cases.append(dict(common, method="keys", key_bits=bits, key_offset=0,
                              candidate_target=0, key_limit=0))
    for number, case in enumerate(cases):
        case["setting_id"] = f"s{number:03d}"
    return cases


# ----- Analysis -----

def predicted_counts(case, facts_by_clusters):
    """What Section 2 says the code should do for this setting, before looking at the run."""
    clusters, probes = case["clusters"], case["probes"]
    if clusters == 1:
        docs = float(DOCUMENTS)
        words = float((DOCUMENTS + WORD_BITS - 1) // WORD_BITS)
        routing_recall = 1.0
    else:
        facts = facts_by_clusters[clusters]
        docs = float(facts["opened_docs"][:, probes - 1].mean())
        words = float(facts["opened_words"][:, probes - 1].mean())
        routing_recall = float(facts["recall_curve"][probes - 1])
    out = dict(docs_in_opened_clusters=docs, mask_words_in_opened_clusters=words, routing_recall=routing_recall)
    if case["method"] == "scan":
        out.update(documents_scored=docs, recall=routing_recall)
    elif case["method"] == "branch":
        if case["leaf_size"] >= DOCUMENTS:
            out.update(documents_scored=docs, split_words=0.0, leaf_words=words, recall=routing_recall)
        elif case["node_budget"] == 0:
            out.update(documents_scored=docs, recall=routing_recall)  # exact; split count is measured
    else:
        keys = 2 ** case["key_bits"]
        out.update(keys=keys, key_lookups=keys * probes, documents_scored=docs, recall=routing_recall)
    return out


def summarize_run(folder, facts_by_clusters):
    completed, failures = load_cases(folder)
    schedule = json.loads((folder / "schedule.json").read_text())
    by_id = {item["case"]["setting_id"]: item for item in completed}
    rows = []
    for case in schedule:
        item = by_id.get(case["setting_id"])
        prediction = predicted_counts(case, facts_by_clusters)
        row = dict(setting_id=case["setting_id"], method=case["method"], clusters=case["clusters"],
                   probes=case["probes"], target=case["target"], leaf_size=case.get("leaf_size", 0),
                   node_budget=case.get("node_budget", 0), key_bits=case.get("key_bits", 0),
                   status="missing" if item is None else "complete")
        for name, value in prediction.items():
            row[f"predicted_{name}"] = value
        if item is None:
            rows.append(row)
            continue
        queries = item["queries"]
        first = [q for q in queries if q["repetition"] == 0]
        stable = ("recall", "documents_scored", "bitplane_words", "leaf_words", "key_attempts", "keys_generated", "nodes")
        expected = {q["query"]: tuple(q[k] for k in stable) for q in first}
        assert all(tuple(q[k] for k in stable) == expected[q["query"]] for q in queries), "work differed between repeats"
        result = item["result"]
        storage, memory = result["storage"], result["memory"]
        row.update(
            recall=mean_recall([q["recall"] for q in first], TOP_K),
            query_ms_p50=float(np.median([q["query_ms"] for q in queries])),
            query_ms_p90=float(np.percentile([q["query_ms"] for q in queries], 90)),
            search_ms_p50=float(np.median([q["search_ms"] for q in queries])),
            routing_ms_p50=float(np.median([q["routing_ms"] for q in queries])),
            documents_scored=float(np.mean([q["documents_scored"] for q in first])),
            split_words=float(np.mean([q["bitplane_words"] for q in first])),
            leaf_words=float(np.mean([q["leaf_words"] for q in first])),
            nodes=float(np.mean([q["nodes"] for q in first])),
            key_lookups=float(np.mean([q["key_attempts"] for q in first])),
            keys=float(np.mean([q["keys_generated"] for q in first])),
            stop_reasons=json.dumps(sorted({q["stop_reason"] for q in first})),
            codes_bytes=storage["codes_bytes"], row_ids_bytes=storage["row_ids_bytes"],
            bitplanes_bytes=storage["bitplanes_bytes"], directory_bytes=storage["key_directory_payload_bytes"],
            occupied_keys=storage["occupied_keys"], logical_index_bytes=storage["logical_bytes"],
            routing_payload_bytes=memory["routing_payload_bytes"],
            resident_bytes=memory["rss_before_queries"], lifetime_peak_bytes=memory["lifetime_peak_bytes"],
            fits_ram=memory["budget_status"] == "fits_by_lifetime_peak",
            build_ms=result["build_ms"], measurements=len(queries),
            power_unchanged=result["power_before"]["available"] and result["power_before"] == result["power_after"],
        )
        rows.append(row)
    return rows, failures


def select_fastest(rows):
    """For each recall target and method, the fastest measured setting that reaches the target."""
    selected = []
    for target in TARGETS:
        for method in METHODS:
            eligible = [r for r in rows if r["method"] == method and r["status"] == "complete"
                        and r["fits_ram"] and r["recall"] >= target]
            if eligible:
                best = min(eligible, key=lambda r: (r["query_ms_p50"], r["setting_id"]))
                selected.append(dict(recall_target=target, **best))
    return selected


def check_formulas(rows, facts_by_clusters):
    """Compare what Section 2 predicts with what the code recorded."""
    checks = []
    for r in rows:
        if r["status"] != "complete":
            continue
        entry = dict(setting_id=r["setting_id"], method=r["method"], clusters=r["clusters"], probes=r["probes"])
        exact = (r["method"] == "scan" or (r["method"] == "branch" and r["node_budget"] == 0)
                 or r["method"] == "keys")
        if exact:
            entry["documents_scored"] = dict(predicted=r["predicted_documents_scored"], measured=r["documents_scored"])
            entry["recall"] = dict(predicted=r["predicted_routing_recall"], measured=r["recall"])
        if r["method"] == "branch" and r["leaf_size"] >= DOCUMENTS:
            entry["leaf_words"] = dict(predicted=r["predicted_leaf_words"], measured=r["leaf_words"])
            entry["split_words"] = dict(predicted=0.0, measured=r["split_words"])
        if r["method"] == "keys":
            entry["keys"] = dict(predicted=r["predicted_keys"], measured=r["keys"])
            entry["key_lookups"] = dict(predicted=r["predicted_key_lookups"], measured=r["key_lookups"])
            entry["directory_bytes"] = dict(predicted=r["occupied_keys"] * 20, measured=r["directory_bytes"])
        entry["codes_bytes"] = dict(predicted=DOCUMENTS * 32, measured=r["codes_bytes"])
        entry["row_ids_bytes"] = dict(predicted=DOCUMENTS * 8, measured=r["row_ids_bytes"])
        if r["method"] == "branch":
            if r["clusters"] == 1:
                total_words = (DOCUMENTS + WORD_BITS - 1) // WORD_BITS
            else:
                folder = ROUTERS / f"ivf-{r['clusters']}-seed42"
                sizes = np.bincount(np.load(folder / "assignments.npy"), minlength=r["clusters"])
                total_words = int(((sizes + WORD_BITS - 1) // WORD_BITS).sum())
            entry["bitplanes_bytes"] = dict(predicted=DIMENSIONS * 8 * total_words, measured=r["bitplanes_bytes"])
        checks.append(entry)
    return checks


def fit_unit_costs(rows):
    """Milliseconds per counted operation, by nonnegative least squares on search time.

    The features are the operation counts that Section 2 names. The fit uses every complete
    setting of a method; the report shows how far each setting's predicted time is from its
    measured time, so a poor formula cannot hide.
    """
    from scipy.optimize import nnls
    models = {}
    features = {
        "scan": ["call", "documents_scored"],
        "branch": ["call", "documents_scored", "split_words", "leaf_words", "nodes"],
        "keys": ["call", "documents_scored", "key_lookups", "keys"],
    }
    for method, names in features.items():
        sample = [r for r in rows if r["method"] == method and r["status"] == "complete"]
        if len(sample) < len(names):
            continue
        matrix = np.array([[1.0 if n == "call" else r[n] for n in names] for r in sample])
        observed = np.array([r["search_ms_p50"] for r in sample])
        weights = 1 / observed  # relative error matters equally for fast and slow settings
        scales = np.maximum(np.linalg.norm(matrix * weights[:, None], axis=0), 1e-12)
        coefficients, _ = nnls(matrix * weights[:, None] / scales, observed * weights)
        coefficients = coefficients / scales
        predicted = matrix @ coefficients
        errors = 100 * (predicted - observed) / observed
        for r, p, e in zip(sample, predicted, errors):
            r["predicted_search_ms"] = float(p)
            r["search_ms_error_percent"] = float(e)
        models[method] = dict(features=names, ms_per_unit=dict(zip(names, coefficients.tolist())),
                              settings=len(sample), median_abs_error_percent=float(np.median(np.abs(errors))),
                              max_abs_error_percent=float(np.max(np.abs(errors))))
    return models


def analyze(output):
    config = json.loads((output / "configuration.json").read_text())
    query_rows = list(range(config["queries"]))
    with threadpool_limits(limits=1):
        facts = {c: routing_facts(c, query_rows) for c in CLUSTER_COUNTS}
    rows, failures = summarize_run(output / "runs", facts)
    models = fit_unit_costs(rows)
    checks = check_formulas(rows, facts)
    selected = select_fastest(rows)
    write_table(output / "settings.csv", rows)
    if selected:
        write_table(output / "selected.csv", selected)
    save_json(output / "checks.json", checks)
    save_json(output / "unit-costs.json", models)
    save_json(output / "routing.json", {c: dict(clusters=c, probes={str(k): v for k, v in f["probes"].items()},
                                                curve=f["curve"], cluster_sizes=f["cluster_sizes"])
                                        for c, f in facts.items()})
    save_json(output / "summary.json", dict(
        settings=len(rows), complete=sum(r["status"] == "complete" for r in rows), failures=failures,
        selected=selected, unit_costs=models,
        scope="Same documents, queries and reference for every setting. Fastest eligible setting per "
              "method and recall target on this benchmark; not a claim about other datasets."))
    print(f"{len(rows)} settings, {sum(r['status'] == 'complete' for r in rows)} complete, "
          f"{len(failures)} failed; {len(selected)} selected rows.")


def run(output, queries, repetitions, smoke):
    manifest = verify_inputs()
    output.mkdir(parents=True, exist_ok=False)
    query_rows = list(range(queries))
    config = dict(documents=DOCUMENTS, dimensions=DIMENSIONS, queries=queries, top_k=TOP_K,
                  targets=list(TARGETS), cluster_counts=list(CLUSTER_COUNTS), repetitions=repetitions,
                  schedule_seed=20260912, smoke=smoke,
                  limits=dict(ram_budget_bytes=32_000_000_000, worker_stop_bytes=34_359_738_368,
                              case_seconds=2400),
                  scope="Only index parameters vary. Every method is measured on the same (clusters, opened) pairs.")
    save_json(output / "configuration.json", config)
    save_json(output / "inputs.json", manifest)
    with threadpool_limits(limits=1):
        facts = {c: routing_facts(c, query_rows) for c in CLUSTER_COUNTS}
    layouts = layouts_for(facts)
    cases = build_cases(layouts, query_rows, repetitions, config["limits"]["ram_budget_bytes"])
    if smoke:
        cases = [c for c in cases if c["clusters"] in (1, 256) and c["target"] in (None, 0.8)]
        cases = [c for c in cases if not (c["clusters"] == 1 and c["method"] == "branch" and c["leaf_size"] == 128 and c["node_budget"] == 0)]
    save_json(output / "layouts.json", layouts)
    sources = [Path(__file__), ROOT / "experiments/worker.py", ROOT / "experiments/isolated.py",
               ROOT / "experiments/probe_cutoffs.py", Path(bitplane_index.__file__)]
    save_json(output / "source-hashes.json", {str(p): file_hash(p) for p in sources})
    runs = output / "runs"
    runs.mkdir()
    save_json(runs / "configuration.json", config)
    order = np.random.default_rng(config["schedule_seed"]).permutation(len(cases))
    print(f"Measuring {len(cases)} settings on {queries} queries.", flush=True)
    execute_cases(config, runs, [cases[int(i)] for i in order],
                  "Fixed benchmark: same documents, queries and reference for every setting; one CPU thread.")
    after = verify_inputs()
    if after["arrays"] != manifest["arrays"]:
        raise ValueError("benchmark arrays changed during the run")
    analyze(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="a handful of settings, for checking the script")
    args = parser.parse_args()
    if args.stage == "run":
        run(args.output.resolve(), args.queries, args.repetitions, args.smoke)
    else:
        analyze(args.output.resolve())


if __name__ == "__main__":
    main()
