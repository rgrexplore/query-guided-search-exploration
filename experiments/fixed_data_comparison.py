"""Compare index settings while keeping document, query and reference arrays fixed.

The sweep and repeated measurements use the same 200 cached queries. Results
describe this benchmark and the declared settings, not unseen-query performance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import bitplane_index
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.analysis import load_cases, write_csv
from experiments.isolated import ROOT, execute_cases
from experiments.models import mean_recall
from experiments.probe_cutoffs import routing_ranks, probe_cutoff, cover_boundary_ties


POOL = ROOT / "data/real-evaluation-2026-09-10/n1000000"
ROUTER_SOURCE = ROOT / "data/real-scale-2026-09-10/n1000000"
TARGETS = [.8, .9, .95, .99]
ARRAY_NAMES = ("codes.npy", "queries.npy", "reference.npy", "documents.npy")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_arrays(pool):
    metadata = json.loads((pool / "pool.json").read_text())
    hashes = {}
    for name in ARRAY_NAMES:
        hashes[name] = file_hash(pool / name)
        if hashes[name] != metadata["hashes"][name]:
            raise ValueError(f"unchanged benchmark input check failed: {name}")
    return hashes


def verify_inputs(pool, router_source):
    hashes = verify_arrays(pool)
    metadata = json.loads((pool / "pool.json").read_text())
    identity = metadata["identity"]
    if (identity["documents"], identity["dimensions"], identity["queries"], identity["top_k"]) != (1_000_000, 256, 200, 100):
        raise ValueError("this comparison requires the declared 1M-document, 200-query benchmark")
    router_metadata = json.loads((router_source / "pool.json").read_text())
    for name in ("codes.npy", "documents.npy"):
        if file_hash(router_source / name) != hashes[name] or router_metadata["hashes"][name] != hashes[name]:
            raise ValueError(f"router source uses different documents: {name}")
    router_hashes = {}
    for clusters in (256, 4096, 8192):
        folder = router_source / "routers" / f"ivf-{clusters}-seed42"
        info = json.loads((folder / "router.json").read_text())
        if (info["kind"], info["clusters"], info["seed"]) != ("ivf", clusters, 42):
            raise ValueError(f"different cached routing setup: {folder}")
        current = {name: file_hash(folder / name) for name in
                   ("assignments.npy", "centroids.npy", "labels.npy", "router.json")}
        if current["assignments.npy"] != info["assignments_sha256"] or current["centroids.npy"] != info["centroids_sha256"]:
            raise ValueError(f"cached routing arrays changed: {folder}")
        assignments = np.load(folder / "assignments.npy", mmap_mode="r")
        labels = np.load(folder / "labels.npy")
        np.testing.assert_array_equal(labels, np.arange(clusters))
        np.testing.assert_array_equal(np.bincount(assignments, minlength=clusters), info["cluster_sizes"])
        if len(assignments) != identity["documents"]:
            raise ValueError("cached assignments do not cover the document collection")
        router_hashes[str(folder)] = current
    return dict(pool=str(pool), array_hashes=hashes, query_ids=metadata["query_ids"],
                query_rows=list(range(200)), router_source=str(router_source), router_hashes=router_hashes,
                pool_metadata_sha256=file_hash(pool / "pool.json"),
                scope="The existing arrays are read in place. No query, document, reference or dimension is changed.")


def derive_probes(pool, router, clusters, targets):
    base = dict(pool=str(pool), router=str(router), router_kind="ivf", query_rows=list(range(200)))
    ranks, scores, *_ = routing_ranks(base)
    cutoffs, counts = [], {clusters}
    for target in targets:
        proposed = probe_cutoff(ranks, target)
        probes = cover_boundary_ties(scores, proposed)
        counts.add(probes)
        cutoffs.append(dict(target=target, rank_cutoff=proposed, probes=probes,
                            routing_recall=float(np.count_nonzero(ranks <= probes) / ranks.size)))
    histogram = np.bincount(ranks.ravel(), minlength=clusters + 1)
    cdf = (np.cumsum(histogram)[1:] / ranks.size).tolist()
    return sorted(counts), dict(clusters=clusters, cutoffs=cutoffs, rank_cdf=cdf,
                                query_rows=list(range(200)), scope="Same benchmark queries and reference IDs for every method."), ranks


def build_cases(pool, layouts):
    cases = []
    for layout in layouts:
        for probes in layout["probes"]:
            common = dict(pool=str(pool), documents=1_000_000, dimensions=256, query_rows=list(range(200)),
                          router=layout["path"], router_kind=layout["kind"], clusters=layout["clusters"],
                          probes=probes, top_k=100, repetitions=2, ram_budget_bytes=32_000_000_000)
            cases.append(dict(common, method="scan"))
            for leaf, budget in ((128, 256), (128, 1024), (512, 0), (1_000_000, 0)):
                cases.append(dict(common, method="branch", leaf_size=leaf, node_budget=budget))
            for bits in (1, 4, 8):
                cases.append(dict(common, method="keys", key_bits=bits, key_offset=0, candidate_target=0, key_limit=0))
    for number, case in enumerate(cases):
        case["setting_id"] = f"setting-{number:03d}"
    return cases


def choose_settings(rows, targets):
    selected = []
    for target in targets:
        for method in ("scan", "branch", "keys"):
            eligible = [row for row in rows if row["method"] == method and row["qualified"]
                        and row["recall"] is not None and row["recall"] >= target]
            if eligible:
                chosen = min(eligible, key=lambda row: (row["p50_ms"], row["setting_id"]))
                selected.append(dict(target=target, **chosen))
    return selected


def repeat_cases(originals, selected, blocks=3):
    chosen = {row["setting_id"] for row in selected}
    return [dict(case, block=block) for block in range(blocks)
            for case in originals if case["setting_id"] in chosen]


def extended_cases(pool, layouts):
    # Reuse the same inputs and routing choices already supplied to scan.
    templates = [case for case in build_cases(pool, layouts) if case["method"] == "scan"]
    cases = []
    for common in templates:
        for leaf in (32, 128):
            cases.append(dict(common, method="branch", leaf_size=leaf, node_budget=4096))
        for bits in (16, 20, 24):
            for limit in (4096, 65536):
                cases.append(dict(common, method="keys", key_bits=bits, key_offset=0,
                                  key_limit=limit, candidate_target=0))
    for number, case in enumerate(cases):
        case["setting_id"] = f"extra-{number:03d}"
    return cases


def key_bound_diagnostic(pool):
    queries = np.load(pool / "queries.npy")
    reference = np.load(pool / "reference.npy")
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    positions = np.arange(queries.shape[1])
    last_rows = np.asarray(codes[reference[:, -1]])
    signs = (last_rows[:, positions // 64] >> (positions % 64).astype(np.uint64)) & np.uint64(1)
    penalties = np.sum((signs != (queries >= 0)) * np.abs(queries).astype(np.float64), axis=1)
    records = []
    for width in (1, 4, 8, 12, 16, 20, 24):
        maximum = np.abs(queries[:, :width]).astype(np.float64).sum(axis=1)
        records.append(dict(key_bits=width, queries_where_all_keys_remain_possible=int(np.sum(maximum < penalties)),
                            queries=len(queries), maximum_key_penalties=maximum.tolist()))
    return dict(widths=records, global_kth_penalties=penalties.tolist(),
                scope="If every key penalty is below the independent global K-th penalty, this bound cannot skip any key. The opposite inequality only permits pruning; it does not establish recall or speed.")


def summarize(folder, expected_blocks):
    completed, failures = load_cases(folder)
    schedule = json.loads((folder / "schedule.json").read_text())
    cases = {case["setting_id"]: case for case in schedule}
    by_setting = defaultdict(list)
    for item in completed:
        by_setting[item["case"]["setting_id"]].append(item)
    rows, details = [], []
    for identifier, case in cases.items():
        items = by_setting[identifier]
        observations = [row for item in items for row in item["queries"]]
        first = [row for item in items[:1] for row in item["queries"] if row["repetition"] == 0]
        stable = ("recall", "documents_scored", "bitplane_words", "leaf_words", "key_attempts")
        expected = {row["query"]: tuple(row[name] for name in stable) for row in first}
        assert all(tuple(row[name] for name in stable) == expected[row["query"]] for row in observations)
        complete = len(items) == expected_blocks and all(len(item["queries"]) == 400 for item in items)
        memory_ok = complete and all(item["result"]["memory"]["budget_status"] == "fits_by_lifetime_peak" for item in items)
        power_ok = complete and all(item["result"]["power_before"]["available"]
                                    and item["result"]["power_before"] == item["result"]["power_after"] for item in items)
        process_medians = [float(np.median([row["query_ms"] for row in item["queries"]])) for item in items]
        record = dict(setting_id=identifier, method=case["method"], router=case["router_kind"],
                      clusters=case["clusters"], probes=case["probes"], leaf_size=case.get("leaf_size", 0),
                      node_budget=case.get("node_budget", 0), key_bits=case.get("key_bits", 0),
                      key_limit=case.get("key_limit", 0),
                      recall=mean_recall([row["recall"] for row in first], 100) if first else None,
                      p50_ms=float(np.median([row["query_ms"] for row in observations])) if observations else None,
                      process_p50_min_ms=min(process_medians) if items else None,
                      process_p50_max_ms=max(process_medians) if items else None,
                      mean_scored=float(np.mean([row["documents_scored"] for row in first])) if first else None,
                      mean_split_words=float(np.mean([row["bitplane_words"] for row in first])) if first else None,
                      mean_leaf_words=float(np.mean([row["leaf_words"] for row in first])) if first else None,
                      mean_key_lookups=float(np.mean([row["key_attempts"] for row in first])) if first else None,
                      logical_index_bytes=max((item["result"]["storage"]["logical_bytes"] for item in items), default=None),
                      lifetime_peak_bytes=max((item["result"]["memory"]["lifetime_peak_bytes"] for item in items), default=None),
                      complete_processes=len(items), qualified=bool(memory_ok and power_ok))
        rows.append(record)
        details.append(dict(**record, case=case, process_p50_ms=process_medians,
                            query_count=len(first), timing_observations=len(observations)))
    write_csv(folder / "settings.csv", rows)
    save_json(folder / "summary.json", dict(settings=details, failures=failures,
        scope="Descriptive results on the same fixed 200-query benchmark; repeated timings are not additional query examples."))
    return rows, details, failures


def record_source(folder):
    sources = [Path(__file__), ROOT / "experiments/worker.py", ROOT / "experiments/isolated.py",
               ROOT / "experiments/probe_cutoffs.py", ROOT / "experiments/models.py", Path(bitplane_index.__file__)]
    save_json(folder / "source-hashes.json", {str(path): file_hash(path) for path in sources})


def sweep(output):
    pool, source = POOL.resolve(), ROUTER_SOURCE.resolve()
    before = verify_inputs(pool, source)
    config = dict(documents=1_000_000, dimensions=256, queries=200, top_k=100,
                  recall_targets=TARGETS, repetitions=2, repeat_processes=3, schedule_seed=20260915,
                  limits=dict(ram_budget_bytes=32_000_000_000, worker_stop_bytes=34_359_738_368, case_seconds=180),
                  scope="Only index parameters vary. Both stages read the same cached document, query and reference arrays.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "configuration.json", config)
    save_json(output / "inputs-manifest.json", before)
    inputs = output / "inputs"
    inputs.mkdir()
    layouts = [dict(path=None, kind="none", clusters=1, probes=[1])]
    with threadpool_limits(limits=1):
        for clusters in (256, 4096, 8192):
            router = source / "routers" / f"ivf-{clusters}-seed42"
            probes, evidence, ranks = derive_probes(pool, router, clusters, TARGETS)
            layouts.append(dict(path=str(router), kind="ivf", clusters=clusters, probes=probes))
            save_json(inputs / f"ivf-{clusters}-cutoffs.json", evidence)
            np.save(inputs / f"ivf-{clusters}-reference-ranks.npy", ranks)
    save_json(output / "layouts.json", layouts)
    cases = build_cases(pool, layouts)
    folder = output / "sweep"
    folder.mkdir()
    save_json(folder / "configuration.json", config)
    record_source(folder)
    order = np.random.default_rng(config["schedule_seed"]).permutation(len(cases))
    execute_cases(config, folder, [cases[int(index)] for index in order],
                  "Index-parameter sweep: every setting reads all 200 unchanged benchmark queries.")
    after = verify_inputs(pool, source)
    save_json(folder / "inputs-after.json", after)
    if after != before:
        raise ValueError("input identity changed during the sweep")
    rows, _, _ = summarize(folder, 1)
    selected = choose_settings(rows, TARGETS)
    save_json(output / "selected-settings.json", selected)
    if selected:
        write_csv(output / "selected-settings.csv", selected)
    print(f"Completed {len(cases)} settings; selected {len(selected)} method/recall-target rows.")


def extend(output):
    config = json.loads((output / "configuration.json").read_text())
    manifest = json.loads((output / "inputs-manifest.json").read_text())
    pool, source = Path(manifest["pool"]), Path(manifest["router_source"])
    if verify_inputs(pool, source) != manifest:
        raise ValueError("benchmark inputs changed before the extension")
    previous = json.loads((output / "sweep" / "summary.json").read_text())
    folder = output / "extra"
    folder.mkdir(exist_ok=False)
    (output / "initial-selected-settings.json").write_bytes((output / "selected-settings.json").read_bytes())
    save_json(folder / "configuration.json", dict(config, extension=dict(
        key_bits=[16, 20, 24], key_limits=[4096, 65536], leaf_sizes=[32, 128], node_budget=4096)))
    save_json(folder / "key-bound-diagnostic.json", key_bound_diagnostic(pool))
    record_source(folder)
    layouts = json.loads((output / "layouts.json").read_text())
    cases = extended_cases(pool, layouts)
    order = np.random.default_rng(config["schedule_seed"] + 2).permutation(len(cases))
    execute_cases(config, folder, [cases[int(index)] for index in order],
                  "Longer keys and larger branch budgets; identical documents, queries, references and routing options.")
    after = verify_inputs(pool, source)
    save_json(folder / "inputs-after.json", after)
    if after != manifest:
        raise ValueError("benchmark inputs changed during the extension")
    rows, _, _ = summarize(folder, 1)
    # The first run may predate the displayed key-limit column; its case record
    # still contains the exact value. Preserve that value when combining rows.
    initial_rows = [{name: row.get(name, row["case"].get(name, 0)) for name in rows[0]}
                    for row in previous["settings"]]
    selected = choose_settings(initial_rows + rows, config["recall_targets"])
    save_json(output / "selected-settings.json", selected)
    if selected:
        write_csv(output / "selected-settings.csv", selected)
    print(f"Added {len(cases)} settings; selection now includes the initial and extended parameter sets.")


def repeat(output):
    config = json.loads((output / "configuration.json").read_text())
    manifest = json.loads((output / "inputs-manifest.json").read_text())
    pool, source = Path(manifest["pool"]), Path(manifest["router_source"])
    before = verify_inputs(pool, source)
    if before != manifest:
        raise ValueError("benchmark inputs changed since the sweep")
    originals = json.loads((output / "sweep" / "schedule.json").read_text())
    extra = output / "extra" / "schedule.json"
    if extra.exists():
        originals.extend(json.loads(extra.read_text()))
    assert len({case["setting_id"] for case in originals}) == len(originals)
    selected = json.loads((output / "selected-settings.json").read_text())
    cases = repeat_cases(originals, selected, config["repeat_processes"])
    folder = output / "repeats"
    folder.mkdir(exist_ok=False)
    save_json(folder / "configuration.json", config)
    save_json(folder / "selected-settings.json", selected)
    record_source(folder)
    order = np.random.default_rng(config["schedule_seed"] + 1).permutation(len(cases))
    execute_cases(config, folder, [cases[int(index)] for index in order],
                  "Repeated selected settings on the same 200 benchmark queries; inputs and search parameters unchanged.")
    after = verify_inputs(pool, source)
    save_json(folder / "inputs-after.json", after)
    if after != manifest:
        raise ValueError("benchmark inputs changed during repeated measurements")
    rows, details, failures = summarize(folder, config["repeat_processes"])
    measured = {row["setting_id"]: row for row in rows}
    targets = [dict(target=choice["target"], met_target=measured[choice["setting_id"]]["recall"] is not None
                    and measured[choice["setting_id"]]["recall"] >= choice["target"],
                    **measured[choice["setting_id"]]) for choice in selected]
    # Report the selected settings even if a repeated run misses its target;
    # do not select replacements from this second set of timings.
    write_csv(output / "results.csv", targets)
    save_json(output / "summary.json", dict(targets=targets, settings=details, failures=failures,
        inputs_manifest_sha256=file_hash(output / "inputs-manifest.json"),
        inputs_unchanged=after == manifest,
        scope="Best observed parameter choices, remeasured on the same fixed benchmark. This is not an unseen-query test or a claim of a global optimum."))
    print(f"Repeated {len(rows)} distinct settings in three processes; input hashes unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sweep", "extend", "repeat"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.stage == "sweep":
        sweep(arguments.output.resolve())
    elif arguments.stage == "extend":
        extend(arguments.output.resolve())
    else:
        repeat(arguments.output.resolve())
