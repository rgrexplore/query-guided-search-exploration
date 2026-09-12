"""Compare three search methods on one document set and three query examples.

Run the declared settings on the same queries, then report the fastest setting
that meets the recall and memory limits. This is a descriptive comparison of a
small tested set; it does not establish a global optimum.
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
from experiments.components import pack_signs
from experiments.controlled_study import prepare_layout
from experiments.isolated import ROOT, execute_cases
from experiments.native_scaling import BYTE_COUNTS, binary_reference


EXAMPLES = {
    "first12": "Large values at the start",
    "spread12": "Large values in different positions",
    "equal": "All values equal in size",
}


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_query_examples(dimensions=256, count=32, strong_bits=12, seed=20260913):
    # Reuse the signs in all three examples. Only the coordinate sizes change.
    signs = np.random.default_rng(seed).integers(0, 2, (count, dimensions), dtype=np.uint8)
    signed_values = 2 * signs.astype(np.float32) - 1
    position_random = np.random.default_rng(seed + 1)
    first_positions = [np.arange(strong_bits) for _ in range(count)]
    spread_positions = [np.sort(position_random.choice(dimensions, strong_bits, replace=False))
                        for _ in range(count)]
    examples = {}
    for name, positions in (("first12", first_positions), ("spread12", spread_positions), ("equal", None)):
        magnitudes = np.ones((count, dimensions), dtype=np.float32)
        if positions is not None:
            magnitudes.fill(.0001)
            for row, selected in enumerate(positions):
                magnitudes[row, selected] = 1
        queries = signed_values * magnitudes
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        examples[name] = (np.ascontiguousarray(queries), positions)
    return examples


def equal_weight_reference(codes, query, top_k):
    """Equal weights rank by mismatch count, then by the original document ID."""
    ideal = pack_signs((query >= 0)[None, :])[0]
    errors = np.empty(len(codes), dtype=np.int32)
    for start in range(0, len(codes), 65536):
        difference = np.bitwise_xor(codes[start:start + 65536], ideal)
        errors[start:start + len(difference)] = BYTE_COUNTS[difference.view(np.uint8)].sum(axis=1)
    threshold = np.partition(errors, top_k - 1)[top_k - 1]
    rows = np.flatnonzero(errors <= threshold)
    return rows[np.lexsort((rows, errors[rows]))[:top_k]]


def verified_documents(source, config):
    metadata = json.loads((source / "pool.json").read_text())
    for name, value in (("documents", config["documents"]),
                        ("dimensions", config["dimensions"]), ("seed", config["document_seed"])):
        if metadata["identity"][name] != value:
            raise ValueError(f"document source has a different {name}")
    for name in ("codes.npy", "documents.npy"):
        if file_hash(source / name) != metadata["hashes"][name]:
            raise ValueError(f"document source hash differs: {name}")
    return metadata


def prepare_inputs(output, source, config):
    original = verified_documents(source, config)
    shared = output / "documents"
    shared.mkdir()
    for name in ("codes.npy", "documents.npy"):
        (shared / name).symlink_to((source / name).resolve())
    save_json(shared / "source.json", dict(path=str(source), hashes=original["hashes"]))
    codes = np.load(shared / "codes.npy", mmap_mode="r")
    examples = make_query_examples(config["dimensions"], config["queries"],
                                   config["large_coordinates"], config["query_seed"])
    pools = {}
    for name, (queries, positions) in examples.items():
        pool = output / "inputs" / name
        pool.mkdir(parents=True)
        for filename in ("codes.npy", "documents.npy"):
            (pool / filename).symlink_to((shared / filename).resolve())
        references, match_counts = [], []
        for row, query in enumerate(queries):
            if positions is None:
                references.append(equal_weight_reference(codes, query, config["top_k"]))
            else:
                reference, count = binary_reference(codes, query, positions[row],
                                                    config["dimensions"], config["top_k"])
                references.append(reference)
                match_counts.append(count)
        np.save(pool / "queries.npy", queries)
        np.save(pool / "reference.npy", np.array(references))
        hashes = {filename: file_hash(pool / filename)
                  for filename in ("queries.npy", "reference.npy")}
        hashes["codes.npy"] = original["hashes"]["codes.npy"]
        hashes["documents.npy"] = original["hashes"]["documents.npy"]
        save_json(pool / "pool.json", dict(
            example=name, label=EXAMPLES[name], hashes=hashes,
            query_ids=[f"seed{config['query_seed']}-q{row}" for row in range(len(queries))],
            large_positions=None if positions is None else [list(map(int, row)) for row in positions],
            matching_all_large_coordinates=match_counts,
            scope="All methods read these same query/reference arrays. Every generated query is retained."))
        pools[name] = pool
    assert len({json.loads((pool / "pool.json").read_text())["hashes"]["codes.npy"]
                for pool in pools.values()}) == 1
    return shared, pools


def prepare_layouts(shared, source, config):
    layouts = {"global": dict(path=None, kind="none", clusters=1, routing_bits=0, probes=1)}
    requests = [("direct", 4096, 1), ("direct", 8192, 2),
                ("direct", 16, 16), ("ivf", 256, 256)]
    (shared / "routers").mkdir()
    for kind, clusters, probes in requests:
        directory = f"{kind}-{clusters}-seed{config['document_seed']}"
        cached = source / "routers" / directory
        if cached.exists():
            metadata = json.loads((cached / "router.json").read_text())
            for name in ("assignments", "centroids"):
                path = cached / f"{name}.npy"
                if path.exists() and file_hash(path) != metadata[f"{name}_sha256"]:
                    raise ValueError(f"cached router hash differs: {path}")
            (shared / "routers" / directory).symlink_to(cached.resolve(), target_is_directory=True)
        folder = prepare_layout(shared, kind, clusters, config["document_seed"])
        metadata = json.loads((folder / "router.json").read_text())
        layouts[f"{kind}{clusters}"] = dict(path=str(folder), kind=kind, clusters=clusters,
                                          routing_bits=metadata["routing_bits"], probes=probes)
    return {
        "first12": [layouts[name] for name in ("global", "direct4096", "direct8192")],
        "spread12": [layouts[name] for name in ("global", "ivf256", "direct16")],
        "equal": [layouts[name] for name in ("global", "ivf256", "direct16")],
    }


def comparison_cases(example, pool, layouts, documents=1_000_000, dimensions=256, query_count=32):
    cases = []
    for layout in layouts:
        common = dict(example=example, pool=str(pool), documents=documents, dimensions=dimensions,
                      query_rows=list(range(query_count)), router=layout["path"],
                      router_kind=layout["kind"], clusters=layout["clusters"], probes=layout["probes"],
                      top_k=100, repetitions=2, ram_budget_bytes=32_000_000_000)
        cases.append(dict(common, method="scan"))
        leaf_sizes = [128, 288, 320, documents] if example == "first12" else [288, 320, documents]
        for leaf in leaf_sizes:
            budget = 4096 if example == "equal" and leaf < documents else 0
            cases.append(dict(common, method="branch", leaf_size=leaf, node_budget=budget))
        if example == "first12":
            widths = [1, 12] if layout["kind"] == "none" else [max(1, 12 - layout["routing_bits"])]
        else:
            widths = [1, 8, 12]
        for width in widths:
            cases.append(dict(common, method="keys", key_bits=width, key_offset=layout["routing_bits"],
                              candidate_target=0, key_limit=0))
    for number, case in enumerate(cases):
        case["setting_id"] = f"{example}-{number:02d}"
    return cases


def summarize(output, config):
    completed, failures = load_cases(output / "runs")
    schedule = json.loads((output / "runs" / "schedule.json").read_text())
    original_cases = {case["setting_id"]: case for case in schedule}
    by_setting = defaultdict(list)
    for item in completed:
        by_setting[item["case"]["setting_id"]].append(item)
    rows, details = [], []
    for identifier, case in original_cases.items():
        items = by_setting[identifier]
        observations = [row for item in items for row in item["queries"]]
        first = [row for item in items[:1] for row in item["queries"] if row["repetition"] == 0]
        # Repeated timings do not create more query examples. Check that work
        # and recall agree before reducing the repeats to one descriptive row.
        stable = ("recall", "documents_scored", "bitplane_words", "leaf_words", "key_attempts")
        expected = {row["query"]: tuple(row[key] for key in stable) for row in first}
        assert all(tuple(row[key] for key in stable) == expected[row["query"]] for row in observations)
        process_medians = [float(np.median([row["query_ms"] for row in item["queries"]])) for item in items]
        complete = len(items) == config["process_repeats"]
        memory_ok = complete and all(item["result"]["memory"]["budget_status"] == "fits_by_lifetime_peak"
                                     for item in items)
        power_ok = complete and all(item["result"]["power_before"]["available"]
                                    and item["result"]["power_before"] == item["result"]["power_after"]
                                    for item in items)
        recall = float(np.mean([row["recall"] for row in first])) if first else None
        eligible = bool(memory_ok and power_ok and recall is not None and recall >= config["recall_target"])
        row = dict(example=case["example"], label=EXAMPLES[case["example"]], setting_id=identifier,
                   method=case["method"], clusters=case["clusters"], probes=case["probes"],
                   router=case["router_kind"], leaf_size=case.get("leaf_size", 0),
                   node_budget=case.get("node_budget", 0), key_bits=case.get("key_bits", 0),
                   recall=recall, p50_ms=float(np.median([row["query_ms"] for row in observations])) if observations else None,
                   process_p50_min_ms=min(process_medians) if items else None,
                   process_p50_max_ms=max(process_medians) if items else None,
                   mean_scored=float(np.mean([row["documents_scored"] for row in first])) if first else None,
                   mean_split_words=float(np.mean([row["bitplane_words"] for row in first])) if first else None,
                   mean_leaf_words=float(np.mean([row["leaf_words"] for row in first])) if first else None,
                   mean_key_lookups=float(np.mean([row["key_attempts"] for row in first])) if first else None,
                   logical_index_bytes=max((item["result"]["storage"]["logical_bytes"] for item in items), default=None),
                   index_array_capacity_bytes=max((item["result"]["storage"]["array_capacity_bytes"] for item in items), default=None),
                   lifetime_peak_bytes=max((item["result"]["memory"]["lifetime_peak_bytes"] for item in items), default=None),
                   complete_processes=len(items), memory_ok=memory_ok, power_unchanged=power_ok, eligible=eligible)
        rows.append(row)
        details.append(dict(**row, process_p50_ms=process_medians, case=case,
                            observed_queries=len(first), timing_observations=len(observations)))
    selected = []
    for example in EXAMPLES:
        for method in ("scan", "branch", "keys"):
            choices = [row for row in rows if row["example"] == example and row["method"] == method and row["eligible"]]
            if choices:
                selected.append(min(choices, key=lambda row: (row["p50_ms"], row["setting_id"])))
    write_csv(output / "settings.csv", rows)
    if selected:
        write_csv(output / "selected.csv", selected)
    save_json(output / "summary.json", dict(
        scope="Fastest eligible tested settings on this same query batch; descriptive, not a global optimum.",
        recall_target=config["recall_target"], settings=details, selected=selected, failures=failures,
        memory_scope="Logical payload and reserved array capacity are not process RAM. Lifetime peak includes index construction.",
        repeat_scope="The range contains the three process medians. It is not a confidence interval."))
    print(f"Saved {len(rows)} settings; {len(selected)} eligible method/example choices; {len(failures)} failed processes.")


def run(output, source):
    config = dict(documents=1_000_000, dimensions=256, document_seed=73, query_seed=20260913,
                  queries=32, large_coordinates=12, small_coordinate_weight=.0001, top_k=100,
                  recall_target=.99, process_repeats=3, query_repetitions=2, schedule_seed=20260914,
                  limits=dict(ram_budget_bytes=32_000_000_000, worker_stop_bytes=34_359_738_368, case_seconds=180))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "configuration.json", config)
    paths = [Path(__file__), ROOT / "experiments/worker.py", ROOT / "experiments/isolated.py",
             ROOT / "experiments/native_scaling.py", ROOT / "experiments/controlled_study.py",
             ROOT / "experiments/direct_routing.py", Path(bitplane_index.__file__)]
    save_json(output / "source-hashes.json", {str(path): file_hash(path) for path in paths})
    with threadpool_limits(limits=1):
        shared, pools = prepare_inputs(output, source, config)
        layouts = prepare_layouts(shared, source, config)
    save_json(output / "layouts.json", layouts)
    base_cases = [case for example in EXAMPLES
                  for case in comparison_cases(example, pools[example], layouts[example])]
    cases = [dict(case, block=block) for block in range(config["process_repeats"]) for case in base_cases]
    order = np.random.default_rng(config["schedule_seed"]).permutation(len(cases))
    cases = [cases[int(index)] for index in order]
    runs = output / "runs"
    runs.mkdir()
    save_json(runs / "configuration.json", config)
    print(f"Running {len(base_cases)} settings × {config['process_repeats']} processes.", flush=True)
    execute_cases(config, runs, cases, "Same document set and same 32 queries per example for all methods; one CPU thread.")
    summarize(output, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path,
                        default=ROOT / "data/controlled-fixed-2026-09-10/n1000000")
    args = parser.parse_args()
    run(args.output.resolve(), args.source_pool.resolve())


if __name__ == "__main__":
    main()
