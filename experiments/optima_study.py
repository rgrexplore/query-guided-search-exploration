"""Prepare, tune, and evaluate the declared fixed/adaptive query-weight study.

Each stage is a separate command so predictions can be frozen after tuning and
before final evaluation. Search itself stays in the existing isolated worker.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import bitplane_index
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.choices import combine
from experiments.controlled_study import evaluation_cases, layout_probes, prepare_layout
from experiments.evaluation_report import report
from experiments.isolated import ROOT, execute_cases
from experiments.native_scaling import binary_reference, make_queries


def default_config():
    return {
        "data": dict(documents=1_000_000, dimensions=256, document_seed=73,
                     query_seed=20260912, queries=96, tuning_queries=32,
                     strong_bits=12, weak_weight=.0001),
        "search": dict(top_k=100, recall_targets=[.8, .9, .95, .99]),
        "sweep": dict(direct_bits=[4, 8, 10, 11, 12, 13, 14], ivf_clusters=[256, 4096],
                      global_leaf_sizes=[128, 256, 288, 320, 384, 640, 1280, 4096],
                      routed_leaf_sizes=[128, 320], global_key_bits=[1, 8, 11, 12, 13, 14]),
        "measurement": dict(repetitions=2, evaluation_repetitions=2, blocks=3,
                            schedule_seed=20260913),
        "limits": dict(ram_budget_bytes=32_000_000_000,
                       worker_stop_bytes=34_359_738_368, case_seconds=180),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_hashes():
    names = ["experiments/optima_study.py", "experiments/native_scaling.py",
             "experiments/controlled_study.py", "experiments/isolated.py",
             "experiments/worker.py", "experiments/choices.py",
             "experiments/evaluation_report.py", "experiments/direct_routing.py",
             "experiments/real_study.py", "experiments/probe_cutoffs.py",
             "cpp/index.cpp", "cpp/key_index.cpp", "cpp/score.hpp"]
    hashes = {name: file_hash(ROOT / name) for name in names}
    hashes["native_module"] = file_hash(bitplane_index.__file__)
    return hashes


def verify_source_pool(source, config):
    """The old query distribution does not matter; the document arrays must match."""
    metadata = json.loads((source / "pool.json").read_text())
    data = config["data"]
    for name, expected in (("documents", data["documents"]),
                           ("dimensions", data["dimensions"]), ("seed", data["document_seed"])):
        if metadata["identity"][name] != expected:
            raise ValueError(f"source document identity differs: {name}")
    for name in ("codes.npy", "documents.npy"):
        if file_hash(source / name) != metadata["hashes"][name]:
            raise ValueError(f"source hash mismatch: {name}")
    return metadata


def prepare_fresh_pool(config, folder, support, source, verified):
    data = config["data"]
    # make_queries uses this exact two-weight distribution. Do not silently
    # accept a different value in a configuration and label it as tested.
    if data["weak_weight"] != .0001:
        raise ValueError("this study's query generator uses weak_weight=0.0001")
    if not 0 < data["tuning_queries"] < data["queries"]:
        raise ValueError("tuning and evaluation both need separate query rows")
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("codes.npy", "documents.npy"):
        (folder / name).symlink_to((source / name).resolve())
    queries, supports = make_queries(data["dimensions"], data["queries"],
                                    data["strong_bits"], support, data["query_seed"])
    old_queries = np.load(source / "queries.npy", mmap_mode="r")
    old_rows = {row.tobytes() for row in old_queries}
    if any(row.tobytes() in old_rows for row in queries):
        raise ValueError("the new query vectors overlap the source study")

    codes = np.load(folder / "codes.npy", mmap_mode="r")
    references, preferred_counts = [], []
    for query, positions in zip(queries, supports, strict=True):
        truth, strong_matches = binary_reference(
            codes, query, positions, data["dimensions"], config["search"]["top_k"])
        references.append(truth)
        # This counts documents directly from their stored bits. It does not
        # call either search method, so the later work check is independent.
        active = np.ones(len(codes), dtype=bool)
        counts = [len(codes)]
        for bit in positions:
            signs = (codes[:, bit // 64] >> np.uint64(bit % 64)) & np.uint64(1)
            active &= signs == (query[bit] >= 0)
            counts.append(int(active.sum()))
        assert counts[-1] == strong_matches
        preferred_counts.append(counts)
    arrays = {"queries.npy": queries, "reference.npy": np.array(references),
              "preferred_counts.npy": np.array(preferred_counts, dtype=np.int64)}
    for name, array in arrays.items():
        np.save(folder / name, array)
    query_ids = [f"{support}-seed{data['query_seed']}-q{row}" for row in range(len(queries))]
    split = data["tuning_queries"]
    hashes = {name: verified["hashes"][name] for name in ("codes.npy", "documents.npy")}
    hashes.update({name: file_hash(folder / name) for name in arrays})
    metadata = dict(
        identity=dict(data, support=support, top_k=config["search"]["top_k"]),
        hashes=hashes, query_ids=query_ids, tuning_query_ids=query_ids[:split],
        evaluation_query_ids=query_ids[split:], supports=[list(map(int, row)) for row in supports],
        strong_match_counts=[row[-1] for row in preferred_counts],
        source_pool=str(source), source_pool_sha256=file_hash(source / "pool.json"),
        expected_strong_matches=len(codes) / 2 ** data["strong_bits"],
        scope="Constructed weights with fresh queries; no query was removed for low match count.")
    write_json(folder / "pool.json", metadata)
    return folder


def reuse_router(pool, source, kind, clusters, seed):
    """Reuse an existing layout only after its document source was verified."""
    name = f"{kind}-{clusters}-seed{seed}"
    previous = source / "routers" / name
    if not previous.exists():
        return
    metadata = json.loads((previous / "router.json").read_text())
    for array in ("assignments", "centroids"):
        path = previous / f"{array}.npy"
        if path.exists() and file_hash(path) != metadata[f"{array}_sha256"]:
            raise ValueError(f"cached router hash mismatch: {path}")
    (pool / "routers").mkdir(exist_ok=True)
    (pool / "routers" / name).symlink_to(previous.resolve(), target_is_directory=True)


def tuning_probes(config, pool, support, layout):
    if layout["kind"] == "none":
        return [1]
    base = dict(pool=str(pool), router=layout["path"], router_kind=layout["kind"],
                query_rows=list(range(config["data"]["tuning_queries"])))
    counts = set(layout_probes(base, config["search"]["recall_targets"] + [1.0]))
    if support == "fixed" and layout["kind"] == "direct":
        # Opening all weak-bit suffixes preserves every strong-bit match.
        extra_bits = max(0, layout["routing_bits"] - config["data"]["strong_bits"])
        counts.add(2 ** extra_bits)
    return sorted(counts)


def study_cases(config, pool, support, layouts):
    data, sweep = config["data"], config["sweep"]
    cases = []
    for layout in layouts:
        global_search = layout["kind"] == "none"
        leaf_sizes = set(sweep["global_leaf_sizes"] if global_search else sweep["routed_leaf_sizes"])
        leaf_sizes.add(data["documents"])
        key_bits = (set(sweep["global_key_bits"]) if global_search else
                    {1, 4, max(1, data["strong_bits"] - layout["routing_bits"])})
        for probes in layout["probes"]:
            shared = dict(pool=str(pool), support=support, documents=data["documents"],
                          dimensions=data["dimensions"], query_rows=list(range(data["tuning_queries"])),
                          router=layout["path"], router_kind=layout["kind"], clusters=layout["clusters"],
                          probes=probes, top_k=config["search"]["top_k"],
                          repetitions=config["measurement"]["repetitions"],
                          ram_budget_bytes=config["limits"]["ram_budget_bytes"])
            cases.append(dict(shared, method="scan"))
            for leaf in sorted(leaf_sizes):
                cases.append(dict(shared, method="branch", node_budget=0, leaf_size=leaf))
            for bits in sorted(key_bits):
                cases.append(dict(shared, method="keys", key_bits=bits,
                                  key_offset=layout["routing_bits"], candidate_target=0, key_limit=0))
    return cases


def shuffled(cases, seed):
    order = np.random.default_rng(seed).permutation(len(cases))
    return [cases[int(index)] for index in order]


def prepare(output, config, source):
    verified = verify_source_pool(source, config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(output / "source-hashes.json", source_hashes())
    for support in ("fixed", "adaptive"):
        folder = output / support
        folder.mkdir()
        write_json(folder / "configuration.json", config)
        pool = prepare_fresh_pool(config, folder / "pool", support, source, verified)
        layouts = [dict(path=None, kind="none", clusters=1, routing_bits=0, probes=[1])]
        requests = [("direct", 2 ** bits) for bits in config["sweep"]["direct_bits"]]
        requests += [("ivf", count) for count in config["sweep"]["ivf_clusters"]]
        router_files = {}
        with threadpool_limits(limits=1):
            for kind, clusters in requests:
                reuse_router(pool, source, kind, clusters, config["data"]["document_seed"])
                router = prepare_layout(pool, kind, clusters, config["data"]["document_seed"])
                metadata = json.loads((router / "router.json").read_text())
                layout = dict(path=str(router), kind=kind, clusters=clusters,
                              routing_bits=metadata["routing_bits"])
                layout["probes"] = tuning_probes(config, pool, support, layout)
                layouts.append(layout)
                router_files[str(router)] = {path.name: file_hash(path) for path in router.iterdir()
                                           if path.is_file()}
        write_json(folder / "layouts.json", layouts)
        write_json(folder / "router-hashes.json", router_files)
        cases = shuffled(study_cases(config, pool, support, layouts), config["measurement"]["schedule_seed"])
        write_json(folder / "tuning-cases.json", cases)
        print(f"Prepared {support}: {len(layouts)} layouts, {len(cases)} tuning cases.", flush=True)


def tune(folder):
    config = json.loads((folder / "configuration.json").read_text())
    cases = json.loads((folder / "tuning-cases.json").read_text())
    output = folder / "tuning"
    output.mkdir(exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(output / "source-hashes.json", source_hashes())
    execute_cases(config, output, cases, "Tuning queries only; all methods receive every declared routing layout.")
    combine([output], folder / "frozen", config["search"]["recall_targets"])


def final_cases(config, measured_choices, formula_choices):
    choices = measured_choices + formula_choices
    identifiers = [choice["setting_id"] for choice in choices]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("measured and formula setting IDs must be unique")
    # evaluation_cases changes query rows, repetitions and process blocks only.
    # Search settings remain exactly as supplied by each selection procedure.
    return evaluation_cases(config, choices)


def evaluate(folder, formula_path=None):
    config = json.loads((folder / "configuration.json").read_text())
    shortlist_path = folder / "frozen" / "shortlist.json"
    measured = json.loads(shortlist_path.read_text())
    formula = []
    if formula_path is not None:
        formula = [choice for choice in json.loads(formula_path.read_text())
                   if choice.get("support", folder.name) == folder.name]
    cases = final_cases(config, measured, formula)
    output = folder / "evaluation"
    output.mkdir(exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(output / "source-hashes.json", source_hashes())
    write_json(output / "frozen-shortlist.json", measured + formula)
    metadata = json.loads((folder / "pool" / "pool.json").read_text())
    write_json(output / "evaluation-source.json", dict(
        shortlist_sha256=file_hash(shortlist_path),
        formula_choices_sha256=None if formula_path is None else file_hash(formula_path),
        query_ids=metadata["evaluation_query_ids"], query_hash=metadata["hashes"]["queries.npy"],
        reference_hash=metadata["hashes"]["reference.npy"],
        scope="Fresh final query subset; measured-best and formula choices frozen before this stage."))
    cases = shuffled(cases, config["measurement"]["schedule_seed"] + 1)
    execute_cases(config, output, cases, "Frozen choices on disjoint final queries; no selection from these results.")
    report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "tune", "evaluate"))
    parser.add_argument("--output", type=Path, required=True, help="Study folder shared by the three stages.")
    parser.add_argument("--support", choices=("fixed", "adaptive"),
                        help="Run one condition during tune/evaluate; omitted means both.")
    parser.add_argument("--config", type=Path, help="Optional complete JSON configuration for prepare.")
    parser.add_argument("--source-pool", type=Path,
                        default=ROOT / "data/controlled-fixed-2026-09-10/n1000000",
                        help="Existing document pool; its hashes are checked before reuse.")
    parser.add_argument("--formula-choices", type=Path,
                        help="Additional frozen shortlist records for evaluate; use unique setting IDs.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        config = default_config() if args.config is None else json.loads(args.config.read_text())
        prepare(output, config, args.source_pool.resolve())
        return
    supports = [args.support] if args.support else ["fixed", "adaptive"]
    for support in supports:
        folder = output / support
        if args.stage == "tune":
            tune(folder)
        else:
            evaluate(folder, args.formula_choices)


if __name__ == "__main__":
    main()
