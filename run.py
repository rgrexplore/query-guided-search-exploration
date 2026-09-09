"""The whole path: load data, encode it once, compare searches, save a report."""

import argparse
import json
import math
import tomllib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from analyze import analyze_run
from benchmark import plot_results, run_benchmark, write_trace
from data import load_dataset
from embeddings import prepare_embeddings


def load_config(path):
    """Catch misspelled settings and invalid budgets before any data work starts."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    required = {
        "data": ["dataset", "cache_dir", "max_documents", "max_queries"],
        "embedding": ["model", "dimensions", "batch_size", "max_length", "device"],
        "routing": ["methods", "clusters", "sign_bits", "probes", "seed", "threads"],
        "search": [
            "candidate_limit",
            "top_k",
            "node_budgets",
            "leaf_size",
            "explore_probabilities",
            "seeds",
        ],
        "benchmark": ["warmup_queries", "repetitions", "output_dir"],
    }
    optional = {
        "embedding": {"revision"},
        "benchmark": {"complete_routing_endpoint", "bootstrap_samples", "analysis_seed"},
    }
    if set(config) != set(required):
        raise ValueError(f"Expected config sections: {', '.join(required)}")
    for section, names in required.items():
        # An unknown key is usually a typo. Silently ignoring it makes runs confusing.
        missing = set(names) - set(config[section])
        unknown = set(config[section]) - set(names) - optional.get(section, set())
        if missing or unknown:
            raise ValueError(f"{section}: missing {sorted(missing)}, unknown {sorted(unknown)}")

    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def integer_list(values, name, minimum=1):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
        for value in values:
            integer(value, name, minimum)

    for name in ("max_documents", "max_queries"):
        integer(config["data"][name], name, 0)
    for name in ("dimensions", "batch_size", "max_length"):
        integer(config["embedding"][name], name)
    for name in ("clusters", "sign_bits", "threads"):
        integer(config["routing"][name], name)
    integer(config["routing"]["seed"], "routing.seed", 0)
    integer_list(config["routing"]["probes"], "probes")
    methods = config["routing"]["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or len(set(methods)) != len(methods)
        or set(methods) - {"ivf", "sign"}
    ):
        raise ValueError("routing.methods must contain 'ivf' and/or 'sign', without duplicates")
    for name in ("candidate_limit", "top_k", "leaf_size"):
        integer(config["search"][name], name)
    if config["search"]["candidate_limit"] < config["search"]["top_k"]:
        raise ValueError("candidate_limit must be at least top_k")
    integer_list(config["search"]["node_budgets"], "node_budgets", 0)
    integer_list(config["search"]["seeds"], "seeds", 0)
    probabilities = config["search"]["explore_probabilities"]
    if (
        not isinstance(probabilities, list)
        or not probabilities
        or len(set(probabilities)) != len(probabilities)
    ):
        raise ValueError("explore_probabilities must be a nonempty list without duplicates")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in probabilities
    ):
        raise ValueError("explore_probabilities must be between 0 and 1")
    integer(config["benchmark"]["warmup_queries"], "warmup_queries", 0)
    integer(config["benchmark"]["repetitions"], "repetitions")
    options = config["benchmark"]
    if not isinstance(options.get("complete_routing_endpoint", False), bool):
        raise ValueError("complete_routing_endpoint must be a boolean")
    integer(options.get("bootstrap_samples", 1000), "bootstrap_samples")
    integer(options.get("analysis_seed", 20260909), "analysis_seed", 0)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment.toml"))
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Cache the data and embeddings, then stop",
    )
    parser.add_argument(
        "--eval-queries",
        type=int,
        default=0,
        help="Evaluate just the first N queries without changing the embedding cache",
    )
    parser.add_argument(
        "--trace-query",
        help="Dataset query ID for the saved trace; defaults to the first query",
    )
    args = parser.parse_args()
    if args.eval_queries < 0:
        parser.error("--eval-queries must be >= 0")
    config = load_config(args.config)
    root = args.config.resolve().parent
    data_options, embedding = config["data"], config["embedding"]
    cache_dir = root / data_options["cache_dir"]

    print("[1/5] Load documents, queries and relevance labels", flush=True)
    start = perf_counter()
    dataset = load_dataset(
        cache_dir,
        data_options["dataset"],
        data_options["max_documents"],
        data_options["max_queries"],
    )
    loading_ms = (perf_counter() - start) * 1000
    print(
        f"{len(dataset.corpus_ids):,} documents, {len(dataset.query_ids):,} queries",
        flush=True,
    )

    print("[2/5] Prepare embeddings (reuse the cache when it matches)", flush=True)
    start = perf_counter()
    arrays = prepare_embeddings(
        dataset,
        cache_dir,
        model_name=embedding["model"],
        model_revision=embedding.get("revision"),
        dimensions=embedding["dimensions"],
        batch_size=embedding["batch_size"],
        max_length=embedding["max_length"],
        device=embedding["device"],
    )
    preparation_ms = (perf_counter() - start) * 1000
    if args.prepare_only:
        print(f"Prepared arrays in {cache_dir}", flush=True)
        return

    # This limit changes the evaluation, not the document/query encoding cache.
    if args.eval_queries and args.eval_queries < len(dataset.query_ids):
        limit = args.eval_queries
        dataset = replace(
            dataset,
            query_ids=dataset.query_ids[:limit],
            queries=dataset.queries[:limit],
        )
        arrays = replace(
            arrays,
            queries=arrays.queries[:limit],
            full_queries=arrays.full_queries[:limit],
        )
    if args.trace_query and args.trace_query not in dataset.query_ids:
        parser.error("--trace-query must be one of the selected evaluation query IDs")

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = root / config["benchmark"]["output_dir"] / run_name
    print("[3/5] Build buckets and compare searches", flush=True)
    summary, metadata = run_benchmark(dataset, arrays, config, output_dir)
    metadata["preparation_call_ms"] = preparation_ms
    metadata["dataset_loading_ms"] = loading_ms
    metadata["preparation_note"] = "This is the current call time; a matching cache skips encoding."
    print("[4/5] Save measurements", flush=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[5/5] Plot latency and quality", flush=True)
    plot_results(summary, metadata, output_dir)
    write_trace(dataset, arrays, config, output_dir, args.trace_query)
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    analysis = analyze_run(
        output_dir,
        bootstrap_samples=config["benchmark"].get("bootstrap_samples", 1000),
        seed=config["benchmark"].get("analysis_seed", 20260909),
    )
    print(f"Analysis: {analysis['report']}", flush=True)


if __name__ == "__main__":
    main()
