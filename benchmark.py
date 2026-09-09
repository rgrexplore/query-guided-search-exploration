"""Run the same queries through each search method, then save the tradeoffs."""

import csv
import hashlib
import json
import math
import platform
import random
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

import bitplane_index
from bitplane_index import Index
from routing import build_router, exact_search


# --- Small metrics. Document IDs and array rows are different things. ---


def neighbor_recall(rows, reference):
    """How many reference neighbors survived? Padding and duplicate rows don't count."""
    expected = {int(row) for row in reference if row >= 0}
    found = {int(row) for row in rows if row >= 0}
    return len(found & expected) / len(expected) if expected else 0.0


def relevance_metrics(document_ids, judgments, top_k):
    """Use the dataset's labels here, not the exact vector search results."""
    grades = [max(0, judgments.get(doc_id, 0)) for doc_id in document_ids[:top_k]]
    ideal = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)[:top_k]

    def dcg(values):
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values))

    ideal_score = dcg(ideal)
    relevant_count = sum(grade > 0 for grade in judgments.values())
    hits = sum(grade > 0 for grade in grades)
    return {
        "ndcg": dcg(grades) / ideal_score if ideal_score else 0.0,
        "precision": hits / top_k,
        "recall": hits / relevant_count if relevant_count else 0.0,
    }


def rerank(rows, documents, query, top_k):
    """All methods get the same float reranker and the same candidate allowance."""
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[rows >= 0]
    if len(rows) == 0:
        return rows

    # Gather only the returned rows. A missed candidate can't come back at this stage.
    scores = documents[rows] @ query

    # lexsort uses the last key first: highest score, then lowest row ID for a tie.
    order = np.lexsort((rows, -scores))[:top_k]
    return rows[order]


def summarize(measurements):
    """Average quality over queries/repeats, but keep each random seed separate."""
    keys = ("method", "probes", "node_budget", "explore_probability", "seed")
    groups = defaultdict(list)
    for row in measurements:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for identity, rows in sorted(groups.items()):
        summary = dict(zip(keys, identity))
        summary["measurements"] = len(rows)
        for key in (
            "float_recall",
            "binary_recall",
            "ndcg",
            "precision",
            "relevance_recall",
            "candidates",
            "nodes",
            "random_nodes",
            "documents_scored",
            "budget_exhausted",
            "bitplane_words",
            "routed_documents",
            "selected_buckets",
            "routing_float_recall",
            "empty_result",
            "fewer_than_k",
            "candidate_fill_rate",
        ):
            values = [row[key] for row in rows if row.get(key) is not None]
            summary[key] = float(np.mean(values)) if values else None
        for key in ("retrieval", "core", "native", "routing", "rerank"):
            values = [row[f"{key}_ms"] for row in rows if row.get(f"{key}_ms") is not None]
            for percentile in (50, 95):
                summary[f"{key}_p{percentile}_ms"] = (
                    float(np.percentile(values, percentile)) if values else None
                )
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    if not rows:
        raise ValueError("There are no measurements to write.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- Each setting changes one native search policy; the data stays put. ---


def _settings(search, seed):
    settings = [dict(kind="scan", node_budget=0, explore_probability=0.0, seed=seed)]
    for budget in search["node_budgets"]:
        for probability in search["explore_probabilities"]:
            # Deterministic search needs one seed. Random search gets several trials.
            seeds = search["seeds"] if probability else [seed]
            for run_seed in seeds:
                settings.append(
                    dict(
                        kind="branch",
                        node_budget=budget,
                        explore_probability=probability,
                        seed=run_seed,
                    )
                )
    return settings


def _query_run(
    index,
    router,
    query,
    full_query,
    full_documents,
    expected_buckets,
    probes,
    search,
    setting,
    query_number,
):
    """This wall clock includes routing, the Python/C++ call, and float reranking."""
    start = perf_counter()
    if setting["kind"] == "float":
        # Faiss does routing inside this call, so its routing time isn't separated out.
        core_start = perf_counter()
        rows, _, _ = router.ivf_search(query[None], search["candidate_limit"], probes)
        candidates = rows[0]
        core_ms = (perf_counter() - core_start) * 1000
        routing_ms = None
        stats = dict(
            elapsed_ms=None,
            nodes=0,
            random_nodes=0,
            bitplane_words=0,
            documents_scored=None,
            stop_reason="faiss",
        )
    else:
        buckets, _ = router.select(query[None], probes)
        routing_ms = (perf_counter() - start) * 1000
        core_start = perf_counter()
        if setting["kind"] == "scan":
            result = index.scan(query[None], buckets, search["candidate_limit"])
        else:
            result = index.search(
                query[None],
                buckets,
                search["candidate_limit"],
                node_budget=setting["node_budget"],
                leaf_size=search["leaf_size"],
                explore_probability=setting["explore_probability"],
                seed=setting["seed"] + query_number,
            )
        core_ms = (perf_counter() - core_start) * 1000
        valid_count = result["counts"][0]
        candidates = result["rows"][0, :valid_count]
        stats = result["stats"][0]
    rerank_start = perf_counter()
    final_rows = rerank(candidates, full_documents, full_query, search["top_k"])
    stop = perf_counter()
    # Check after timing. A routing change must not slip into a scan/branch comparison.
    if setting["kind"] != "float" and not np.array_equal(buckets[0], expected_buckets):
        raise RuntimeError("The router changed its bucket selection between methods.")
    return (
        candidates,
        final_rows,
        stats,
        {
            "routing_ms": routing_ms,
            "core_ms": core_ms,
            "native_ms": stats["elapsed_ms"],
            "rerank_ms": (stop - rerank_start) * 1000,
            "retrieval_ms": (stop - start) * 1000,
        },
    )


def _run_metadata(dataset, arrays, config):
    """Keep the inputs and environment beside the results so runs can be compared."""
    source_dir = Path(__file__).resolve().parent
    sources = [
        "run.py",
        "data.py",
        "embeddings.py",
        "routing.py",
        "benchmark.py",
        "cpp/index.hpp",
        "cpp/index.cpp",
        "cpp/bindings.cpp",
        "CMakeLists.txt",
        "pyproject.toml",
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "faiss-cpu",
                "torch",
                "transformers",
                "sentence-transformers",
                "matplotlib",
            )
        },
        "source_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in sources
        },
        "native_sha256": hashlib.sha256(Path(bitplane_index.__file__).read_bytes()).hexdigest(),
        "dataset": {
            "source": dataset.source,
            "fingerprint": dataset.fingerprint,
            "documents": len(dataset.corpus_ids),
            "queries": len(dataset.query_ids),
            "full_documents": dataset.full_document_count,
            "full_queries": dataset.full_query_count,
            "is_subset": dataset.is_subset,
            "query_ids_sha256": hashlib.sha256(json.dumps(dataset.query_ids).encode()).hexdigest(),
        },
        "embedding": arrays.metadata,
        "timing": "Warm CPU retrieval with cached query embeddings; routing + call + float rerank. Encoding excluded.",
        "float_reference": "Exhaustive inner product over full normalized document embeddings.",
        "binary_reference": "Exact LUT scan in the same selected buckets, at candidate_limit.",
        "seed_rule": "Native search seed = setting seed + query row; repeats reuse that seed.",
        "builds": [],
    }


def _build_cases(routers, config):
    """List the whole experiment before shuffling or measuring anything."""
    route = config["routing"]
    cases = []

    def add_case(method, probes, setting):
        cases.append(
            dict(case_id=f"case-{len(cases):04d}", routing_method=method, probes=probes, **setting)
        )

    for method in route["methods"]:
        settings = _settings(config["search"], route["seed"])
        if method == "ivf":
            settings.append(
                dict(kind="float", node_budget=0, explore_probability=0.0, seed=route["seed"])
            )
        for probes in route["probes"]:
            for setting in settings:
                add_case(method, probes, setting)

        # IVF includes empty centroid lists; sign routing only visits occupied addresses.
        count_field = "possible_buckets" if method == "ivf" else "occupied_buckets"
        all_buckets = routers[method].info[count_field]
        if config["benchmark"].get("complete_routing_endpoint", False) and all_buckets > max(
            route["probes"]
        ):
            for kind in ("scan", "branch"):
                add_case(
                    method,
                    all_buckets,
                    dict(kind=kind, node_budget=0, explore_probability=0.0, seed=route["seed"]),
                )
    return cases


def _build_schedule(cases, query_count, repetitions, seed):
    """Shuffle across routers and probes, while keeping paired query order shared."""
    schedule = []
    for repeat in range(repetitions):
        randomizer = random.Random(seed + repeat)
        case_ids = [case["case_id"] for case in cases]
        query_rows = list(range(query_count))
        randomizer.shuffle(case_ids)
        randomizer.shuffle(query_rows)
        schedule.append(dict(repeat=repeat, case_ids=case_ids, query_rows=query_rows))
    return schedule


def _routing_diagnostics(assignments, buckets, full_reference_rows):
    """Measure what routing kept before a search policy can lose any candidates."""
    labels, counts = np.unique(assignments, return_counts=True)
    population = dict(zip(labels.tolist(), counts.tolist()))
    routed_documents, selected_buckets, coverage = [], [], []
    for selected, reference in zip(buckets, full_reference_rows):
        selected_ids = {int(bucket) for bucket in selected if bucket >= 0}
        reference = reference[reference >= 0]
        hits = sum(int(assignments[row]) in selected_ids for row in reference)
        routed_documents.append(sum(population.get(bucket, 0) for bucket in selected_ids))
        selected_buckets.append(len(selected_ids))
        coverage.append(hits / len(reference) if len(reference) else 0.0)
    return dict(
        routed_documents=np.asarray(routed_documents),
        selected_buckets=np.asarray(selected_buckets),
        routing_float_recall=np.asarray(coverage),
    )


def run_benchmark(dataset, arrays, config, output_dir):
    """Build once, save a complete schedule, then time each request in that order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    route, search, options = config["routing"], config["search"], config["benchmark"]
    query_count = len(dataset.query_ids)
    metadata = _run_metadata(dataset, arrays, config)
    metadata["status"] = "preparing"
    routers, indexes, scopes = {}, {}, {}
    # Flush each completed case so an interrupted run still leaves its measurements.
    measurements = []
    verified_unlimited = 0
    with threadpool_limits(limits=route["threads"]):
        print("Computing the full-float reference...", flush=True)
        reference, _, reference_ms = exact_search(
            arrays.full_documents, arrays.full_queries, search["top_k"], route["threads"]
        )
        metadata["float_reference_p50_ms"] = float(np.median(reference_ms))
        np.save(output_dir / "float_reference_rows.npy", reference)
        reference_quality = []
        for query_number, query_id in enumerate(dataset.query_ids):
            ids = [dataset.corpus_ids[row] for row in reference[query_number] if row >= 0]
            quality = relevance_metrics(ids, dataset.qrels[query_id], search["top_k"])
            reference_quality.append(
                dict(
                    query_id=query_id,
                    **{f"float_reference_{name}": value for name, value in quality.items()},
                )
            )
        _write_csv(output_dir / "reference.csv", reference_quality)

        for method in route["methods"]:
            print(f"Building {method} buckets...", flush=True)
            router = build_router(
                arrays.documents,
                method=method,
                clusters=route["clusters"],
                routing_bits=route["sign_bits"],
                seed=route["seed"],
                threads=route["threads"],
            )
            start = perf_counter()
            index = Index(arrays.codes, router.assignments, arrays.documents.shape[1])
            metadata["builds"].append(
                dict(
                    routing=router.info,
                    routing_build_ms=router.build_ms,
                    native_build_ms=(perf_counter() - start) * 1000,
                    native_storage=index.info(),
                )
            )
            routers[method], indexes[method] = router, index

        cases = _build_cases(routers, config)
        schedule = _build_schedule(cases, query_count, options["repetitions"], route["seed"])
        case_by_id = {case["case_id"]: case for case in cases}
        metadata["cases"] = cases
        metadata["expected_requests"] = len(cases) * query_count * options["repetitions"]
        (output_dir / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
        for case in cases:
            key = (case["routing_method"], case["probes"])
            if key in scopes:
                continue
            router, index = routers[key[0]], indexes[key[0]]
            fixed_buckets, _ = router.select(arrays.queries, key[1])
            binary_reference = index.scan(arrays.queries, fixed_buckets, search["candidate_limit"])
            diagnostics = _routing_diagnostics(router.assignments, fixed_buckets, reference)
            scopes[key] = (fixed_buckets, binary_reference, diagnostics)

        metadata["status"] = "measuring"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"Measuring {len(cases):,} cases × {query_count:,} queries × {options['repetitions']} repeats",
            flush=True,
        )
        with (output_dir / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for repeat_plan in schedule:
                repeat = repeat_plan["repeat"]
                for case_number, case_id in enumerate(repeat_plan["case_ids"], start=1):
                    setting = case_by_id[case_id]
                    method, probes = setting["routing_method"], setting["probes"]
                    router, index = routers[method], indexes[method]
                    fixed_buckets, binary_reference, diagnostics = scopes[(method, probes)]
                    warmup_rows = repeat_plan["query_rows"][: options["warmup_queries"]]
                    for query_number in warmup_rows:
                        _query_run(
                            index,
                            router,
                            arrays.queries[query_number],
                            arrays.full_queries[query_number],
                            arrays.full_documents,
                            fixed_buckets[query_number],
                            probes,
                            search,
                            setting,
                            query_number,
                        )
                    batch = []
                    for query_number in repeat_plan["query_rows"]:
                        query_id = dataset.query_ids[query_number]
                        candidates, final_rows, stats, times = _query_run(
                            index,
                            router,
                            arrays.queries[query_number],
                            arrays.full_queries[query_number],
                            arrays.full_documents,
                            fixed_buckets[query_number],
                            probes,
                            search,
                            setting,
                            query_number,
                        )
                        # Verification and metrics happen after the retrieval clock stops.
                        if setting["kind"] == "branch" and setting["node_budget"] == 0:
                            count = binary_reference["counts"][query_number]
                            expected = binary_reference["rows"][query_number, :count]
                            if not np.array_equal(candidates, expected):
                                raise RuntimeError(
                                    f"Unlimited search disagrees with scan: {case_id}, query {query_id}"
                                )
                            verified_unlimited += 1
                        ids = [dataset.corpus_ids[row] for row in final_rows]
                        quality = relevance_metrics(ids, dataset.qrels[query_id], search["top_k"])
                        returned = int(np.count_nonzero(candidates >= 0))
                        routed = int(diagnostics["routed_documents"][query_number])
                        available = min(search["candidate_limit"], routed)
                        batch.append(
                            {
                                "case_id": case_id,
                                "method": f"{method}/{setting['kind']}",
                                "probes": probes,
                                "node_budget": setting["node_budget"],
                                "explore_probability": setting["explore_probability"],
                                "seed": setting["seed"],
                                "repeat": repeat,
                                "query_row": query_number,
                                "query_id": query_id,
                                **times,
                                "float_recall": neighbor_recall(
                                    final_rows, reference[query_number]
                                ),
                                "binary_recall": neighbor_recall(
                                    candidates, binary_reference["rows"][query_number]
                                )
                                if setting["kind"] != "float"
                                else None,
                                "ndcg": quality["ndcg"],
                                "precision": quality["precision"],
                                "relevance_recall": quality["recall"],
                                "candidates": returned,
                                "nodes": stats["nodes"],
                                "random_nodes": stats["random_nodes"],
                                "bitplane_words": stats["bitplane_words"],
                                "documents_scored": stats["documents_scored"],
                                "stop_reason": stats["stop_reason"],
                                "budget_exhausted": int(stats["stop_reason"] == "budget"),
                                "routed_documents": routed,
                                "selected_buckets": int(
                                    diagnostics["selected_buckets"][query_number]
                                ),
                                "routing_float_recall": float(
                                    diagnostics["routing_float_recall"][query_number]
                                ),
                                "empty_result": int(returned == 0),
                                "fewer_than_k": int(len(final_rows) < search["top_k"]),
                                "candidate_fill_rate": returned / available if available else 0.0,
                            }
                        )
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(batch[0]))
                        writer.writeheader()
                    writer.writerows(batch)
                    stream.flush()
                    # Raw timings are needed for exact p50/p95 after all repetitions.
                    measurements.extend(batch)
                    if case_number % 50 == 0 or case_number == len(cases):
                        print(
                            f"Repeat {repeat + 1}/{options['repetitions']}: {case_number}/{len(cases)} cases saved",
                            flush=True,
                        )
    summary = summarize(measurements)
    _write_csv(output_dir / "summary.csv", summary)
    metadata["status"] = "complete"
    metadata["measured_requests"] = len(measurements)
    metadata["unlimited_candidate_checks"] = verified_unlimited
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, metadata


# --- Plots read the same summary we write to CSV. ---


def plot_results(summary, metadata, output_dir):
    output_dir = Path(output_dir)
    colors = {
        "ivf/scan": "#64748b",
        "ivf/float": "#111827",
        "ivf/branch": "#007f73",
        "sign/scan": "#b45309",
        "sign/branch": "#b54371",
    }
    # Draw straight to files. No GUI window or global pyplot state is needed.
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for method in sorted({row["method"] for row in summary}):
        rows = [row for row in summary if row["method"] == method]
        for axis, metric in zip(axes, ("float_recall", "ndcg")):
            for probability in sorted({row["explore_probability"] for row in rows}):
                points = [row for row in rows if row["explore_probability"] == probability]
                suffix = f", explore={probability:g}" if method.endswith("branch") else ""
                axis.scatter(
                    [row["retrieval_p50_ms"] for row in points],
                    [row[metric] for row in points],
                    s=44,
                    alpha=0.8,
                    marker="o" if probability == 0 else "^",
                    color=colors.get(method),
                    label=method + suffix,
                )
    top_k = metadata["config"]["search"]["top_k"]
    axes[0].set_ylabel(f"Recall@{top_k} against full-float neighbors")
    axes[1].set_ylabel(f"nDCG@{top_k} against relevance labels")
    for axis in axes:
        axis.set_xlabel("Median retrieval time (ms; query embedding cached)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="best")
    data = metadata["dataset"]
    subset = " · subset run" if data["is_subset"] else ""
    figure.suptitle(f"{data['documents']:,} documents · {data['queries']:,} queries{subset}")
    figure.savefig(output_dir / "tradeoffs.png", dpi=180)
    figure.savefig(output_dir / "tradeoffs.svg")
    figure.clear()

    best = sorted(summary, key=lambda row: (-row["float_recall"], row["retrieval_p50_ms"]))
    lines = [
        "# Search results",
        "",
        f"{data['documents']:,} documents, {data['queries']:,} queries.",
        "",
        "This is a subset run."
        if data["is_subset"]
        else "The full selected dataset split was used.",
        "Query embeddings are cached. Times include routing, the search call and float reranking.",
        "",
        "| Method | Probes | Node budget | Explore | Seed | p50 ms | Float recall | nDCG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['method']} | {row['probes']} | {row['node_budget']} | "
            f"{row['explore_probability']:g} | {row['seed']} | {row['retrieval_p50_ms']:.3f} | "
            f"{row['float_recall']:.3f} | {row['ndcg']:.3f} |"
        )
    lines += [
        "",
        "![Latency and quality](tradeoffs.png)",
        "",
        "For branch search, node budget 0 means unlimited. Scans do not use a node budget.",
        "`queries.csv` has every measured request. `summary.csv` keeps random seeds separate.",
        "`metadata.json` records data/model identity, settings, hardware and build/storage counts.",
        "Binary recall uses the exact scan in the same buckets; float recall uses the whole corpus.",
        "Faiss float-IVF includes routing in its core time and uses float scores to choose candidates.",
        "It is an outside baseline, not the controlled binary-scan comparison.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_trace(dataset, arrays, config, output_dir, query_id=None):
    """A separate untimed search keeps trace collection out of the latency results."""
    query_number = dataset.query_ids.index(query_id) if query_id else 0
    route = config["routing"]
    search = config["search"]
    with threadpool_limits(limits=route["threads"]):
        router = build_router(
            arrays.documents,
            route["methods"][0],
            route["clusters"],
            route["sign_bits"],
            route["seed"],
            route["threads"],
        )
        index = Index(arrays.codes, router.assignments, arrays.documents.shape[1])
        query = arrays.queries[query_number : query_number + 1]
        buckets, _ = router.select(query, route["probes"][0])
        result = index.search(
            query,
            buckets,
            search["candidate_limit"],
            node_budget=search["node_budgets"][-1],
            leaf_size=search["leaf_size"],
            explore_probability=search["explore_probabilities"][-1],
            seed=search["seeds"][0] + query_number,
            trace=True,
        )
        final_rows = rerank(
            result["rows"][0],
            arrays.full_documents,
            arrays.full_queries[query_number],
            search["top_k"],
        )
    trace = {
        "query_id": dataset.query_ids[query_number],
        "query": dataset.queries[query_number],
        "routing": router.info,
        "buckets": buckets[0].tolist(),
        "search": {
            "node_budget": search["node_budgets"][-1],
            "leaf_size": search["leaf_size"],
            "explore_probability": search["explore_probabilities"][-1],
            "seed": search["seeds"][0] + query_number,
            "candidate_limit": search["candidate_limit"],
        },
        "dimension_note": "Dimensions are zero-based; -1 means no split at this node.",
        "stats": result["stats"][0],
        "result_ids": [dataset.corpus_ids[row] for row in final_rows],
    }
    (Path(output_dir) / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    steps = trace["stats"]["trace"]
    lines = [
        "# One query through the branches",
        "",
        trace["query"],
        "",
        f"Query ID: `{trace['query_id']}`. Routing: `{route['methods'][0]}`. "
        f"Explore probability: {trace['search']['explore_probability']:g}. "
        f"Seed: {trace['search']['seed']}.",
        "",
        f"Stopped because: `{trace['stats']['stop_reason']}`. "
        f"Popped {trace['stats']['nodes']} nodes; {trace['stats']['random_nodes']} choices were random.",
        "",
        "This trace was collected separately from timing. Bit numbers below start at 1.",
        "",
        "| Step | Action | Bit | Documents | Penalty | Best possible score | Random choice |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for step_number, step in enumerate(steps[:64], start=1):
        bit = str(step["dimension"] + 1) if step["dimension"] >= 0 else "—"
        lines.append(
            f"| {step_number} | {step['event']} | {bit} | {step['documents']} | "
            f"{step['penalty']:.4f} | {step['upper_bound']:.4f} | "
            f"{'yes' if step['random'] else 'no'} |"
        )
    if len(steps) > 64:
        lines += [
            "",
            f"Showing 64 of {len(steps)} steps. `trace.json` contains the full trace.",
        ]
    lines += ["", "Final document IDs: " + ", ".join(trace["result_ids"]), ""]
    (Path(output_dir) / "trace.md").write_text("\n".join(lines), encoding="utf-8")
