"""Run one fixed-grid scaling case in a fresh process."""

import json
import math
import resource
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Callable

import bitplane_index
import numpy as np
from threadpoolctl import threadpool_limits


class CorrectnessError(RuntimeError):
    """The measured native result disagrees with its independent reference."""


def _strict_json(path: Path, value: dict) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _append_row(stream, value: dict) -> None:
    json.dump(value, stream, allow_nan=False, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes. Linux reports KiB.
    return int(peak if sys.platform == "darwin" else peak * 1_024)


def _timed_native_call(
    index,
    method: str,
    query: np.ndarray,
    selected_buckets: np.ndarray,
    *,
    candidate_limit: int,
    node_budget: int,
    leaf_size: int,
    explore_probability: float,
    seed: int,
    clock: Callable[[], float] = perf_counter,
) -> tuple[dict, float]:
    """Measure from immediately before the native call until its object is returned."""
    started = clock()
    if method == "scan":
        result = index.scan(query, selected_buckets, candidate_limit)
    else:
        result = index.search(
            query,
            selected_buckets,
            candidate_limit,
            node_budget=node_budget,
            leaf_size=leaf_size,
            explore_probability=explore_probability,
            seed=seed,
        )
    return result, (clock() - started) * 1_000


def _decoded_score(code: np.ndarray, query: np.ndarray, dimensions: int) -> float:
    terms = []
    for dimension in range(dimensions):
        word = int(code[dimension // 64])
        sign = 1.0 if word >> (dimension % 64) & 1 else -1.0
        terms.append(float(query[dimension]) * sign)
    return math.fsum(terms)


def verify_native_sample(
    codes: np.ndarray,
    queries: np.ndarray,
    dimensions: int,
    candidate_limit: int = 100,
) -> dict:
    """Compare native scan scores with independently decoded document signs."""
    document_count = min(1_000, len(codes))
    query_count = min(5, len(queries))
    if document_count == 0 or query_count == 0:
        raise CorrectnessError("independent score verification needs documents and queries")

    sample_codes = np.ascontiguousarray(codes[:document_count], dtype=np.uint64)
    sample_queries = np.ascontiguousarray(queries[:query_count], dtype=np.float32)
    assignments = np.zeros(document_count, dtype=np.int64)
    selected_buckets = np.zeros((query_count, 1), dtype=np.int64)
    limit = min(candidate_limit, document_count)
    index = bitplane_index.Index(sample_codes, assignments, dimensions)
    native = index.scan(sample_queries, selected_buckets, limit)
    near_boundary = 0

    for query_number, query in enumerate(sample_queries):
        count = int(native["counts"][query_number])
        if count != limit:
            raise CorrectnessError("independent score verification returned too few rows")
        returned_rows = [int(row) for row in native["rows"][query_number, :count]]
        independent = [
            _decoded_score(sample_codes[row], query, dimensions)
            for row in range(document_count)
        ]
        for position, row in enumerate(returned_rows):
            expected = independent[row]
            observed = float(native["scores"][query_number, position])
            tolerance = 1e-10 * (1.0 + abs(expected))
            if abs(observed - expected) > tolerance:
                raise CorrectnessError(
                    f"independent score mismatch for query {query_number}, document {row}"
                )

        cutoff = independent[returned_rows[-1]]
        returned = set(returned_rows)
        for row, score in enumerate(independent):
            if row in returned:
                continue
            tolerance = 1e-10 * (1.0 + abs(score))
            difference = score - cutoff
            if difference > tolerance:
                raise CorrectnessError(
                    f"independent score found omitted document {row} above the cutoff"
                )
            if abs(difference) <= tolerance:
                near_boundary += 1

    return {
        "status": "pass",
        "checked_documents": document_count,
        "checked_queries": query_count,
        "dimensions": dimensions,
        "candidate_limit": limit,
        "near_boundary_ambiguities": near_boundary,
    }


def _load_array(description: dict, expected_dtype: np.dtype, name: str) -> np.ndarray:
    path = Path(description["path"])
    if not path.is_absolute():
        raise ValueError(f"inputs.arrays.{name}.path must be absolute")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    declared_shape = tuple(description["shape"])
    if array.shape != declared_shape:
        raise ValueError(f"{name} shape does not match metadata")
    if array.dtype != expected_dtype or description["dtype"] != expected_dtype.name:
        raise ValueError(f"{name} dtype does not match metadata")
    return array


def _load_inputs(case: dict) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    run_path = Path(case["run_path"])
    if not run_path.is_absolute():
        raise ValueError("run_path must be absolute")
    metadata = json.loads(run_path.read_text())
    config = metadata["config"]
    inputs = metadata["inputs"]
    dimensions = config["search"]["dimensions"]
    if inputs["dimensions"] != dimensions:
        raise ValueError("input dimensions do not match the run configuration")
    codes = _load_array(inputs["arrays"]["codes"], np.dtype(np.uint64), "codes")
    queries = _load_array(inputs["arrays"]["queries"], np.dtype(np.float32), "queries")
    if codes.ndim != 2 or codes.shape[1] != (dimensions + 63) // 64:
        raise ValueError("codes must have shape [documents, dimensions / 64]")
    if queries.ndim != 2 or queries.shape[1] != dimensions:
        raise ValueError("queries must have shape [queries, dimensions]")
    query_ids = inputs["query_ids"]
    if len(query_ids) != len(queries) or len(set(query_ids)) != len(query_ids):
        raise ValueError("inputs.query_ids must identify every query exactly once")
    return config, inputs, codes, queries


def _validate_case(case: dict, code_count: int, query_count: int) -> None:
    required = {
        "phase",
        "pool_size",
        "method",
        "node_budget",
        "repeat",
        "query_rows",
        "run_path",
        "reference_path",
        "output_dir",
    }
    if set(case) != required:
        raise ValueError(
            f"case fields must be {sorted(required)}; found {sorted(case)}"
        )
    if case["phase"] not in {"reference", "pilot", "development", "evaluation"}:
        raise ValueError("phase is not supported")
    if case["method"] not in {"scan", "branch"}:
        raise ValueError("method must be scan or branch")
    if isinstance(case["pool_size"], bool) or not isinstance(case["pool_size"], int):
        raise ValueError("pool_size must be an integer")
    if not 0 < case["pool_size"] <= code_count:
        raise ValueError("pool_size is outside the code array")
    if isinstance(case["node_budget"], bool) or not isinstance(case["node_budget"], int):
        raise ValueError("node_budget must be an integer")
    if case["node_budget"] < 0 or (case["method"] == "scan" and case["node_budget"] != 0):
        raise ValueError("node_budget is invalid for the method")
    if (
        isinstance(case["repeat"], bool)
        or not isinstance(case["repeat"], int)
        or case["repeat"] < 0
    ):
        raise ValueError("repeat must be a nonnegative integer")
    rows = case["query_rows"]
    if not isinstance(rows, list) or not rows or len(rows) != len(set(rows)):
        raise ValueError("query_rows must be a nonempty list of unique rows")
    if any(
        isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < query_count
        for row in rows
    ):
        raise ValueError("query_rows contain a row outside the query array")
    if not Path(case["output_dir"]).is_absolute():
        raise ValueError("output_dir must be absolute")
    if case["phase"] == "reference":
        if case["method"] != "scan" or case["node_budget"] != 0:
            raise ValueError("reference cases must use scan with node_budget zero")
        if case["reference_path"] is not None:
            raise ValueError("reference_path must be null for a reference case")
    elif case["reference_path"] is None or not Path(case["reference_path"]).is_absolute():
        raise ValueError("reference_path must be absolute outside the reference phase")


def _load_reference(
    path: Path,
    pool_size: int,
    candidate_limit: int,
    query_rows: list[int],
    query_ids: list[str],
) -> dict[int, tuple[np.ndarray, str]]:
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["pool_size"]) != pool_size:
            raise CorrectnessError("reference pool_size does not match the case")
        if int(saved["candidate_limit"]) != candidate_limit:
            raise CorrectnessError("reference candidate_limit does not match the case")
        saved_rows = np.asarray(saved["query_rows"], dtype=np.int64)
        saved_ids = np.asarray(saved["query_ids"])
        candidate_rows = np.asarray(saved["rows"], dtype=np.int64)
    if len(saved_rows) != len(set(saved_rows.tolist())):
        raise CorrectnessError("each reference query row must appear exactly once")
    if saved_ids.shape != saved_rows.shape or candidate_rows.shape != (
        len(saved_rows), candidate_limit
    ):
        raise CorrectnessError("reference row mapping has the wrong shape")
    positions = {int(row): position for position, row in enumerate(saved_rows)}
    selected = {}
    for query_row in query_rows:
        if query_row not in positions:
            raise CorrectnessError(f"reference is missing query row {query_row}")
        position = positions[query_row]
        expected_id = query_ids[query_row]
        if str(saved_ids[position]) != expected_id:
            raise CorrectnessError(f"reference query ID does not match row {query_row}")
        selected[query_row] = (candidate_rows[position], expected_id)
    return selected


def _result_row(
    case: dict,
    query_row: int,
    query_id: str,
    result: dict,
    api_ms: float,
    reference_rows: np.ndarray | None,
) -> dict:
    count = int(result["counts"][0])
    if not 0 <= count <= result["rows"].shape[1]:
        raise CorrectnessError("native result count is outside its returned arrays")
    rows = [int(value) for value in result["rows"][0, :count]]
    scores = [float(value) for value in result["scores"][0, :count]]
    if any(row < 0 for row in rows) or not all(math.isfinite(score) for score in scores):
        raise CorrectnessError("native valid prefix contains padding or a nonfinite score")
    stats = result["stats"][0]

    recall = None
    if reference_rows is not None:
        candidate_limit = len(reference_rows)
        recall = len(set(rows).intersection(int(row) for row in reference_rows)) / candidate_limit
        if (case["method"] == "scan" or case["node_budget"] == 0) and not np.array_equal(
            np.asarray(rows, dtype=np.int64), reference_rows
        ):
            raise CorrectnessError(
                f"{case['method']} query row {query_row} does not exactly match the reference"
            )

    return {
        "phase": case["phase"],
        "pool_size": case["pool_size"],
        "method": case["method"],
        "node_budget": case["node_budget"],
        "repeat": case["repeat"],
        "query_row": query_row,
        "query_id": query_id,
        "api_ms": api_ms,
        "native_ms": float(stats["elapsed_ms"]),
        "rows": rows,
        "scores": scores,
        "count": count,
        "recall": recall,
        "nodes": int(stats["nodes"]),
        "random_nodes": int(stats["random_nodes"]),
        "bitplane_words": int(stats["bitplane_words"]),
        "documents_scored": int(stats["documents_scored"]),
        "stop_reason": str(stats["stop_reason"]),
    }


def run_case(case: dict) -> dict:
    """Run one native scan or branch case and write its evidence files."""
    wall_started = perf_counter()
    setup_started = perf_counter()
    config, inputs, codes, queries = _load_inputs(case)
    _validate_case(case, len(codes), len(queries))
    search = config["search"]
    dimensions = search["dimensions"]
    candidate_limit = search["candidate_limit"]
    if candidate_limit > case["pool_size"]:
        raise ValueError("candidate_limit cannot exceed pool_size")

    output_dir = Path(case["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ids = inputs["query_ids"]
    query_views = {
        row: np.ascontiguousarray(queries[row : row + 1], dtype=np.float32)
        for row in case["query_rows"]
    }
    selected_buckets = np.array([[0]], dtype=np.int64)

    with threadpool_limits(limits=1):
        if case["phase"] == "reference":
            verification = verify_native_sample(
                codes[: case["pool_size"]],
                queries,
                dimensions,
                candidate_limit=candidate_limit,
            )
            _strict_json(output_dir / "verification.json", verification)
        index = bitplane_index.Index(
            codes[: case["pool_size"]],
            np.zeros(case["pool_size"], dtype=np.int64),
            dimensions,
        )
        references = None
        if case["phase"] != "reference":
            references = _load_reference(
                Path(case["reference_path"]),
                case["pool_size"],
                candidate_limit,
                case["query_rows"],
                query_ids,
            )
        setup_ms = (perf_counter() - setup_started) * 1_000

        warmup_count = min(
            config["measurement"]["warmup_queries"], len(case["query_rows"])
        )
        for query_row in case["query_rows"][:warmup_count]:
            _timed_native_call(
                index,
                case["method"],
                query_views[query_row],
                selected_buckets,
                candidate_limit=candidate_limit,
                node_budget=case["node_budget"],
                leaf_size=search["leaf_size"],
                explore_probability=search["explore_probability"],
                seed=search["seed"] + query_row,
            )

        reference_result_rows = []
        reference_result_scores = []
        queries_path = output_dir / "queries.jsonl"
        with queries_path.open("w") as stream:
            for query_row in case["query_rows"]:
                result, api_ms = _timed_native_call(
                    index,
                    case["method"],
                    query_views[query_row],
                    selected_buckets,
                    candidate_limit=candidate_limit,
                    node_budget=case["node_budget"],
                    leaf_size=search["leaf_size"],
                    explore_probability=search["explore_probability"],
                    seed=search["seed"] + query_row,
                )
                expected = None if references is None else references[query_row][0]
                row = _result_row(
                    case,
                    query_row,
                    query_ids[query_row],
                    result,
                    api_ms,
                    expected,
                )
                _append_row(stream, row)
                if case["phase"] == "reference":
                    reference_result_rows.append(row["rows"])
                    reference_result_scores.append(row["scores"])

    if case["phase"] == "reference":
        np.savez(
            output_dir / "reference.npz",
            query_rows=np.asarray(case["query_rows"], dtype=np.int64),
            query_ids=np.asarray(
                [query_ids[row] for row in case["query_rows"]], dtype=np.str_
            ),
            rows=np.asarray(reference_result_rows, dtype=np.int64),
            scores=np.asarray(reference_result_scores, dtype=np.float64),
            pool_size=np.asarray(case["pool_size"], dtype=np.int64),
            candidate_limit=np.asarray(candidate_limit, dtype=np.int64),
        )

    worker = {
        "setup_ms": setup_ms,
        "wall_ms": (perf_counter() - wall_started) * 1_000,
        "native_peak_rss_bytes": _peak_rss_bytes(),
    }
    _strict_json(output_dir / "worker.json", worker)
    return {**worker, "query_count": len(case["query_rows"])}


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python scaling_worker.py CASE_JSON_PATH", file=sys.stderr)
        return 2
    case = None
    try:
        case = json.loads(Path(argv[1]).read_text())
        run_case(case)
        return 0
    except Exception as error:
        if isinstance(case, dict) and isinstance(case.get("output_dir"), str):
            output_dir = Path(case["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            status = (
                "correctness_failure"
                if isinstance(error, CorrectnessError)
                else "worker_error"
            )
            _strict_json(
                output_dir / "error.json",
                {"status": status, "message": str(error)},
            )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
