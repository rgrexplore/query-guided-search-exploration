"""Select fixed-grid settings and report saved scaling measurements."""

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np

import scaling_plots


ROW_FIELDS = {
    "phase",
    "pool_size",
    "method",
    "node_budget",
    "repeat",
    "query_row",
    "query_id",
    "api_ms",
    "native_ms",
    "rows",
    "scores",
    "count",
    "recall",
    "nodes",
    "random_nodes",
    "bitplane_words",
    "documents_scored",
    "stop_reason",
}
STABLE_FIELDS = (
    "rows",
    "scores",
    "count",
    "recall",
    "nodes",
    "random_nodes",
    "bitplane_words",
    "documents_scored",
    "stop_reason",
)
CASE_STATUSES = {
    "complete",
    "timeout",
    "memory_limit",
    "interrupted",
    "worker_error",
    "correctness_failure",
}
CASE_FIELDS = {
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


def choose_case(rows: list[dict], target: float) -> dict | None:
    """Return the fastest tested complete branch setting that reaches ``target``."""
    qualifying = []
    for row in rows:
        latency = row.get("api_p50_ms")
        recall = row.get("development_recall")
        if (
            row.get("method") == "branch"
            and row.get("status") == "complete"
            and isinstance(latency, (int, float))
            and math.isfinite(latency)
            and isinstance(recall, (int, float))
            and recall >= target
        ):
            qualifying.append(row)
    if not qualifying:
        return None
    return min(
        qualifying,
        key=lambda row: (
            row["api_p50_ms"],
            row["node_budget"] == 0,
            row["node_budget"] if row["node_budget"] else math.inf,
        ),
    )


def _read_json(path: Path):
    return json.loads(path.read_text())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    _atomic_text(path, _csv_text(rows))


def _atomic_gzip_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", newline="") as stream:
        stream.write(_csv_text(rows))
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule(run_dir: Path, phase: str, *, required: bool = True) -> list[str]:
    path = run_dir / "schedules" / f"{phase}.json"
    if not path.is_file():
        if required:
            raise ValueError(f"missing scheduled phase {phase}")
        return []
    values = _read_json(path)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path.relative_to(run_dir)} must be a list of case paths")
    return values


def _case_path(run_dir: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("scheduled case paths must be relative")
    result = run_dir / path
    if not result.is_file():
        raise ValueError(f"missing scheduled case {relative}")
    return result


def _query_file(folder: Path) -> Path:
    plain = folder / "queries.jsonl"
    compressed = folder / "queries.jsonl.gz"
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    raise ValueError(f"missing complete query rows in {folder}")


def _query_bytes(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def _query_rows(folder: Path) -> list[dict]:
    path = _query_file(folder)
    try:
        text = _query_bytes(path).decode()
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read complete query rows in {folder}") from error


def _metadata(run_dir: Path) -> dict:
    metadata = _read_json(run_dir / "metadata.json")
    required = {"config", "inputs", "query_splits", "run_fingerprint", "provenance"}
    if not required <= set(metadata):
        raise ValueError("metadata.json is missing scaling run fields")
    query_ids = metadata["inputs"].get("query_ids")
    if not isinstance(query_ids, list) or len(query_ids) != len(set(query_ids)):
        raise ValueError("metadata query IDs must be unique")
    return metadata


def _summary(folder: Path) -> dict:
    path = folder / "summary.json"
    if not path.is_file():
        raise ValueError(f"missing terminal summary for scheduled case {folder}")
    value = _read_json(path)
    required = {
        "status",
        "completed_queries",
        "setup_ms",
        "wall_ms",
        "sampled_peak_rss_bytes",
        "native_peak_rss_bytes",
    }
    if not required <= set(value) or value["status"] not in CASE_STATUSES:
        raise ValueError(f"invalid terminal summary in {folder}")
    return value


def _references(run_dir: Path, metadata: dict) -> dict[int, dict[int, tuple[list, list]]]:
    pools = metadata["config"]["data"]["pool_sizes"]
    expected_rows = set(metadata["query_splits"]["development"])
    expected_rows.update(metadata["query_splits"]["evaluation"])
    query_ids = metadata["inputs"]["query_ids"]
    candidate_limit = metadata["config"]["search"]["candidate_limit"]
    references = {}
    for relative in _schedule(run_dir, "reference"):
        case_path = _case_path(run_dir, relative)
        case = _read_json(case_path)
        folder = case_path.parent
        summary = _summary(folder)
        if summary["status"] != "complete":
            raise ValueError(f"reference case is not complete: {relative}")
        pool = case.get("pool_size")
        if (
            set(case) != CASE_FIELDS
            or case.get("phase") != "reference"
            or case.get("method") != "scan"
            or case.get("node_budget") != 0
            or pool in references
        ):
            raise ValueError(f"invalid reference case {relative}")
        verification_path = folder / "verification.json"
        if not verification_path.is_file() or _read_json(verification_path).get("status") != "pass":
            raise ValueError(f"reference verification did not pass for pool {pool}")
        if summary["completed_queries"] != len(case["query_rows"]):
            raise ValueError(f"reference summary query count is wrong for pool {pool}")
        with np.load(folder / "reference.npz", allow_pickle=False) as saved:
            saved_rows = np.asarray(saved["query_rows"], dtype=np.int64)
            saved_ids = np.asarray(saved["query_ids"])
            candidates = np.asarray(saved["rows"], dtype=np.int64)
            scores = np.asarray(saved["scores"], dtype=np.float64)
            saved_pool = int(saved["pool_size"])
            saved_limit = int(saved["candidate_limit"])
        if (
            saved_pool != pool
            or saved_limit != candidate_limit
            or set(saved_rows.tolist()) != expected_rows
            or len(saved_rows) != len(expected_rows)
            or candidates.shape != (len(saved_rows), candidate_limit)
            or scores.shape != candidates.shape
            or saved_ids.shape != saved_rows.shape
            or not np.isfinite(scores).all()
        ):
            raise ValueError(f"reference inventory is wrong for pool {pool}")
        reference = {}
        for position, query_row in enumerate(saved_rows.tolist()):
            if str(saved_ids[position]) != query_ids[query_row]:
                raise ValueError(f"reference query ID is wrong for row {query_row}")
            reference[query_row] = (
                candidates[position].tolist(),
                scores[position].tolist(),
            )
        references[pool] = reference
    if set(references) != set(pools):
        raise ValueError("reference schedule does not cover every configured pool")
    return references


def _validate_row(
    row: dict,
    case: dict,
    query_ids: list[str],
    reference: dict[int, tuple[list, list]],
) -> dict:
    if set(row) != ROW_FIELDS:
        raise ValueError("complete query row fields do not match the worker record")
    for field in ("phase", "pool_size", "method", "node_budget", "repeat"):
        if row[field] != case[field]:
            raise ValueError(f"query row {field} does not match its case")
    query_row = row["query_row"]
    if query_row not in case["query_rows"] or row["query_id"] != query_ids[query_row]:
        raise ValueError("query row or query ID does not match the saved run")
    if (
        row["count"] != len(row["rows"])
        or row["count"] != len(row["scores"])
        or not math.isfinite(row["api_ms"])
        or not math.isfinite(row["native_ms"])
        or not all(math.isfinite(value) for value in row["scores"])
    ):
        raise ValueError("query result counts, scores, or timing are invalid")
    oracle_rows, oracle_scores = reference[query_row]
    observed_recall = len(set(row["rows"]).intersection(oracle_rows)) / len(oracle_rows)
    if not math.isclose(row["recall"], observed_recall, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"saved recall is wrong for query row {query_row}")
    if (case["method"] == "scan" or case["node_budget"] == 0) and (
        row["rows"] != oracle_rows or row["scores"] != oracle_scores
    ):
        raise ValueError("exact scan or unlimited branch does not match its pool reference")
    return row


def _expected_development(metadata: dict) -> set[tuple]:
    config = metadata["config"]
    settings = [("scan", 0)] + [
        ("branch", budget) for budget in config["tuning"]["node_budgets"]
    ]
    return {
        (pool, method, budget, repeat)
        for pool in config["data"]["pool_sizes"]
        for method, budget in settings
        for repeat in range(config["measurement"]["repetitions"])
    }


def _expected_evaluation(metadata: dict, selection: dict) -> set[tuple]:
    repetitions = metadata["config"]["measurement"]["repetitions"]
    choices = defaultdict(set)
    for row in selection["targets"]:
        if row["status"] == "selected":
            choices[row["pool_size"]].add(row["node_budget"])
    expected = set()
    for pool in metadata["config"]["data"]["pool_sizes"]:
        settings = {("scan", 0), ("branch", 0)}
        settings.update(("branch", budget) for budget in choices[pool])
        for method, budget in settings:
            for repeat in range(repetitions):
                expected.add((pool, method, budget, repeat))
    return expected


def _expected_pilot(metadata: dict) -> set[tuple]:
    pools = metadata["config"]["data"]["pool_sizes"]
    pilot_pools = {pools[0], pools[-1]}
    if 100_000 in pools:
        pilot_pools.add(100_000)
    return {
        (pool, method, budget, 0)
        for pool in pilot_pools
        for method, budget in (("scan", 0), ("branch", 512), ("branch", 0))
    }


def _status_for(summaries: list[dict]) -> str:
    for summary in summaries:
        if summary["status"] != "complete":
            return summary["status"]
    return "complete"


def _summarize_phase(
    run_dir: Path,
    metadata: dict,
    references: dict,
    phase: str,
    expected_inventory: set[tuple] | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if phase == "pilot":
        expected_queries = metadata["query_splits"]["development"][:5]
    else:
        expected_queries = metadata["query_splits"][phase]
    query_ids = metadata["inputs"]["query_ids"]
    grouped = defaultdict(list)
    failures = []
    seen_inventory = set()
    for relative in _schedule(run_dir, phase, required=phase != "pilot"):
        case_path = _case_path(run_dir, relative)
        case = _read_json(case_path)
        folder = case_path.parent
        if (
            set(case) != CASE_FIELDS
            or case.get("phase") != phase
            or set(case.get("query_rows", [])) != set(expected_queries)
        ):
            raise ValueError(f"scheduled {phase} case has the wrong query inventory")
        identity = (
            case.get("pool_size"),
            case.get("method"),
            case.get("node_budget"),
            case.get("repeat"),
        )
        if identity in seen_inventory:
            raise ValueError(f"duplicate scheduled {phase} case {identity}")
        seen_inventory.add(identity)
        summary = _summary(folder)
        item = {"case": case, "summary": summary, "relative": relative, "rows": None}
        if summary["status"] == "complete":
            rows = _query_rows(folder)
            by_query = {}
            for row in rows:
                query_row = row.get("query_row")
                if query_row in by_query:
                    raise ValueError(f"duplicate query row {query_row} in {relative}")
                by_query[query_row] = _validate_row(
                    row, case, query_ids, references[case["pool_size"]]
                )
            if set(by_query) != set(expected_queries):
                raise ValueError(f"complete case has a missing query in {relative}")
            if summary["completed_queries"] != len(expected_queries):
                raise ValueError(f"complete summary query count is wrong in {relative}")
            item["rows"] = by_query
        else:
            failures.append(
                {
                    "phase": phase,
                    "pool_size": case["pool_size"],
                    "method": case["method"],
                    "node_budget": case["node_budget"],
                    "repeat": case["repeat"],
                    "status": summary["status"],
                    "completed_queries": summary["completed_queries"],
                    "error": summary.get("error", ""),
                    "case_path": relative,
                }
            )
        grouped[(case["pool_size"], case["method"], case["node_budget"])].append(item)
    if expected_inventory is not None and seen_inventory != expected_inventory:
        missing = expected_inventory - seen_inventory
        extra = seen_inventory - expected_inventory
        raise ValueError(
            f"{phase} schedule inventory is incomplete; "
            f"missing {sorted(missing)}, extra {sorted(extra)}"
        )

    summaries = []
    for (pool, method, budget), items in sorted(grouped.items()):
        items.sort(key=lambda item: item["case"]["repeat"])
        terminal = [item["summary"] for item in items]
        status = _status_for(terminal)
        row = {
            "phase": phase,
            "pool_size": pool,
            "method": method,
            "node_budget": budget,
            "status": status,
            "case_statuses": "|".join(summary["status"] for summary in terminal),
            "completed_repetitions": sum(summary["status"] == "complete" for summary in terminal),
            "queries": len(expected_queries),
            "requests": None,
            "development_recall": None,
            "mean_recall": None,
            "p10_recall": None,
            "empty_rate": None,
            "api_p50_ms": None,
            "api_p95_ms": None,
            "native_p50_ms": None,
            "native_p95_ms": None,
            "mean_documents_scored": None,
            "mean_fraction_fully_scored": None,
            "mean_nodes": None,
            "mean_random_nodes": None,
            "mean_bitplane_words": None,
            "sampled_peak_rss_bytes": None,
            "native_peak_rss_bytes": None,
            "measurement_key": f"{phase}:n{pool}-{method}-b{budget}",
        }
        complete = [item for item in items if item["rows"] is not None]
        if complete:
            first = complete[0]["rows"]
            for item in complete[1:]:
                for query_row in expected_queries:
                    for field in STABLE_FIELDS:
                        if item["rows"][query_row][field] != first[query_row][field]:
                            raise ValueError(
                                f"quality or work changed across repeats for query {query_row}"
                            )
        if status == "complete":
            required_repetitions = (
                1
                if phase == "pilot"
                else metadata["config"]["measurement"]["repetitions"]
            )
            if len(items) != required_repetitions:
                raise ValueError(f"complete {phase} setting does not have all repetitions")
            quality = {query_row: first[query_row]["recall"] for query_row in expected_queries}
            api_times = {
                query_row: [item["rows"][query_row]["api_ms"] for item in items]
                for query_row in expected_queries
            }
            native_times = [
                item["rows"][query_row]["native_ms"]
                for item in items
                for query_row in expected_queries
            ]
            api_flat = [value for values in api_times.values() for value in values]
            quality_values = list(quality.values())
            row.update(
                requests=len(api_flat),
                development_recall=(
                    float(np.mean(quality_values)) if phase == "development" else None
                ),
                mean_recall=float(np.mean(quality_values)),
                p10_recall=float(np.percentile(quality_values, 10)),
                empty_rate=float(np.mean([first[q]["count"] == 0 for q in expected_queries])),
                api_p50_ms=float(np.percentile(api_flat, 50)),
                api_p95_ms=float(np.percentile(api_flat, 95)),
                native_p50_ms=float(np.percentile(native_times, 50)),
                native_p95_ms=float(np.percentile(native_times, 95)),
                mean_documents_scored=float(
                    np.mean([first[q]["documents_scored"] for q in expected_queries])
                ),
                mean_fraction_fully_scored=float(
                    np.mean([first[q]["documents_scored"] / pool for q in expected_queries])
                ),
                mean_nodes=float(np.mean([first[q]["nodes"] for q in expected_queries])),
                mean_random_nodes=float(
                    np.mean([first[q]["random_nodes"] for q in expected_queries])
                ),
                mean_bitplane_words=float(
                    np.mean([first[q]["bitplane_words"] for q in expected_queries])
                ),
                sampled_peak_rss_bytes=max(
                    item["summary"]["sampled_peak_rss_bytes"] for item in items
                ),
                native_peak_rss_bytes=max(
                    item["summary"]["native_peak_rss_bytes"] for item in items
                ),
                _api_times=api_times,
                _quality=quality,
                _query_rows=first,
            )
        summaries.append(row)
    return summaries, failures, [item for items in grouped.values() for item in items]


def summarize_development(run_dir: Path) -> list[dict]:
    """Summarize only stable, complete development settings from the fixed grid."""
    run_dir = Path(run_dir)
    metadata = _metadata(run_dir)
    references = _references(run_dir, metadata)
    summaries, _, _ = _summarize_phase(
        run_dir,
        metadata,
        references,
        "development",
        _expected_development(metadata),
    )
    return summaries


def _canonical_hash(files: dict[str, tuple[Path, bool]]) -> dict[str, str]:
    result = {}
    for relative, (path, decompress) in sorted(files.items()):
        value = _query_bytes(path) if decompress else path.read_bytes()
        result[relative] = _sha256_bytes(value)
    return result


def _development_hashes(run_dir: Path) -> dict[str, str]:
    files = {}
    for phase in ("reference", "development"):
        schedule_path = run_dir / "schedules" / f"{phase}.json"
        schedule = _schedule(run_dir, phase)
        files[str(schedule_path.relative_to(run_dir))] = (schedule_path, False)
        for relative in schedule:
            case_path = _case_path(run_dir, relative)
            folder = case_path.parent
            files[str(case_path.relative_to(run_dir))] = (case_path, False)
            summary_path = folder / "summary.json"
            files[str(summary_path.relative_to(run_dir))] = (summary_path, False)
            if phase == "development" and _summary(folder)["status"] == "complete":
                query_path = _query_file(folder)
                key = str((folder / "queries.jsonl").relative_to(run_dir))
                files[key] = (query_path, True)
            elif phase == "development":
                plain = folder / "queries.jsonl"
                compressed = folder / "queries.jsonl.gz"
                if plain.is_file() or compressed.is_file():
                    query_path = plain if plain.is_file() else compressed
                    key = str((folder / "queries.jsonl").relative_to(run_dir))
                    files[key] = (query_path, True)
            if phase == "reference":
                reference_path = folder / "reference.npz"
                files[str(reference_path.relative_to(run_dir))] = (reference_path, False)
                verification_path = folder / "verification.json"
                if verification_path.is_file():
                    files[str(verification_path.relative_to(run_dir))] = (
                        verification_path,
                        False,
                    )
    return _canonical_hash(files)


def freeze_selection(run_dir: Path) -> dict:
    """Choose each pool's fastest qualifying tested budget and write it once."""
    run_dir = Path(run_dir)
    metadata = _metadata(run_dir)
    development_files = _development_hashes(run_dir)
    path = run_dir / "selections.json"
    if path.is_file():
        saved = _read_json(path)
        if (
            saved.get("run_fingerprint") != metadata["run_fingerprint"]
            or saved.get("development_files") != development_files
        ):
            raise ValueError("frozen selection does not match the run or development files")
        return saved

    summaries = summarize_development(run_dir)
    targets = []
    for pool in metadata["config"]["data"]["pool_sizes"]:
        pool_rows = [row for row in summaries if row["pool_size"] == pool]
        for target in metadata["config"]["tuning"]["targets"]:
            chosen = choose_case(pool_rows, target)
            targets.append(
                {
                    "pool_size": pool,
                    "target": target,
                    "status": "selected" if chosen else "no_feasible_case",
                    "node_budget": chosen["node_budget"] if chosen else None,
                    "development_recall": chosen["development_recall"] if chosen else None,
                    "development_p50_ms": chosen["api_p50_ms"] if chosen else None,
                }
            )
    selection = {
        "run_fingerprint": metadata["run_fingerprint"],
        "development_files": development_files,
        "analysis_source_sha256": _sha256_file(Path(__file__)),
        "targets": targets,
    }
    _atomic_text(path, json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return selection


def paired_query_bootstrap(
    scan_times: dict[int, list[float]],
    branch_times: dict[int, list[float]],
    quality: dict[int, float],
    *,
    samples: int = 1000,
    seed: int = 45,
) -> dict:
    """Resample query IDs while retaining every timing repeat for each selected ID."""
    query_rows = sorted(scan_times)
    if set(query_rows) != set(branch_times) or set(query_rows) != set(quality):
        raise ValueError("paired bootstrap inputs must contain the same query IDs")
    scan = np.asarray([scan_times[row] for row in query_rows], dtype=np.float64)
    branch = np.asarray([branch_times[row] for row in query_rows], dtype=np.float64)
    quality_values = np.asarray([quality[row] for row in query_rows], dtype=np.float64)
    random = np.random.default_rng(seed)
    draws = random.integers(0, len(query_rows), size=(samples, len(query_rows)))
    speedups = np.empty(samples, dtype=np.float64)
    quality_draws = np.empty(samples, dtype=np.float64)
    for index, draw in enumerate(draws):
        speedups[index] = np.median(scan[draw].reshape(-1)) / np.median(
            branch[draw].reshape(-1)
        )
        quality_draws[index] = np.mean(quality_values[draw])
    speedup_low, speedup_high = np.percentile(speedups, [2.5, 97.5])
    quality_low, quality_high = np.percentile(quality_draws, [2.5, 97.5])
    return {
        "speedup": float(np.median(scan) / np.median(branch)),
        "speedup_ci_low": float(speedup_low),
        "speedup_ci_high": float(speedup_high),
        "quality_mean": float(np.mean(quality_values)),
        "quality_ci_low": float(quality_low),
        "quality_ci_high": float(quality_high),
    }


def _quality_interval(
    quality: dict[int, float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(list(quality.values()), dtype=np.float64)
    random = np.random.default_rng(seed)
    draws = random.integers(0, len(values), size=(samples, len(values)))
    interval = np.percentile(np.mean(values[draws], axis=1), [2.5, 97.5])
    return float(interval[0]), float(interval[1])


def _public_summary(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _target_label(target: float) -> str:
    return f"target{round(target * 100)}"


def _evaluation_labels(selection: dict) -> dict[tuple[int, str, int], list[str]]:
    labels = defaultdict(list)
    for selected in selection["targets"]:
        if selected["status"] == "selected":
            labels[(selected["pool_size"], "branch", selected["node_budget"])].append(
                _target_label(selected["target"])
            )
    return labels


def _target_points(selection: dict, evaluation: list[dict], config: dict) -> list[dict]:
    lookup = {(row["pool_size"], row["method"], row["node_budget"]): row for row in evaluation}
    points = []
    for selected in selection["targets"]:
        pool = selected["pool_size"]
        budget = selected["node_budget"]
        point = {
            **selected,
            "selection_status": selected["status"],
            "evaluation_status": "unavailable",
            "measurement_key": "",
            "evaluation_recall": None,
            "evaluation_p10_recall": None,
            "evaluation_empty_rate": None,
            "api_p50_ms": None,
            "api_p95_ms": None,
            "requests": None,
            "achieved": False,
            "speedup": None,
            "speedup_ci_low": None,
            "speedup_ci_high": None,
            "quality_ci_low": None,
            "quality_ci_high": None,
        }
        if selected["status"] == "selected":
            measured = lookup[(pool, "branch", budget)]
            point["evaluation_status"] = measured["status"]
            point["measurement_key"] = measured["measurement_key"]
            if measured["status"] == "complete":
                scan = lookup[(pool, "scan", 0)]
                point.update(
                    evaluation_recall=measured["mean_recall"],
                    evaluation_p10_recall=measured["p10_recall"],
                    evaluation_empty_rate=measured["empty_rate"],
                    api_p50_ms=measured["api_p50_ms"],
                    api_p95_ms=measured["api_p95_ms"],
                    requests=measured["requests"],
                    achieved=measured["mean_recall"] >= selected["target"],
                )
                samples = config["measurement"]["bootstrap_samples"]
                seed = config["measurement"]["bootstrap_seed"]
                if scan["status"] == "complete":
                    interval = paired_query_bootstrap(
                        scan["_api_times"],
                        measured["_api_times"],
                        measured["_quality"],
                        samples=samples,
                        seed=seed,
                    )
                    point.update(
                        speedup=interval["speedup"],
                        speedup_ci_low=interval["speedup_ci_low"],
                        speedup_ci_high=interval["speedup_ci_high"],
                        quality_ci_low=interval["quality_ci_low"],
                        quality_ci_high=interval["quality_ci_high"],
                    )
                else:
                    low, high = _quality_interval(
                        measured["_quality"], samples=samples, seed=seed
                    )
                    point.update(quality_ci_low=low, quality_ci_high=high)
        points.append(point)
    return points


def _query_summary(evaluation: list[dict], selection: dict) -> list[dict]:
    selected_labels = _evaluation_labels(selection)
    rows = []
    for setting in evaluation:
        if setting["status"] != "complete":
            continue
        key = (setting["pool_size"], setting["method"], setting["node_budget"])
        labels = selected_labels[key]
        if setting["method"] == "scan":
            labels = ["scan"]
        elif setting["node_budget"] == 0:
            labels = labels + ["unlimited"]
        label_text = "|".join(labels)
        for query_row, values in setting["_api_times"].items():
            observed = setting["_query_rows"][query_row]
            rows.append(
                {
                    "pool_size": setting["pool_size"],
                    "method": setting["method"],
                    "node_budget": setting["node_budget"],
                    "measurement_key": setting["measurement_key"],
                    "labels": label_text,
                    "query_row": query_row,
                    "query_id": observed["query_id"],
                    "recall": observed["recall"],
                    "empty": observed["count"] == 0,
                    "api_p50_ms": float(np.percentile(values, 50)),
                    "api_p95_ms": float(np.percentile(values, 95)),
                    "api_ms_repeats": "|".join(f"{value:.12g}" for value in values),
                    "documents_scored": observed["documents_scored"],
                    "nodes": observed["nodes"],
                    "random_nodes": observed["random_nodes"],
                    "bitplane_words": observed["bitplane_words"],
                    "stop_reason": observed["stop_reason"],
                }
            )
    return rows


def _input_hashes(run_dir: Path) -> dict[str, str]:
    files = {
        "metadata.json": (run_dir / "metadata.json", False),
        "selections.json": (run_dir / "selections.json", False),
    }
    for phase in ("reference", "pilot", "development", "evaluation"):
        schedule_path = run_dir / "schedules" / f"{phase}.json"
        if not schedule_path.is_file():
            continue
        files[str(schedule_path.relative_to(run_dir))] = (schedule_path, False)
        for relative in _schedule(run_dir, phase):
            case_path = _case_path(run_dir, relative)
            folder = case_path.parent
            files[str(case_path.relative_to(run_dir))] = (case_path, False)
            summary_path = folder / "summary.json"
            files[str(summary_path.relative_to(run_dir))] = (summary_path, False)
            summary = _summary(folder)
            if summary["status"] == "complete" and phase != "reference":
                query_path = _query_file(folder)
                key = str((folder / "queries.jsonl").relative_to(run_dir))
                files[key] = (query_path, True)
            if phase == "reference":
                reference_path = folder / "reference.npz"
                files[str(reference_path.relative_to(run_dir))] = (reference_path, False)
    return _canonical_hash(files)


def analyze_run(run_dir: Path) -> Path:
    """Validate frozen saved measurements and write the fixed scaling report."""
    run_dir = Path(run_dir)
    selection_path = run_dir / "selections.json"
    if not selection_path.is_file():
        raise ValueError("selections.json must be frozen before reporting")
    metadata = _metadata(run_dir)
    selection = _read_json(selection_path)
    if (
        selection.get("run_fingerprint") != metadata["run_fingerprint"]
        or selection.get("development_files") != _development_hashes(run_dir)
    ):
        raise ValueError("frozen selection does not match the run or development files")
    references = _references(run_dir, metadata)
    development, development_failures, _ = _summarize_phase(
        run_dir, metadata, references, "development", _expected_development(metadata)
    )
    evaluation, evaluation_failures, _ = _summarize_phase(
        run_dir,
        metadata,
        references,
        "evaluation",
        _expected_evaluation(metadata, selection),
    )
    pilot, pilot_failures, _ = _summarize_phase(
        run_dir, metadata, references, "pilot", _expected_pilot(metadata)
    )
    failures = development_failures + evaluation_failures + pilot_failures
    points = _target_points(selection, evaluation, metadata["config"])
    query_rows = _query_summary(evaluation, selection)
    directory = run_dir / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_csv(
        directory / "settings.csv",
        [_public_summary(row) for row in pilot + development + evaluation],
    )
    _atomic_csv(directory / "target-points.csv", points)
    _atomic_csv(directory / "failures.csv", failures)
    _atomic_gzip_csv(directory / "query-summary.csv.gz", query_rows)
    pools = metadata["config"]["data"]["pool_sizes"]
    scaling_plots.write_plots(
        directory,
        pools,
        evaluation,
        points,
        development,
        metadata["config"]["tuning"]["node_budgets"],
    )
    _atomic_text(
        directory / "report.md",
        scaling_plots.report_text(metadata, points, evaluation, failures),
    )
    analysis_metadata = {
        "run_fingerprint": metadata["run_fingerprint"],
        "analysis_source_sha256": {
            "scaling_analysis.py": _sha256_file(Path(__file__)),
            "scaling_plots.py": _sha256_file(Path(scaling_plots.__file__)),
        },
        "input_file_sha256": _input_hashes(run_dir),
        "bootstrap": {
            "samples": metadata["config"]["measurement"]["bootstrap_samples"],
            "seed": metadata["config"]["measurement"]["bootstrap_seed"],
        },
    }
    _atomic_text(
        directory / "metadata.json",
        json.dumps(analysis_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return directory / "report.md"
