"""Predict global exact-prefix work from saved exact scores, then compare records.

  python -m experiments.prefix_exact_prediction_v2 predict BOUNDS_JSON NEW_OUTPUT
  python -m experiments.prefix_exact_prediction_v2 compare PREDICTIONS_JSON STUDY

Real arrays are read and keys sorted only by the explicit predict command. The
native search does not receive these saved scores or predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analysis import write_csv
from experiments.fixed_data_comparison import file_hash, save_json
from experiments.prefix_geometry_v2 import document_keys
from experiments.prefix_study_v2 import read_json, verify_pool

FIELDS = ("final_depth", "documents_scored", "prefix_levels", "prefix_lookups")


def predict_rows(codes, queries, bounds, *, starts=(1, 4, 32), top_ks=(1, 100)):
    dimensions = queries.shape[1]
    width = min(32, dimensions)
    starts = sorted({start for start in starts if 0 <= start <= width})
    sorted_keys = np.sort(document_keys(codes, width)).astype(np.uint64)
    minimums = np.minimum.accumulate(np.abs(queries[:, :width]).astype(np.float64), axis=1)
    query_keys = np.zeros(len(queries), dtype=np.uint32)
    for bit in range(width):
        query_keys = (query_keys << np.uint32(1)) | (queries[:, bit] >= 0).astype(np.uint32)
    records = []
    for bound in bounds:
        if bound["top_k"] not in top_ks:
            continue
        qi, tau, total = bound["query"], bound["kth_score"], bound["query_l1"]
        # Match detail::cannot_improve, including the strict inequality for ties.
        guard = 32 * np.finfo(np.float64).eps * (dimensions + 1) * (total + abs(tau) + 1)
        valid = total - 2 * minimums[qi] + guard < tau
        valid_depths = np.flatnonzero(valid) + 1
        largest = int(valid_depths[-1]) if len(valid_depths) else 0
        for start in starts:
            final = min(start, largest)
            if final:
                shift = width - final
                prefix = int(query_keys[qi]) >> shift
                left = np.searchsorted(sorted_keys, prefix << shift, side="left")
                right = np.searchsorted(sorted_keys, (prefix + 1) << shift, side="left")
                count = int(right - left)
            else:
                count = len(codes)
            levels = start - final + 1
            records.append(dict(query=qi, top_k=bound["top_k"], start_depth=start,
                largest_valid_depth=largest, final_depth=final, documents_scored=count,
                prefix_levels=levels, prefix_lookups=2 * (levels - int(final == 0)),
                query_l1=total, kth_score=tau, rounding_guard=float(guard),
                predicted_stop_reason="bound" if final else "exhausted"))
    return records


def predict(bounds_path, output):
    bounds_path, output = Path(bounds_path), Path(output)
    bounds = read_json(bounds_path)
    pool = Path(bounds["inputs"]["pool"])
    inputs = verify_pool(pool)
    if inputs != bounds["inputs"]:
        raise ValueError("Saved bounds and pool identify different inputs")
    if output.exists():
        raise FileExistsError("Use a new prediction output directory")
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    queries = np.load(pool / "queries.npy", mmap_mode="r")
    rows = predict_rows(codes, queries, bounds["queries"])
    summary = []
    for k, start in sorted({(row["top_k"], row["start_depth"]) for row in rows}):
        own = [row for row in rows if (row["top_k"], row["start_depth"]) == (k, start)]
        item = dict(top_k=k, start_depth=start, queries=len(own),
                    bound_stops=sum(row["final_depth"] > 0 for row in own))
        for name in FIELDS:
            item[f"mean_{name}"] = float(np.mean([row[name] for row in own]))
            item[f"median_{name}"] = float(np.median([row[name] for row in own]))
        summary.append(item)
    report = dict(inputs=inputs, bounds_sha256=file_hash(bounds_path), queries=rows, summary=summary,
        assumptions=["One global cluster containing every document; candidate_target=0 and stop_when_exact=True.",
                     "All cached queries and original sign-coordinate order; complete each depth before checking the bound.",
                     "Saved kth_score is the exact global Kth score; the current heap must be full before native stopping.",
                     "Q and tau describe the same score as the native implementation. Saved bounds use direct float64 sums; native lookup accumulation may differ in the last bits. The guard formula is reproduced and measured comparisons retain any mismatch."],
        derivation="At depth ell, every unseen row mismatches a constrained sign, so its score is at most Q-2*min(abs(q[:ell])). The prefix minimum cannot increase with depth, so valid positive depths form 1..y. At any valid depth every true top-K row lies in the visited prefix, making the heap threshold tau. A smaller current heap threshold cannot prove a stop at an invalid depth. Therefore final_depth=min(start_depth,y), with zero meaning the root. This is an actual-stop prediction under the stated assumptions, not B's optimistic all-mismatch bound.",
        scope="Offline count prediction using saved reference scores; references are not consulted by actual search. No latency prediction or new search run.")
    output.mkdir(parents=True)
    save_json(output / "predictions.json", report)
    write_csv(output / "queries.csv", rows)
    write_csv(output / "summary.csv", summary)
    return report


def compare(predictions_path, study, phase="main"):
    predictions_path, study = Path(predictions_path), Path(study)
    prediction = read_json(predictions_path)
    if read_json(study / "inputs-manifest.json") != prediction["inputs"]:
        raise ValueError("Measured study and predictions identify different inputs")
    expected = {(row["query"], row["top_k"], row["start_depth"]): row
                for row in prediction["queries"]}
    compared, mismatches, coverage_errors, settings, skipped = 0, [], [], [], []
    measured_predictions = set()
    for job in read_json(study / phase / "schedule.json"):
        if job["method"] != "prefix" or job["clusters"] != 1 or job["router"] is not None:
            continue
        variants = {v["setting_id"]: v for v in job["variants"]
                    if v["candidate_target"] == 0 and v.get("stop_when_exact", False)
                    and all((qi, v["top_k"], v["start_depth"]) in expected for qi in job["query_rows"])}
        if not variants:
            continue
        folder = study / phase / "cases" / job["job_id"]
        process = read_json(folder / "process.json") if (folder / "process.json").exists() else {}
        result = read_json(folder / "run/result.json") if (folder / "run/result.json").exists() else {}
        if process.get("status") != "complete" or result.get("status") != "complete":
            skipped.append(dict(job_id=job["job_id"], reason="job_not_complete"))
            continue
        variants = {key: value for key, value in variants.items()
                    if key in result.get("completed_variants", [])}
        observed = {key: [] for key in variants}
        with (folder / "run/queries.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                identifier = row["setting_id"]
                if identifier not in variants:
                    continue
                variant = variants[identifier]
                key = (row["query"], variant["top_k"], variant["start_depth"])
                wanted = expected[key]
                observed[identifier].append((row["query"], row["repetition"]))
                measured_predictions.add(key)
                compared += 1
                differences = {name: dict(predicted=wanted[name], measured=row[name])
                               for name in FIELDS if wanted[name] != row[name]}
                if differences:
                    mismatches.append(dict(job_id=job["job_id"], setting_id=identifier,
                        query=row["query"], repetition=row["repetition"], differences=differences))
        for identifier, variant in variants.items():
            wanted_pairs = {(qi, repeat) for qi in job["query_rows"] for repeat in range(variant["repetitions"])}
            seen = observed[identifier]
            if set(seen) != wanted_pairs or len(seen) != len(wanted_pairs):
                coverage_errors.append(dict(job_id=job["job_id"], setting_id=identifier,
                    expected_records=len(wanted_pairs), observed_records=len(seen)))
            settings.append(dict(job_id=job["job_id"], setting_id=identifier,
                                  top_k=variant["top_k"], start_depth=variant["start_depth"], records=len(seen)))
    report = dict(predictions_sha256=file_hash(predictions_path), study=str(study.resolve()), phase=phase,
        compared_records=compared, measured_prediction_rows=len(measured_predictions),
        unmeasured_prediction_rows=len(expected) - len(measured_predictions), settings=settings,
        mismatches=mismatches, coverage_errors=coverage_errors, skipped_jobs=skipped,
        matches=bool(compared) and not mismatches and not coverage_errors,
        scope="Compare four integer work counts only for completed global exact-stop variants; unmatched predictions are not claimed as verified.")
    save_json(predictions_path.parent / f"comparison-{study.name}-{phase}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "compare"))
    parser.add_argument("path", type=Path)
    parser.add_argument("output_or_study", type=Path)
    parser.add_argument("--phase", default="main")
    args = parser.parse_args()
    if args.stage == "predict":
        result = predict(args.path, args.output_or_study)
        print(result["summary"])
    else:
        result = compare(args.path, args.output_or_study, args.phase)
        print({name: result[name] for name in ("compared_records", "matches", "unmeasured_prediction_rows")})


if __name__ == "__main__":
    main()
