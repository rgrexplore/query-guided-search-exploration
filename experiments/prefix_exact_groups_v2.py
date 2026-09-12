"""Break down saved global-search repeats by exact-code membership.

Usage:
  python -m experiments.prefix_exact_groups_v2 STUDY GEOMETRY_JSON NEW_OUTPUT

Settings come from the study's whole-query selection at recall 1.0, K=1.
Groups come from the earlier geometry calculation, never from timing or final depth.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.prefix_comparison_v2 import file_hash, read_json, save_json
from experiments.prefix_figures_v2 import METHODS, save_csv
from experiments.prefix_study_v2 import WORK_FIELDS


def query_groups(geometry_rows, query_ids):
    """A query is present when at least one document has its complete 32-bit code."""
    rows = [row for row in geometry_rows if row["depth"] == 32 and row["top_k"] == 1]
    expected = set(range(len(query_ids)))
    if (len(set(query_ids)) != len(query_ids) or len(rows) != len(query_ids)
            or {row["query"] for row in rows} != expected):
        raise ValueError("Geometry must partition every unique query ID exactly once.")
    present = sorted(row["query"] for row in rows if row["matched_rows"] > 0)
    absent = sorted(row["query"] for row in rows if row["matched_rows"] == 0)
    if set(present) & set(absent) or set(present) | set(absent) != expected:
        raise ValueError("Present and absent groups must partition every query.")
    return {"all": sorted(expected), "exact-code-present": present, "exact-code-absent": absent}


def summarize_groups(records, groups, query_count):
    """Each method contributes the same queries once in each of three repeat blocks."""
    keys = [(row["method"], row["block"], row["query"]) for row in records]
    expected = {(method, block, query) for method in METHODS for block in range(3)
                for query in range(query_count)}
    if len(keys) != len(expected) or set(keys) != expected or any(row["repetition"] != 0 for row in records):
        raise ValueError("Each method must contain the same queries in all three completed repeat blocks.")
    summary = []
    for method in METHODS:
        for name, query_rows in groups.items():
            members = set(query_rows)
            own = [row for row in records if row["method"] == method and row["query"] in members]
            times = [row["query_ms"] for row in own]
            processes = defaultdict(list)
            for row in own:
                processes[row["block"]].append(row["query_ms"])
            medians = [float(np.median(values)) for values in processes.values()]
            row = dict(method=method, group=name, queries=len(members), fraction=len(members) / query_count,
                       measurements=len(own), mean_query_ms=float(np.mean(times)) if times else None,
                       median_query_ms=float(np.median(times)) if times else None,
                       p95_query_ms=float(np.percentile(times, 95)) if times else None,
                       recall=float(np.mean([item["recall"] for item in own])) if own else None,
                       process_median_min_ms=min(medians) if medians else None,
                       process_median_max_ms=max(medians) if medians else None)
            row.update({f"mean_{field}": float(np.mean([item.get(field, 0) for item in own])) if own else None
                        for field in WORK_FIELDS})
            summary.append(row)
    checks = []
    for method in METHODS:
        rows = {row["group"]: row for row in summary if row["method"] == method}
        # Means combine by query count. Medians and percentiles do not.
        for field in ("mean_query_ms", "recall", *(f"mean_{name}" for name in WORK_FIELDS)):
            combined = sum(row["fraction"] * row[field] for name, row in rows.items()
                           if name != "all" and row["queries"])
            overall = rows["all"][field]
            matches = bool(np.isclose(combined, overall, rtol=1e-12, atol=1e-12))
            checks.append(dict(method=method, field=field, overall=overall, weighted_groups=combined,
                               difference=combined - overall, matches=matches))
    if not all(check["matches"] for check in checks):
        raise ValueError("Weighted group means do not reproduce the overall means.")
    return summary, checks


def prepare(study, geometry_path, output):
    study, geometry_path, output = Path(study).resolve(), Path(geometry_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new conditional-report directory.")
    manifest = read_json(study / "inputs-manifest.json")
    geometry = read_json(geometry_path)
    if any(manifest[key] != geometry["inputs"][key] for key in ("identity", "array_hashes")):
        raise ValueError("The geometry and study must use identical query and document inputs.")
    query_ids = manifest["identity"]["query_ids"]
    if len(query_ids) != 1000 or manifest["identity"]["dimensions"] != 32:
        raise ValueError("This report requires the complete 1,000-query, 32-bit study.")
    groups = query_groups(geometry["queries"], query_ids)
    chosen = [row for row in read_json(study / "selected-settings.json")
              if row["target"] == 1.0 and row["top_k"] == 1]
    if len(chosen) != 3 or {row["method"] for row in chosen} != set(METHODS):
        raise ValueError("The whole-query selection must provide A, B and C at recall 1.0, K=1.")
    if any(row["clusters"] != 1 or row["router"] is not None for row in chosen):
        raise ValueError("This conditional report uses the global, single-cluster study only.")
    repeated = {row["setting_id"]: row for row in read_json(study / "repeat/settings.json")}
    if any(not repeated.get(row["setting_id"], {}).get("qualified")
           or not repeated[row["setting_id"]]["complete"] for row in chosen):
        raise ValueError("All three selected settings need completed, qualified repeats.")
    by_id = {row["setting_id"]: row["method"] for row in chosen}
    source_files = [study / name for name in ("inputs-manifest.json", "selected-settings.json",
                    "repeat/settings.json", "repeat/schedule.json")]
    records, used_cases = [], []
    for job in read_json(study / "repeat/schedule.json"):
        identifiers = {variant["setting_id"] for variant in job["variants"]} & set(by_id)
        if not identifiers:
            continue
        folder = study / "repeat/cases" / job["job_id"]
        process = read_json(folder / "process.json")
        result = read_json(folder / "run/result.json")
        power = result["power_before"]
        if (process["status"] != "complete" or result["status"] != "complete"
                or not power.get("available") or power != result["power_after"]
                or result["memory"]["budget_status"] != "fits_by_lifetime_peak"
                or not identifiers <= set(result["completed_variants"])):
            raise ValueError("Only completed, qualified repeat records may enter this report.")
        if job["query_rows"] != list(range(1000)) or job["pool"] != manifest["pool"]:
            raise ValueError("Repeated settings must preserve every query and the same pool.")
        query_file = folder / "run/queries.jsonl"
        with query_file.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["setting_id"] in identifiers:
                    records.append(dict(row, method=by_id[row["setting_id"]], block=job["block"]))
        used_cases.append(dict(job_id=job["job_id"], block=job["block"], setting_ids=sorted(identifiers)))
        source_files.extend([folder / "process.json", folder / "run/result.json", query_file])
    summary, checks = summarize_groups(records, groups, len(query_ids))
    output.mkdir(parents=True)
    save_json(output / "group-summary.json", summary)
    save_csv(output / "group-summary.csv", summary)
    save_json(output / "groups.json", {name: dict(query_rows=rows, query_ids=[query_ids[row] for row in rows],
              count=len(rows), fraction=len(rows) / len(query_ids)) for name, rows in groups.items()})
    save_json(output / "mean-checks.json", checks)
    save_json(output / "selection-provenance.json", dict(study=str(study), selected_settings=chosen,
        repeated_settings=[repeated[row["setting_id"]] for row in chosen], used_cases=used_cases,
        geometry=str(geometry_path), geometry_sha256=file_hash(geometry_path),
        source_file_hashes={str(path.relative_to(study)): file_hash(path) for path in source_files},
        group_rule="Geometry at depth 32, top_k 1: exact-code-present means matched_rows>0; exact-code-absent means matched_rows=0.",
        scope="Conditional breakdown of the same global settings selected on all 1,000 queries. No setting is retuned for a group. This is not a cluster-optimized subgroup comparison and does not establish an overall Method C advantage.",
        aggregation="Time statistics use all repeated observations. Each query appears once in each of three process blocks. The process median range is the minimum and maximum subgroup median over those blocks. Weighted subgroup means reproduce the overall means; medians and percentiles are not additive."))
    return dict(output=str(output), queries=len(query_ids), measurements=len(records),
                exact_code_present=len(groups["exact-code-present"]),
                exact_code_absent=len(groups["exact-code-absent"]), mean_checks=len(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.study, args.geometry, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
