"""Check the paper's storage and loop counts against saved native runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.fixed_data_comparison import save_json
from experiments.prefix_geometry_v2 import storage_audit


def check_study(study):
    study = Path(study)
    storage, counts, failures = [], {}, []
    sizes_by_router = {}
    for case in sorted(study.glob("main/cases/*")):
        process_path = case / "process.json"
        result_path = case / "run/result.json"
        if not process_path.exists() or not result_path.exists():
            continue
        process = json.loads(process_path.read_text())
        result = json.loads(result_path.read_text())
        if process["status"] != "complete" or result["status"] != "complete":
            continue
        job = result["job"]
        router = job["router"]
        if router not in sizes_by_router:
            if router is None:
                sizes_by_router[router] = [job["documents"]]
            else:
                assignments = np.load(Path(router) / "assignments.npy", mmap_mode="r")
                _, sizes = np.unique(assignments, return_counts=True)
                sizes_by_router[router] = sizes.tolist()
        sizes = sizes_by_router[router]
        audit = storage_audit(job["method"], result["storage"], sizes)
        storage.append(dict(job_id=job["job_id"], clusters=job["clusters"], **audit))
        if not audit["matches"]:
            failures.append(dict(job_id=job["job_id"], check="stored field bytes"))

        variants = {row["setting_id"]: row for row in job["variants"]}
        # If every centroid has documents, each probe opens one actual cluster.
        # Otherwise we leave that particular count untested here.
        all_clusters_present = len(sizes) == result["memory"]["routing_label_count"]
        with (case / "run/queries.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                variant = variants[row["setting_id"]]
                expected = {}
                if job["method"] == "prefix":
                    levels = variant["start_depth"] - row["final_depth"] + 1
                    expected["prefix_levels"] = levels
                    if all_clusters_present:
                        nonzero_levels = levels - int(row["final_depth"] == 0)
                        expected["prefix_lookups"] = 2 * variant["probes"] * nonzero_levels
                    if row["final_depth"] == 0 and variant["probes"] >= len(sizes):
                        expected["documents_scored"] = job["documents"]
                elif job["method"] == "scan" and variant["probes"] >= len(sizes):
                    expected["documents_scored"] = job["documents"]
                elif job["method"] == "branch" and job["clusters"] == 1:
                    # Every split and every scored leaf walks the same full-width
                    # mask, even after the active document count has shrunk.
                    words = (job["documents"] + 63) // 64
                    expected["all_mask_words"] = row["nodes"] * words
                    row["all_mask_words"] = row["bitplane_words"] + row["leaf_words"]
                for name, value in expected.items():
                    counts[name] = counts.get(name, 0) + 1
                    if row[name] != value:
                        failures.append(dict(job_id=job["job_id"], setting_id=row["setting_id"],
                            query=row["query"], check=name, expected=value, measured=row[name]))
    report = dict(storage=storage, checked_query_counts=counts, failures=failures,
        scope="Exact logical field sizes include cluster padding. Query checks compare "
              "loop counts, not elapsed-time predictions. Only complete processes are checked. "
              "This does not measure every allocation or establish a constant cost per CPU operation.")
    save_json(study / "work-checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path, nargs="+")
    args = parser.parse_args()
    for study in args.studies:
        report = check_study(study)
        print(f"{study.name}: {len(report['storage'])} storage checks, "
              f"{sum(report['checked_query_counts'].values())} query-count checks, "
              f"{len(report['failures'])} mismatches", flush=True)
        if report["failures"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
