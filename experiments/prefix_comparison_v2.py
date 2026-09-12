"""Combine saved study summaries for paper tables, without rerunning searches.

Usage:
  python -m experiments.prefix_comparison_v2 OUTPUT SOURCE_STUDY [SOURCE_STUDY ...]
  python -m experiments.prefix_comparison_v2 OUTPUT SOURCE_STUDY [...] --plots

Sources sharing a pool must have identical input identities and array hashes.
Probability studies are excluded because they compare options within Method B.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.prefix_figures_v2 import METHODS, draw_study, save_csv


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def qualified(row):
    return (row.get("complete") and row.get("qualified") and row.get("recall") is not None
            and row.get("p50_ms") is not None and row["p50_ms"] > 0)


def combine_settings(sources):
    """Keep source IDs and attach storage and opened-row counts to each setting."""
    merged = {"main": [], "repeat": []}
    router_bytes = {None: 0}
    for source in sources:
        for phase in merged:
            for original in source[phase]:
                router = original["router"]
                if router not in router_bytes:
                    router_bytes[router] = read_json(Path(router) / "router.json")["routing_payload_bytes"]
                logical = original.get("logical_index_bytes")
                merged[phase].append(dict(original,
                    setting_id=f"{source['name']}::{original['setting_id']}",
                    original_setting_id=original["setting_id"], source_study=source["name"],
                    source_folder=str(source["folder"]),
                    source_ram_budget_bytes=source["config"].get("ram_budget_bytes"),
                    router_stored_fields_bytes=router_bytes[router],
                    stored_fields_with_router_bytes=logical + router_bytes[router] if logical is not None else None))
    scans = defaultdict(list)
    for row in merged["main"]:
        if row["method"] == "scan" and qualified(row) and row.get("mean_documents_scored") is not None:
            scans[(row["router"], row["probes"])].append(row)
    for phase, rows in merged.items():
        rows.sort(key=lambda row: row["setting_id"])
        for row in rows:
            matching = scans[(row["router"], row["probes"])]
            scan = min(matching, key=lambda other: (other["top_k"] != row["top_k"],
                        other["p50_ms"], other["setting_id"])) if matching else None
            # A scan scores every opened row. Reuse that count only for this same router/probe choice.
            row["mean_opened_rows"] = scan["mean_documents_scored"] if scan else None
            row["opened_rows_scan_setting_id"] = scan["setting_id"] if scan else None
    return merged


def comparison_table(main, repeats, top_ks, targets):
    """Select on main results; attach only the repeat of that exact source setting."""
    repeated_by_id = {row["setting_id"]: row for row in repeats}
    table, missing = [], []
    for k in top_ks:
        for target in targets:
            for method in METHODS:
                eligible = [row for row in main if qualified(row) and row["method"] == method
                            and row["top_k"] == k and row["recall"] >= target]
                if not eligible:
                    continue
                chosen = min(eligible, key=lambda row: (row["p50_ms"], row["setting_id"]))
                repeat = repeated_by_id.get(chosen["setting_id"])
                status = "missing" if repeat is None else "qualified" if qualified(repeat) else "unqualified"
                table.append(dict(chosen, target=target, repeat_status=status,
                    repeat_setting_id=repeat["setting_id"] if repeat else None,
                    repeat_p50_ms=repeat.get("p50_ms") if repeat else None,
                    repeat_p95_ms=repeat.get("p95_ms") if repeat else None,
                    repeat_recall=repeat.get("recall") if repeat else None,
                    repeat_meets_target=repeat["recall"] >= target if repeat and repeat.get("recall") is not None else None,
                    repeat_process_p50_min_ms=repeat.get("process_p50_min_ms") if repeat else None,
                    repeat_process_p50_max_ms=repeat.get("process_p50_max_ms") if repeat else None))
                if status != "qualified":
                    missing.append(dict(setting_id=chosen["setting_id"], original_setting_id=chosen["original_setting_id"],
                                        source_folder=chosen["source_folder"], method=method, top_k=k,
                                        target=target, repeat_status=status))
    return table, missing


def prepare(source_folders, output, *, plots=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new derived-comparison directory.")
    groups, excluded, names = defaultdict(list), [], set()
    for source_folder in source_folders:
        folder = Path(source_folder).resolve()
        config = read_json(folder / "configuration.json")
        if "probability_source" in config or (folder / "probability-selection.json").exists():
            excluded.append(dict(source=str(folder), reason="Probability comparison within Method B."))
            continue
        if folder.name in names:
            raise ValueError("Source study names must be unique so stamped setting IDs remain distinct.")
        names.add(folder.name)
        manifest = read_json(folder / "inputs-manifest.json")
        pool = Path(manifest["pool"]).name
        if groups[pool]:
            previous = groups[pool][0]["manifest"]
            if (manifest["identity"] != previous["identity"]
                    or manifest["array_hashes"] != previous["array_hashes"]):
                raise ValueError(f"Sources for {pool} must have identical identities and array hashes.")
        repeat_file = folder / "repeat/settings.json"
        files = ["configuration.json", "inputs-manifest.json", "main/settings.json"]
        if repeat_file.exists():
            files.append("repeat/settings.json")
        groups[pool].append(dict(name=folder.name, folder=folder, config=config, manifest=manifest,
            main=read_json(folder / "main/settings.json"), repeat=read_json(repeat_file) if repeat_file.exists() else [],
            file_hashes={name: file_hash(folder / name) for name in files}))

    output.mkdir(parents=True)
    completed = []
    for pool, sources in sorted(groups.items()):
        folder = output / pool
        (folder / "main").mkdir(parents=True)
        (folder / "repeat").mkdir()
        merged = combine_settings(sources)
        top_ks = sorted({k for source in sources for k in source["config"]["top_ks"]})
        targets = sorted({target for source in sources for target in source["config"]["recall_targets"]})
        config = dict(pool=sources[0]["manifest"]["pool"], top_ks=top_ks, recall_targets=targets,
                      scope="Derived tables from saved summaries; no new search measurements.",
                      storage_scope="Stored fields with router = native logical_index_bytes plus router.json routing_payload_bytes. This excludes unused allocation capacity, original float vectors, runtime and query workspace.")
        save_json(folder / "configuration.json", config)
        save_json(folder / "inputs-manifest.json", sources[0]["manifest"])
        save_json(folder / "source-studies.json", [dict(name=source["name"], folder=str(source["folder"]),
                  file_hashes=source["file_hashes"]) for source in sources])
        for phase, rows in merged.items():
            save_json(folder / phase / "settings.json", rows)
        table, missing = comparison_table(merged["main"], merged["repeat"], top_ks, targets)
        save_json(folder / "comparison-table.json", table)
        save_csv(folder / "comparison-table.csv", table)
        save_json(folder / "missing-repeats.json", missing)
        save_csv(folder / "missing-repeats.csv", missing)
        if plots:
            draw_study(folder)
        completed.append(dict(pool=pool, folder=str(folder), sources=len(sources),
                              main_settings=len(merged["main"]), repeated_settings=len(merged["repeat"]),
                              table_rows=len(table), missing_repeat_rows=len(missing)))
    result = dict(groups=completed, excluded_sources=excluded)
    save_json(output / "groups.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--plots", action="store_true", help="Draw the existing study figures from merged summaries.")
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output, plots=args.plots), indent=2), flush=True)


if __name__ == "__main__":
    main()
