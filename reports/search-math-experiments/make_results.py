"""Build figures from the completed comparison with unchanged benchmark inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ["scan", "branch", "keys"]
LETTERS = dict(zip(METHODS, "ABC"))
LABELS = ["A · Cluster and scan", "B · Bitplane branching", "C · Backward walk"]
COLORS = ["#286493", "#C36620", "#7850A4"]
ROUTES = {"none": "No routing", "ivf": "IVF"}


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def number(value):
    if value is None:
        return "---"
    return f"{value:,.0f}" if value >= 1000 or value == round(value) else f"{value:,.2f}"


def time_label(value):
    if value is None:
        return "---"
    return f"{value:.4f}" if value < 1 else f"{value:.2f}"


def write_table(path, columns, headings, rows):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def draw_recall_time(rows, target):
    figure, axis = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    for method, label, color, marker in zip(METHODS, LABELS, COLORS, ["o", "s", "^"], strict=True):
        own = sorted([row for row in rows if row["method"] == method], key=lambda row: row["target"])
        times = [row["p50_ms"] if row["met_target"] and row["qualified"] else np.nan for row in own]
        axis.plot([row["target"] for row in own], times, color=color, marker=marker,
                  markersize=6, linewidth=1.8, label=label)
    axis.set_xticks(sorted({row["target"] for row in rows}))
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set(xlabel="Required recall", ylabel="Median retrieval time (ms)", ylim=(0, None))
    axis.grid(alpha=.2)
    axis.legend(frameon=False, loc="upper left", fontsize=10)
    figure.savefig(target)
    plt.close(figure)


def cluster_choices(settings, target=.99):
    selected = []
    for clusters in (1, 256, 4096, 8192):
        for method in METHODS:
            eligible = [row for row in settings if row["clusters"] == clusters and row["method"] == method
                        and row["qualified"] and row["recall"] is not None and row["recall"] >= target]
            if eligible:
                selected.append(min(eligible, key=lambda row: (row["p50_ms"], row["setting_id"])))
    return selected


def draw_cluster_time(rows, target):
    clusters = [1, 256, 4096, 8192]
    positions = np.arange(len(clusters))
    figure, axis = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    for method_index, (method, color, label) in enumerate(zip(METHODS, COLORS, LABELS, strict=True)):
        own = {row["clusters"]: row for row in rows if row["method"] == method}
        values = [own[count]["p50_ms"] if count in own else np.nan for count in clusters]
        bars = axis.bar(positions + (method_index - 1) * .24, values, width=.23, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                axis.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                              xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    axis.set_xticks(positions, [f"{count:,}" for count in clusters])
    axis.set(xlabel="Number of clusters", ylabel="Median retrieval time (ms)", ylim=(0, None))
    axis.margins(y=.18)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, 1.18))
    figure.savefig(target)
    plt.close(figure)


def verify_input_files(study):
    manifest_path = study / "inputs-manifest.json"
    manifest = read_json(manifest_path)
    hashes = {str(manifest_path): file_hash(manifest_path)}
    pool = Path(manifest["pool"])
    for name, expected in manifest["array_hashes"].items():
        actual = file_hash(pool / name)
        if actual != expected:
            raise ValueError(f"benchmark array changed: {name}")
        hashes[str(pool / name)] = actual
    for router, files in manifest["router_hashes"].items():
        for name, expected in files.items():
            actual = file_hash(Path(router) / name)
            if actual != expected:
                raise ValueError(f"router input changed: {router}/{name}")
            hashes[str(Path(router) / name)] = actual
    if manifest["query_rows"] != list(range(200)):
        raise ValueError("the report requires all 200 unchanged query rows")
    return manifest, hashes


def selected_cluster_sizes(case, queries, documents):
    """Count rows using saved cluster sizes and the original query ranking."""
    if case["router"] is None:
        return [np.array([documents], dtype=np.int64) for _ in queries], np.array([documents]), 0
    folder = Path(case["router"])
    metadata = read_json(folder / "router.json")
    sizes = np.array(metadata["cluster_sizes"], dtype=np.int64)
    centroids = np.load(folder / "centroids.npy")
    labels = np.load(folder / "labels.npy")
    quantizer = faiss.IndexFlatIP(centroids.shape[1])
    quantizer.add(centroids)
    faiss.omp_set_num_threads(1)
    opened = []
    for query in queries:
        # Original query and centroids. No document scores or measured search
        # counters are used to count the selected rows.
        _, positions = quantizer.search(query[None, :], min(case["probes"], len(labels)))
        opened.append(sizes[labels[positions[0]]])
    return opened, sizes, int(centroids.nbytes + labels.nbytes)


def reference_penalties(codes, queries, reference):
    dimensions = np.arange(queries.shape[1])
    last_rows = np.asarray(codes[reference[:, -1]])
    signs = (last_rows[:, dimensions // 64] >> (dimensions % 64).astype(np.uint64)) & np.uint64(1)
    return np.sum((signs != (queries >= 0)) * np.abs(queries).astype(np.float64), axis=1)


def work_checks(case, selected, opened, penalties, queries):
    scored = float(np.mean([rows.sum() for rows in opened]))
    records = []

    def add(field, calculated, source):
        measured = selected[field]
        records.append(dict(field=field, calculated=calculated, measured=measured,
                            matches=math.isclose(calculated, measured, rel_tol=0, abs_tol=1e-9), source=source))

    if case["method"] == "scan":
        add("mean_scored", scored, "Sum of saved sizes of clusters selected by each original query.")
    elif case["method"] == "branch":
        nodes_fit = all(np.max(rows) <= case["leaf_size"] for rows in opened)
        budget_sufficient = case["node_budget"] == 0 or all(len(rows) <= case["node_budget"] for rows in opened)
        if nodes_fit and budget_sufficient:
            add("mean_scored", scored, "Every selected cluster fits in a leaf and the node budget covers all clusters.")
            add("mean_split_words", 0.0, "The initial groups already meet the leaf stopping condition.")
            add("mean_leaf_words", float(np.mean([np.ceil(rows / 64).sum() for rows in opened])),
                "Sum of padded bitmap widths of the selected clusters.")
    else:
        start, width = case["key_offset"], case["key_bits"]
        maximum_key_penalty = np.sum(np.abs(queries[:, start:start + width]).astype(np.float64), axis=1)
        # Fewer clusters cannot improve on the global K-th score. If even the
        # global penalty exceeds every local-key penalty, none can be skipped.
        if case["candidate_target"] == case["key_limit"] == 0 and np.all(maximum_key_penalty < penalties):
            add("mean_scored", scored, "All local-key penalties are below the global reference K-th penalty.")
            add("mean_key_lookups", float(np.mean([len(rows) for rows in opened])) * 2 ** width,
                "Every key pattern in every selected cluster, including empty lookups.")
    return records


def make_results(study):
    summary_path = study / "summary.json"
    summary = read_json(summary_path)
    config = read_json(study / "configuration.json")
    if not summary["inputs_unchanged"]:
        raise ValueError("the experiment did not confirm unchanged inputs")
    manifest, hashes = verify_input_files(study)
    pool = Path(manifest["pool"])
    queries = np.load(pool / "queries.npy")
    reference = np.load(pool / "reference.npy")
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    penalties = reference_penalties(codes, queries, reference)
    rows = sorted(summary["targets"], key=lambda row: (row["target"], METHODS.index(row["method"])))
    parameter_summary_path = study / "sweep" / "summary.json"
    parameter_summary = read_json(parameter_summary_path)
    extra_summary_path = study / "extra" / "summary.json"
    extra_summary = read_json(extra_summary_path) if extra_summary_path.exists() else dict(settings=[], failures=[])
    cluster_rows = cluster_choices(parameter_summary["settings"] + extra_summary["settings"])
    details = {row["setting_id"]: row for row in summary["settings"]}
    selected99 = [row for row in rows if row["target"] == .99]
    if len(selected99) != 3:
        raise ValueError("Three method choices at 99% are required; missing choices must be explained first.")
    schedule_path = study / "repeats" / "schedule.json"
    schedule = read_json(schedule_path)
    for case in schedule:
        if Path(case["pool"]).resolve() != pool.resolve() or case["query_rows"] != manifest["query_rows"]:
            raise ValueError("a repeated setting used different inputs")
    for path in (summary_path, parameter_summary_path, study / "configuration.json", schedule_path, Path(__file__)):
        hashes[str(path)] = file_hash(path)
    if extra_summary_path.exists():
        for path in (extra_summary_path, study / "extra" / "schedule.json", study / "extra" / "key-bound-diagnostic.json"):
            hashes[str(path)] = file_hash(path)

    figures, evidence = HERE / "figures", HERE / "evidence"
    figures.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    draw_recall_time(rows, figures / "recall-time.pdf")
    draw_cluster_time(cluster_rows, figures / "cluster-time.pdf")

    results_table = [[f"{100 * row['target']:.0f}\\%", LETTERS[row["method"]],
                      f"{100 * row['recall']:.2f}" if row["recall"] is not None else "---",
                      time_label(row["p50_ms"])] for row in rows]
    settings_table, work_table, memory_table, checks = [], [], [], []
    for row in selected99:
        detail = details[row["setting_id"]]
        case = detail["case"]
        letter = LETTERS[row["method"]]
        settings_table.append([letter, ROUTES[row["router"]], number(row["clusters"]), number(row["probes"]),
                               number(row["leaf_size"]) if row["method"] == "branch" else "---",
                               number(row["node_budget"]) if row["method"] == "branch" else
                               (number(case["key_limit"]) if row["method"] == "keys" else "---"),
                               row["key_bits"] if row["method"] == "keys" else "---",
                               f"{row['lifetime_peak_bytes'] / 1e6:.1f}" if row["lifetime_peak_bytes"] is not None else "---"])
        work_table.append([letter, number(row["mean_scored"]), number(row["mean_split_words"]),
                           number(row["mean_leaf_words"]), number(row["mean_key_lookups"])])
        raw_paths = [study / "repeats" / "cases" / f"{index:04d}" / "run" / "result.json"
                     for index, saved in enumerate(schedule) if saved["setting_id"] == row["setting_id"]]
        raw = [read_json(path) for path in raw_paths if path.exists()]
        for path in raw_paths:
            if path.exists():
                hashes[str(path)] = file_hash(path)
                observation_path = path.parent / "queries.jsonl"
                hashes[str(observation_path)] = file_hash(observation_path)
        if not raw:
            checks.append(dict(method=row["method"], setting_id=row["setting_id"], status="no completed process"))
            continue
        opened, sizes, router_bytes = selected_cluster_sizes(case, queries, config["documents"])
        calculated_work = work_checks(case, row, opened, penalties, queries)
        storage = raw[0]["storage"]
        code_bytes = config["documents"] * math.ceil(config["dimensions"] / 64) * 8
        id_bytes = config["documents"] * 8
        plane_bytes = config["dimensions"] * 8 * int(np.ceil(sizes / 64).sum()) if row["method"] == "branch" else 0
        occupied = storage["occupied_keys"] if row["method"] == "keys" else 0
        directory_bytes = occupied * 20
        native_calculated = code_bytes + id_bytes + plane_bytes + directory_bytes
        measured_router_bytes = raw[0]["memory"]["routing_payload_bytes"]
        calculated = native_calculated + router_bytes
        measured = storage["logical_bytes"] + measured_router_bytes
        memory_table.append([letter, f"{calculated / 1e6:.3f}", f"{measured / 1e6:.3f}",
                             f"{row['lifetime_peak_bytes'] / 1e6:.1f}"])
        checks.append(dict(method=row["method"], setting_id=row["setting_id"], calculated_work=calculated_work,
                           work_scope="Counts not covered by an input-derived condition remain measured counts only.",
                           memory=dict(code_bytes=code_bytes, row_id_bytes=id_bytes, plane_bytes=plane_bytes,
                                       occupied_keys_measured_input=occupied, directory_bytes=directory_bytes,
                                       calculated_router_bytes=router_bytes, measured_router_bytes=measured_router_bytes,
                                       native_calculated_bytes=native_calculated, native_measured_bytes=storage["logical_bytes"],
                                       total_calculated_bytes=calculated, total_measured_bytes=measured,
                                       matches=calculated == measured,
                                       scope="Logical index data plus routing arrays. Hash allocation and runtime overhead belong in the separate measured process peak.")))

    write_table(evidence / "fixed-results-table.tex", "rlrr", ["Required recall", "Method", "Actual recall (\\%)", "Median (ms)"], results_table)
    write_table(evidence / "fixed-settings-table.tex", "llrrrrrr", ["Method", "Route", "$C$", "$P$", "$L$", "Budget", "$h$", "RAM (MB)"], settings_table)
    write_table(evidence / "fixed-work-table.tex", "lrrrr", ["Method", "Rows scored", "Split words", "Leaf words", "Key lookups"], work_table)
    write_table(evidence / "fixed-memory-table.tex", "lrrr", ["Method", "Calculated (MB)", "Index data (MB)", "Peak RAM (MB)"], memory_table)
    result = dict(setup=config, inputs_manifest=manifest, inputs_unchanged=True,
                  repeated_results=rows, cluster_comparison=cluster_rows, checks=checks,
                  repeat_details=summary["settings"], source_hashes=hashes,
                  parameter_comparison_failures=parameter_summary["failures"],
                  extension_failures=extra_summary["failures"], repeated_process_failures=summary["failures"],
                  scope=summary["scope"],
                  recall_chart_note="Selected settings measured in three separate processes on the same 200 queries. Missing targets or unqualified conditions are not plotted as successful results.",
                  cluster_chart_note="Fastest eligible setting per method and cluster count in the parameter comparison: one process and two repetitions on the same 200 queries. These bars are not the three-process measurements.",
                  timing_note="Measured milliseconds only; no timing predictor or fitted coefficients are used.",
                  memory_note="MB means 1,000,000 bytes. Process peak includes construction, library memory and temporary arrays.")
    (evidence / "fixed-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not all(check.get("memory", {}).get("matches", False) for check in checks):
        raise AssertionError("An index-data formula did not match; inspect evidence/fixed-results.json.")
    if not all(work["matches"] for check in checks for work in check["calculated_work"]):
        raise AssertionError("An input-derived work count did not match; inspect evidence/fixed-results.json.")
    print(f"Wrote {len(rows)} repeated target results, {len(cluster_rows)} cluster comparisons and unchanged-input evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/fixed-data-comparison-2026-09-12")
    make_results(parser.parse_args().study.resolve())
