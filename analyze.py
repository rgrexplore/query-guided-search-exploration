"""Check a recorded run, compare matching queries, and draw the measured tradeoffs."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


QUALITY = ("float_recall", "binary_recall", "ndcg", "precision", "relevance_recall")
RATES = QUALITY + (
    "budget_exhausted",
    "routing_float_recall",
    "empty_result",
    "fewer_than_k",
    "candidate_fill_rate",
)
COUNTS = (
    "candidates",
    "nodes",
    "random_nodes",
    "documents_scored",
    "bitplane_words",
    "routed_documents",
    "selected_buckets",
)
STABLE = RATES + COUNTS
TIMES = ("retrieval_ms", "core_ms", "native_ms", "routing_ms", "rerank_ms")
POLICY_KEYS = ("routing_method", "probes", "kind", "node_budget", "explore_probability")
ROUTERS = ("ivf", "sign")
DISPLAY_PROBES = (1, 4, 16, 64)
COLORS = {0.0: "#087e8b", 0.1: "#b45c22"}


@dataclass
class RecordedRun:
    metadata: dict
    cases: list
    query_ids: list
    stable: np.ndarray
    times: np.ndarray
    reference: dict


def _number(value, name, optional=False):
    if value in (None, "") and optional:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def validate_planned_cases(metadata):
    """Check the declared case list against the sweep, not just against its CSV rows."""
    config = metadata["config"]
    route = config.get("routing", {})
    search = config.get("search", {})
    if not {"methods", "probes", "seed"} <= route.keys():
        return
    if not {"node_budgets", "explore_probabilities", "seeds"} <= search.keys():
        return
    if "builds" not in metadata:
        return
    builds = {build["routing"]["method"]: build["routing"] for build in metadata["builds"]}
    expected = set()
    default_seed = route["seed"]
    for method in route["methods"]:
        if method not in builds:
            raise ValueError(f"Missing build metadata for planned router {method}")
        for probes in route["probes"]:
            expected.add((method, probes, "scan", 0, 0.0, default_seed))
            if method == "ivf":
                expected.add((method, probes, "float", 0, 0.0, default_seed))
            for budget in search["node_budgets"]:
                for probability in search["explore_probabilities"]:
                    seeds = search["seeds"] if probability else [default_seed]
                    expected.update(
                        (method, probes, "branch", budget, probability, seed) for seed in seeds
                    )
        count_field = "possible_buckets" if method == "ivf" else "occupied_buckets"
        all_buckets = builds[method][count_field]
        if config["benchmark"].get("complete_routing_endpoint", False) and all_buckets > max(
            route["probes"]
        ):
            expected.update(
                (method, all_buckets, kind, 0, 0.0, default_seed) for kind in ("scan", "branch")
            )
    identity_fields = POLICY_KEYS + ("seed",)
    declared = [tuple(case[key] for key in identity_fields) for case in metadata["cases"]]
    if len(declared) != len(set(declared)):
        raise ValueError("Duplicate planned case identity")
    missing, extra = expected - set(declared), set(declared) - expected
    if missing or extra:
        raise ValueError(
            f"Declared cases differ from the planned sweep: missing {sorted(missing)}, extra {sorted(extra)}"
        )


def load_run(run_dir):
    """Stream requests into fixed arrays instead of keeping a million dictionaries."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    cases = metadata["cases"]
    case_lookup = {case["case_id"]: index for index, case in enumerate(cases)}
    if not cases or len(case_lookup) != len(cases):
        raise ValueError("Case IDs must be present and unique")
    validate_planned_cases(metadata)
    query_count = metadata["dataset"]["queries"]
    repeats = metadata["config"]["benchmark"]["repetitions"]
    if query_count <= 0 or repeats <= 0:
        raise ValueError("A run needs queries and timing repetitions")

    # The reference file establishes one shared query order for every comparison.
    with (run_dir / "reference.csv").open(newline="") as stream:
        references = list(csv.DictReader(stream))
    query_ids = [row["query_id"] for row in references]
    if len(query_ids) != query_count or len(set(query_ids)) != query_count:
        raise ValueError("reference.csv must contain each query exactly once")
    reference = {}
    for metric in ("ndcg", "precision", "recall"):
        name = "float_reference_" + metric
        reference[metric] = np.array([_number(row[name], name) for row in references])
        if np.any(reference[metric] > 1):
            raise ValueError(f"{name} must be between zero and one")

    shape = (len(cases), query_count)
    stable = np.full(shape + (len(STABLE),), np.nan)
    times = np.full(shape + (repeats, len(TIMES)), np.nan)
    seen = np.zeros(shape + (repeats,), dtype=bool)
    stops = np.full(shape, "", dtype="U16")
    positions = {name: index for index, name in enumerate(STABLE)}

    with (run_dir / "queries.csv").open(newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                case_index = case_lookup[row["case_id"]]
                query = int(row["query_row"])
                repeat = int(row["repeat"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"Unknown case or invalid row identity on line {line}") from error
            if not 0 <= query < query_count or not 0 <= repeat < repeats:
                raise ValueError(f"Query or repeat outside the declared run on line {line}")
            if row["query_id"] != query_ids[query]:
                raise ValueError(f"Query ID and query row disagree on line {line}")
            if seen[case_index, query, repeat]:
                raise ValueError(f"Duplicate request on line {line}")
            case = cases[case_index]
            if row["method"] != f"{case['routing_method']}/{case['kind']}":
                raise ValueError(f"Method and case disagree on line {line}")
            for name in ("probes", "node_budget", "explore_probability", "seed"):
                if _number(row[name], name) != case[name]:
                    raise ValueError(f"{name} and case disagree on line {line}")

            values = np.array(
                [
                    _number(
                        row[name],
                        name,
                        optional=case["kind"] == "float"
                        and name in ("binary_recall", "documents_scored"),
                    )
                    for name in STABLE
                ]
            )
            for name in RATES:
                if values[positions[name]] > 1:
                    raise ValueError(f"{name} must be between zero and one on line {line}")
            for name in COUNTS:
                value = values[positions[name]]
                if math.isfinite(value) and value != int(value):
                    raise ValueError(f"{name} must be an integer on line {line}")
            if values[positions["routed_documents"]] > metadata["dataset"]["documents"]:
                raise ValueError(f"Routed documents exceed the corpus on line {line}")
            stop = row["stop_reason"]
            if stop not in ("budget", "bound", "exhausted", "faiss"):
                raise ValueError(f"Unknown stop reason on line {line}")
            if seen[case_index, query].any():
                if not np.array_equal(stable[case_index, query], values, equal_nan=True):
                    raise ValueError(
                        f"Quality or work changed across timing repeats on line {line}"
                    )
                if stops[case_index, query] != stop:
                    raise ValueError(f"Stop reason changed across timing repeats on line {line}")
            else:
                stable[case_index, query] = values
                stops[case_index, query] = stop
            times[case_index, query, repeat] = [
                _number(
                    row[name],
                    name,
                    optional=case["kind"] == "float" and name in ("native_ms", "routing_ms"),
                )
                for name in TIMES
            ]
            seen[case_index, query, repeat] = True
    if not seen.all():
        raise ValueError(f"Missing {seen.size - int(seen.sum())} expected requests")

    # Routing must stay fixed across search policies and random exploration seeds.
    routing_fields = [
        positions[name] for name in ("routed_documents", "selected_buckets", "routing_float_recall")
    ]
    routing_reference = {}
    for case_index, case in enumerate(cases):
        key = (case["routing_method"], case["probes"])
        values = stable[case_index][:, routing_fields]
        if key in routing_reference and not np.array_equal(values, routing_reference[key]):
            raise ValueError(f"Routing changed between policies for {key}")
        routing_reference[key] = values
    return RecordedRun(metadata, cases, query_ids, stable, times, reference)


def bootstrap_weights(query_count, samples=1000, seed=20260909):
    """Every contrast uses the same resampled queries, including all their seeds."""
    if query_count <= 0 or samples <= 0:
        raise ValueError("Bootstrap query and sample counts must be positive")
    random = np.random.default_rng(seed)
    return (
        random.multinomial(query_count, np.full(query_count, 1 / query_count), size=samples)
        / query_count
    )


def _interval(values, weights):
    samples = weights @ values
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _optional_mean(values):
    return float(np.mean(values)) if np.isfinite(values).all() else None


def summarize_settings(run, weights):
    """One observation per query; exploration seeds are averaged inside each query."""
    groups = defaultdict(list)
    for index, case in enumerate(run.cases):
        groups[tuple(case[key] for key in POLICY_KEYS)].append(index)
    policies, summaries = [], []
    for identity, case_indices in sorted(groups.items()):
        details = dict(zip(POLICY_KEYS, identity))
        seeds = [run.cases[index]["seed"] for index in case_indices]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate search seed in policy {identity}")
        values = np.mean(run.stable[case_indices], axis=0)
        policy = dict(details, values=values, case_indices=case_indices)
        row = dict(
            details,
            queries=len(run.query_ids),
            search_seeds=len(seeds),
            seeds=",".join(str(seed) for seed in sorted(seeds)),
            timing_repetitions=run.times.shape[2],
            requests=len(seeds) * len(run.query_ids) * run.times.shape[2],
        )
        for column, name in enumerate(STABLE):
            row[name] = _optional_mean(values[:, column])
            if name in QUALITY and row[name] is not None:
                row[name + "_ci_low"], row[name + "_ci_high"] = _interval(
                    values[:, column], weights
                )
                seed_means = np.mean(run.stable[case_indices, :, column], axis=1)
                row[name + "_seed_min"] = float(seed_means.min())
                row[name + "_seed_max"] = float(seed_means.max())
            elif name in QUALITY:
                for suffix in ("_ci_low", "_ci_high", "_seed_min", "_seed_max"):
                    row[name + suffix] = None
        for column, name in enumerate(TIMES):
            samples = run.times[case_indices, :, :, column]
            prefix = name.removesuffix("_ms")
            for percentile in (50, 95):
                row[f"{prefix}_request_p{percentile}_ms"] = (
                    float(np.percentile(samples, percentile))
                    if np.isfinite(samples).all()
                    else None
                )
        request_times = run.times[case_indices, :, :, 0]
        query_medians = np.median(request_times, axis=2)
        # Equal seed weights within each query, rather than extra query observations.
        mean_query_medians = np.mean(query_medians, axis=0)
        for percentile in (50, 95):
            row[f"request_p{percentile}_ms"] = float(np.percentile(request_times, percentile))
            row[f"query_median_p{percentile}_ms"] = float(
                np.percentile(mean_query_medians, percentile)
            )
        policy["summary"] = row
        policies.append(policy)
        summaries.append(row)
    return policies, summaries


def paired_differences(policies, weights):
    lookup = {tuple(policy[key] for key in POLICY_KEYS): policy for policy in policies}
    rows = []
    for policy in policies:
        if policy["kind"] != "branch":
            continue
        router, probes = policy["routing_method"], policy["probes"]
        comparisons = [("branch-minus-scan", (router, probes, "scan", 0, 0.0))]
        if policy["explore_probability"]:
            comparisons.append(
                (
                    "exploration-minus-deterministic",
                    (router, probes, "branch", policy["node_budget"], 0.0),
                )
            )
        for label, key in comparisons:
            if key not in lookup:
                raise ValueError(f"Missing paired baseline {key}")
            baseline = lookup[key]
            for metric in QUALITY:
                column = STABLE.index(metric)
                differences = policy["values"][:, column] - baseline["values"][:, column]
                low, high = _interval(differences, weights)
                rows.append(
                    {
                        **{key: policy[key] for key in POLICY_KEYS},
                        "comparison": label,
                        "metric": metric,
                        "queries": len(differences),
                        "search_seeds": len(policy["case_indices"]),
                        "difference": float(np.mean(differences)),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return rows


def budget_axis(budgets):
    values = set(budgets)
    ordered = sorted(values - {0}) + ([0] if 0 in values else [])
    return ordered, ["Unlimited" if budget == 0 else str(budget) for budget in ordered]


def nondominated_points(points):
    """Measured points with no faster point reaching at least the same quality."""
    kept = []
    best_quality = -math.inf
    # For equal times, consider the highest quality first.
    for time, quality in sorted(set(points), key=lambda point: (point[0], -point[1])):
        if quality > best_quality:
            kept.append((time, quality))
            best_quality = quality
    return kept


def _write_csv(path, rows):
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def write_analysis_metadata(run_dir, directory, samples, seed):
    """Analysis can change without rerunning searches; keep its identity separate."""
    source_hashes = {}
    for name in ("metadata.json", "queries.csv", "reference.csv"):
        digest = hashlib.sha256()
        with (Path(run_dir) / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        source_hashes[name] = digest.hexdigest()
    metadata = {
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap": {"samples": samples, "seed": seed},
        "source_sha256": source_hashes,
    }
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _query_summaries(run):
    for case_index, case in enumerate(run.cases):
        for query, query_id in enumerate(run.query_ids):
            row = dict(
                case, query_id=query_id, query_row=query, timing_repetitions=run.times.shape[2]
            )
            row.update(
                {
                    name: (float(value) if math.isfinite(value) else None)
                    for name, value in zip(STABLE, run.stable[case_index, query])
                }
            )
            for column, name in enumerate(TIMES):
                values = run.times[case_index, query, :, column]
                row[name.removesuffix("_ms") + "_median_ms"] = (
                    float(np.median(values)) if np.isfinite(values).all() else None
                )
            yield row


def _figure(rows, columns, width=13, height=6.5):
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    return figure, axes


def _save(figure, directory, name):
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=160)
    figure.clear()
    return name + ".png"


def _select(policies, router, kind, probes=None, probability=None):
    return [
        policy
        for policy in policies
        if policy["routing_method"] == router
        and policy["kind"] == kind
        and (probes is None or policy["probes"] == probes)
        and (probability is None or policy["explore_probability"] == probability)
    ]


def _ndcg_upper(policies, reference=None):
    """One zero-based nDCG scale per figure, shared across methods and routers."""
    values = [
        value
        for policy in policies
        for value in (policy["summary"]["ndcg"], policy["summary"]["ndcg_ci_high"])
        if value is not None
    ]
    if reference is not None:
        values.append(reference)
    return min(1.0, max(0.1, math.ceil(max(values, default=0.0) * 10) / 10))


def plot_routing(policies, run, directory):
    figure, axes = _figure(2, 2, width=11)
    for row, router in enumerate(ROUTERS):
        scans = sorted(_select(policies, router, "scan"), key=lambda p: p["probes"])
        probes = [p["probes"] for p in scans]
        for column, metric in enumerate(("routing_float_recall", "routed_documents")):
            values = [p["summary"][metric] for p in scans]
            if metric == "routed_documents":
                values = [value / run.metadata["dataset"]["documents"] for value in values]
            axes[row, column].plot(probes, values, "o-", color="#087e8b")
            axes[row, column].set(
                xscale="log",
                ylim=(-0.02, 1.02),
                xlabel="Selected buckets",
                title=router.upper()
                + (" · neighbor coverage" if column == 0 else " · fraction of corpus routed"),
            )
            axes[row, column].set_xticks(probes, [str(probe) for probe in probes])
    top_k = run.metadata["config"]["search"]["top_k"]
    figure.suptitle(
        f"Routing alone · fraction of full-float top-{top_k} neighbors and corpus covered"
    )
    return _save(figure, directory, "routing")


def plot_overview(policies, directory):
    """Leave every measured policy visible, including points below its envelope."""
    figure, axes = _figure(2, 2, width=12, height=8)
    ndcg_upper = _ndcg_upper(policies)
    styles = (
        ("scan", 0.0, "#64748b", "Exact binary scan"),
        ("float", 0.0, "#111827", "Faiss float-IVF"),
        ("branch", 0.0, COLORS[0.0], "Deterministic branches"),
        ("branch", 0.1, COLORS[0.1], "Explore 0.1 · seed-averaged quality"),
    )
    for row, router in enumerate(ROUTERS):
        for column, metric in enumerate(("float_recall", "ndcg")):
            axis = axes[row, column]
            for kind, probability, color, label in styles:
                selected = _select(policies, router, kind, probability=probability)
                points = [
                    (policy["summary"]["request_p50_ms"], policy["summary"][metric])
                    for policy in selected
                ]
                if not points:
                    continue
                axis.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    s=26,
                    alpha=0.42,
                    label=label,
                )
                envelope = nondominated_points(points)
                axis.plot(
                    [point[0] for point in envelope],
                    [point[1] for point in envelope],
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                    marker="o",
                    markersize=3,
                    markerfacecolor="none",
                )
            axis.set(
                xlabel="Recorded request p50 latency (ms)",
                ylim=(0, 1 if column == 0 else ndcg_upper),
                title=router.upper() + (" · float neighbor recall" if column == 0 else " · nDCG"),
            )
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7, loc="best")
    figure.suptitle(
        "All measured policies · dashed lines are within-method empirical envelopes\n"
        "Exploratory description of this grid; no smooth fit or unseen operating points"
    )
    return _save(figure, directory, "overview")


def plot_probes(policies, run, directory):
    figure, axes = _figure(2, 2, width=11)
    reference_ndcg = float(np.mean(run.reference["ndcg"]))
    ndcg_upper = _ndcg_upper(
        [p for p in policies if p["kind"] in ("scan", "float")], reference_ndcg
    )
    for row, router in enumerate(ROUTERS):
        for kind, color, label in (
            ("scan", "#64748b", "Exact binary scan"),
            ("float", "#111827", "Faiss float-IVF"),
        ):
            selected = sorted(_select(policies, router, kind), key=lambda p: p["probes"])
            if not selected:
                continue
            for column, metric in enumerate(("float_recall", "ndcg")):
                axis = axes[row, column]
                x = [p["probes"] for p in selected]
                y = [p["summary"][metric] for p in selected]
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="--" if kind == "float" else "-",
                    color=color,
                    label=label,
                )
                axis.fill_between(
                    x,
                    [p["summary"][metric + "_ci_low"] for p in selected],
                    [p["summary"][metric + "_ci_high"] for p in selected],
                    color=color,
                    alpha=0.12,
                )
        probes = sorted({p["probes"] for p in _select(policies, router, "scan")})
        for column, axis in enumerate(axes[row]):
            axis.set(
                xscale="log",
                ylim=(0, 1 if column == 0 else ndcg_upper),
                xlabel="Selected buckets",
                title=router.upper() + (" · float neighbor recall" if column == 0 else " · nDCG"),
            )
            axis.set_xticks(probes, [str(probe) for probe in probes])
        axes[row, 1].axhline(
            reference_ndcg,
            color="#999999",
            linestyle=":",
            label="Full-float reference nDCG",
        )
        for axis in axes[row]:
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
    figure.suptitle("Exact scans at each probe count · binary scoring and candidate cutoff remain")
    return _save(figure, directory, "probes")


def plot_budgets(policies, directory, probes):
    figure, axes = _figure(2, 3, width=15, height=7)
    ndcg_upper = _ndcg_upper(
        [p for p in policies if p["probes"] == probes and p["kind"] in ("scan", "branch")]
    )
    metrics = ("float_recall", "ndcg", "request_p50_ms")
    for row, router in enumerate(ROUTERS):
        branches = _select(policies, router, "branch", probes)
        budgets, labels = budget_axis(p["node_budget"] for p in branches)
        for column, metric in enumerate(metrics):
            axis = axes[row, column]
            for probability in sorted({p["explore_probability"] for p in branches}):
                selected = {
                    p["node_budget"]: p["summary"]
                    for p in branches
                    if p["explore_probability"] == probability
                }
                x = [index for index, budget in enumerate(budgets) if budget in selected]
                points = [selected[budgets[index]] for index in x]
                color = COLORS.get(probability, "#875baf")
                label = "Deterministic" if probability == 0 else f"Explore {probability:g}"
                axis.plot(x, [p[metric] for p in points], "o-", color=color, label=label)
                if column < 2:
                    axis.fill_between(
                        x,
                        [p[metric + "_ci_low"] for p in points],
                        [p[metric + "_ci_high"] for p in points],
                        alpha=0.15,
                        color=color,
                    )
                else:
                    axis.plot(
                        x,
                        [p["request_p95_ms"] for p in points],
                        ":",
                        color=color,
                        label=label + " · p95",
                    )
            for baseline in _select(policies, router, "scan", probes):
                axis.axhline(
                    baseline["summary"][metric],
                    color="#64748b",
                    linestyle="--",
                    label="Same-bucket binary scan",
                )
                if column == 2:
                    axis.axhline(
                        baseline["summary"]["request_p95_ms"], color="#64748b", linestyle=":"
                    )
            axis.set_xticks(range(len(budgets)), labels, rotation=45, ha="right")
            axis.set_xlabel("Node budget (categorical; 0 is shown as Unlimited)")
            if 0 in budgets:
                axis.axvline(len(budgets) - 1.5, color="#bbbbbb", linestyle=":")
            if column < 2:
                axis.set_ylim(0, 1 if column == 0 else ndcg_upper)
            axis.set_title(
                router.upper() + " · " + ("float neighbor recall", "nDCG", "retrieval ms")[column]
            )
        for column, size in ((0, 7), (2, 6)):
            if axes[row, column].get_legend_handles_labels()[0]:
                axes[row, column].legend(fontsize=size, loc="best")
    figure.suptitle(
        f"{probes} selected buckets · measured budgets only · quality bands are query bootstrap intervals"
    )
    return _save(figure, directory, f"budget-p{probes:02d}")


def plot_exploration(differences, directory):
    figure, axes = _figure(2, 4, width=17, height=7)
    for row, router in enumerate(ROUTERS):
        for column, probes in enumerate(DISPLAY_PROBES):
            axis = axes[row, column]
            points = [
                p
                for p in differences
                if p["comparison"] == "exploration-minus-deterministic"
                and p["metric"] == "ndcg"
                and p["routing_method"] == router
                and p["probes"] == probes
            ]
            budgets, labels = budget_axis(p["node_budget"] for p in points)
            ordered = {p["node_budget"]: p for p in points}
            points = [ordered[budget] for budget in budgets]
            means = np.array([p["difference"] for p in points])
            if points:
                # A percentile interval needn't contain its sample mean; draw bounds directly.
                axis.plot(range(len(points)), means, "o-", color=COLORS[0.1])
                axis.vlines(
                    range(len(points)),
                    [p["ci_low"] for p in points],
                    [p["ci_high"] for p in points],
                    color=COLORS[0.1],
                )
            axis.axhline(0, color="#777777", linestyle="--")
            axis.set_xticks(range(len(budgets)), labels, rotation=60, ha="right")
            axis.set(
                title=f"{router.upper()} · probes={probes}", ylabel="Δ nDCG", xlabel="Node budget"
            )
    figure.suptitle("Exploration minus deterministic · paired queries · unadjusted 95% intervals")
    return _save(figure, directory, "exploration")


def plot_completion(policies, directory):
    figure, axes = _figure(2, 4, width=17, height=7)
    for row, router in enumerate(ROUTERS):
        for column, probes in enumerate(DISPLAY_PROBES):
            axis = axes[row, column]
            branches = _select(policies, router, "branch", probes)
            budgets, labels = budget_axis(p["node_budget"] for p in branches)
            for probability in sorted({p["explore_probability"] for p in branches}):
                points = {
                    p["node_budget"]: p["summary"]
                    for p in branches
                    if p["explore_probability"] == probability
                }
                x = [i for i, budget in enumerate(budgets) if budget in points]
                for metric, style, label in (
                    ("empty_result", "-", "empty"),
                    ("fewer_than_k", ":", "fewer than k"),
                ):
                    axis.plot(
                        x,
                        [points[budgets[i]][metric] for i in x],
                        marker="o",
                        linestyle=style,
                        color=COLORS.get(probability),
                        label=f"explore={probability:g} · {label}",
                    )
            axis.set_xticks(range(len(budgets)), labels, rotation=60, ha="right")
            axis.set(
                title=f"{router.upper()} · probes={probes}",
                ylim=(-0.02, 1.02),
                ylabel="Fraction of queries",
            )
            if branches and column == 0:
                axis.legend(fontsize=6)
    figure.suptitle(
        "Returned-result counts · empty and short results stay in every quality average"
    )
    return _save(figure, directory, "completion")


def write_report(run, summaries, artifacts, directory, samples, seed):
    queries = len(run.query_ids)
    repeats = run.times.shape[2]
    lines = [
        "# Search measurements",
        "",
        f"{run.metadata['dataset']['documents']:,} documents, {queries:,} queries, "
        f"{len(run.cases):,} cases, {repeats} timing repetitions per query and case.",
        "",
        "Quality uses each query once. Random policies average their search seeds within each query; "
        "timing repetitions do not add quality observations. The settings table records seed counts and the range of seed means.",
        "",
        "Times describe warm CPU requests with cached query embeddings: routing, search call and float reranking. "
        "Request p50/p95 include every recorded timing repetition. Query-median p50/p95 first take the median "
        "over timing repetitions, then average seeds within each query. These are different summaries; "
        "neither is an uncertainty interval for another machine or run.",
        "",
        f"Quality bands and paired differences use {samples:,} query-bootstrap draws (seed {seed}). "
        "Every comparison uses the same resampled query IDs. Intervals are unadjusted and exploratory, "
        "conditional on this corpus, query set, router and search seeds. They do not correct for selecting among many settings.",
        "",
        "IVF and sign routing have different bucket sizes and counts. Equal probes do not mean equal work. "
        "Routing coverage measures full-float reference neighbors present before binary scoring and candidate selection. "
        "The scan curves include those later losses. Float-IVF is a separate baseline, not the controlled binary comparison.",
        "",
        "Unlimited is a separate category after the finite budgets, not numeric zero on a log axis. "
        "All measured settings and failures remain in the tables; lines only connect measurements. "
        "The detailed budget figures show the preselected probe counts 1, 4, 16 and 64.",
        "nDCG panels start at zero and share one upper limit within each figure, rounded up to the next tenth "
        "from all displayed values, confidence bounds and any reference line (capped at 1). Recall panels retain the full 0–1 scale. "
        "This display choice does not change measurements or comparisons.",
        "The overview retains every policy point. Dashed envelopes connect measured nondominated points "
        "within each method and exploration policy. They are descriptive, selected from this same grid, "
        "and do not establish tuned winners or performance between observations.",
        "",
    ]
    for artifact in artifacts:
        lines += [f"![{Path(artifact).stem.replace('-', ' ')}]({artifact})", ""]
    unlimited = [row for row in summaries if row["kind"] == "branch" and row["node_budget"] == 0]
    mismatches = [row for row in unlimited if row["binary_recall"] != 1.0]
    lines += [
        "## Recorded checks",
        "",
        f"Unlimited settings: {len(unlimited)}; settings below exact same-bucket binary recall: {len(mismatches)}.",
        "",
        "An unlimited mismatch needs checking before treating the run as valid. An empty routed scope has zero recall by convention; "
        "the query table records routed_documents so that case can be distinguished from a search mismatch.",
        "",
        "## All settings",
        "",
        "| Router / method | Probes | Budget | Explore | Seeds | p50 ms | p95 ms | Float recall | nDCG | Empty | < k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        budget = (
            "—"
            if row["kind"] != "branch"
            else ("Unlimited" if row["node_budget"] == 0 else str(row["node_budget"]))
        )
        lines.append(
            f"| {row['routing_method']}/{row['kind']} | {row['probes']} | {budget} | "
            f"{row['explore_probability']:g} | {row['search_seeds']} | {row['request_p50_ms']:.4f} | "
            f"{row['request_p95_ms']:.4f} | {row['float_recall']:.3f} | {row['ndcg']:.3f} | "
            f"{row['empty_result']:.3f} | {row['fewer_than_k']:.3f} |"
        )
    lines += [
        "",
        "[Settings and seed ranges](settings.csv) · [Paired differences](paired_differences.csv) · "
        "[Per-query, per-seed summaries](query_summary.csv)",
        "[Analysis code, bootstrap settings and input hashes](metadata.json)",
        "",
        "The query table retains one row per case and query after checking repeat stability. "
        "The settings table includes binary recall, actual nodes, documents scored, bitplane words, "
        "candidate fill, short-result rates and budget-stop rates.",
        "",
    ]
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze_run(run_dir, bootstrap_samples=1000, seed=20260909):
    run = load_run(run_dir)
    weights = bootstrap_weights(len(run.query_ids), bootstrap_samples, seed)
    policies, summaries = summarize_settings(run, weights)
    differences = paired_differences(policies, weights)
    directory = Path(run_dir) / "analysis"
    directory.mkdir(exist_ok=True)
    provenance = write_analysis_metadata(run_dir, directory, bootstrap_samples, seed)
    _write_csv(directory / "query_summary.csv", _query_summaries(run))
    _write_csv(directory / "settings.csv", summaries)
    _write_csv(directory / "paired_differences.csv", differences)
    artifacts = [
        plot_overview(policies, directory),
        plot_routing(policies, run, directory),
        plot_probes(policies, run, directory),
    ]
    artifacts += [plot_budgets(policies, directory, probes) for probes in DISPLAY_PROBES]
    artifacts += [plot_exploration(differences, directory), plot_completion(policies, directory)]
    report = write_report(run, summaries, artifacts, directory, bootstrap_samples, seed)
    return dict(
        report=str(report),
        metadata=str(provenance),
        queries=len(run.query_ids),
        cases=len(run.cases),
        policies=len(summaries),
        requests=int(np.prod(run.times.shape[:3])),
        artifacts=[str(directory / artifact) for artifact in artifacts],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.bootstrap_samples, args.seed)
    print(f"Analysis: {result['report']}")


if __name__ == "__main__":
    main()
