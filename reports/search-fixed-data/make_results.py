"""Figures, tables and numbers for the fixed-dataset report.

Two kinds of input:
  * the dataset itself (codes, queries, reference answers, cached cluster assignments),
    which gives the facts that Section 3 reasons from;
  * a finished run of experiments/fixed_data_study.py, which gives Section 4.

Usage, from the repository root:
  python reports/search-fixed-data/make_results.py facts
  python reports/search-fixed-data/make_results.py results --run results/fixed-data-2026-09-12
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
sys.path.insert(0, str(ROOT))
FIGURES, EVIDENCE = REPORT / "figures", REPORT / "evidence"
POOL = ROOT / "data/real-evaluation-2026-09-10/n1000000"
ROUTERS = ROOT / "data/real-scale-2026-09-10/n1000000/routers"
CLUSTER_COUNTS = (64, 256, 1024, 4096, 8192)
TARGETS = (0.8, 0.9, 0.95, 0.99)
N, D, K = 1_000_000, 256, 100

# Colours: A blue, B orange, C green, matching the report's boxes; muted greys for chrome.
COLOR = dict(scan="#2A78D6", branch="#EB6834", keys="#1BAF7A")
LABEL = dict(scan="A: original", branch="B: bitplanes", keys="C: backward walk")
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 6,
})


def save_json(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def write_tex(path, text):
    path.write_text(text)


def fmt(value, digits=0):
    return f"{value:,.{digits}f}"


# ----- Dataset facts -----

def load_dataset():
    queries = np.load(POOL / "queries.npy").astype(np.float64)
    reference = np.load(POOL / "reference.npy")
    codes = np.load(POOL / "codes.npy")
    bits = np.unpackbits(codes.view(np.uint8), axis=1, bitorder="little").astype(bool)
    return queries, reference, bits


def routing(queries, reference):
    """Recall and documents opened as a function of P, for every cached cluster count."""
    import faiss
    faiss.omp_set_num_threads(1)
    out = {}
    for clusters in CLUSTER_COUNTS:
        folder = ROUTERS / f"ivf-{clusters}-seed42"
        centroids = np.load(folder / "centroids.npy")
        assignments = np.load(folder / "assignments.npy")
        sizes = np.bincount(assignments, minlength=clusters)
        index = faiss.IndexFlatIP(centroids.shape[1])
        index.add(centroids)
        _, positions = index.search(np.ascontiguousarray(queries.astype(np.float32)), clusters)
        rank_of = np.empty_like(positions)
        for row in range(len(queries)):
            rank_of[row, positions[row]] = np.arange(1, clusters + 1)
        neighbour_ranks = rank_of[np.arange(len(queries))[:, None], assignments[reference]]
        histogram = np.bincount(neighbour_ranks.ravel(), minlength=clusters + 1)
        recall = np.cumsum(histogram)[1:] / neighbour_ranks.size
        docs = np.cumsum(sizes[positions], axis=1).mean(0)
        probes = {str(t): int(np.searchsorted(recall, t) + 1) for t in TARGETS}
        out[clusters] = dict(clusters=clusters, probes=probes,
                             docs_at_target={str(t): float(docs[p - 1]) for t, p in zip(TARGETS, probes.values())},
                             recall_at_target={str(t): float(recall[p - 1]) for t, p in zip(TARGETS, probes.values())},
                             curve_P=list(range(1, clusters + 1)), curve_recall=recall.tolist(), curve_docs=docs.tolist(),
                             cluster_size_median=float(np.median(sizes)), cluster_size_max=int(sizes.max()))
    return out


def weights_and_gap(queries, reference, bits):
    """Query weight concentration, the gap g, and how the reference documents agree with query signs."""
    Q = np.abs(queries).sum(1)
    sorted_weights = -np.sort(-np.abs(queries), axis=1)
    top_share = (np.cumsum(sorted_weights, axis=1) / Q[:, None]).mean(0)  # share of Q in the j largest weights
    gaps = {1: [], 10: [], 100: []}
    agree_all = np.zeros(64)  # only the largest 64 weights are needed
    agree_neighbours = np.zeros(D)
    for i in range(len(queries)):
        signs = bits[reference[i]].astype(np.float64) * 2 - 1
        scores = np.sort(signs @ queries[i])[::-1]
        for k in gaps:
            gaps[k].append((Q[i] - scores[k - 1]) / 2 / Q[i])
        order = np.argsort(-np.abs(queries[i]))
        preferred = queries[i] >= 0
        agree_all += (bits[:, order[:64]] == preferred[order[:64]]).mean(0)
        agree_neighbours += (bits[reference[i]][:, order] == preferred[order]).mean(0)
    agree_all /= len(queries)
    agree_neighbours /= len(queries)
    return dict(top_share=top_share.tolist(),
                gap={str(k): dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in gaps.items()},
                agree_all_by_rank=agree_all.tolist(), agree_neighbours_by_rank=agree_neighbours.tolist(),
                splits_before_pruning=float(np.mean([int(np.searchsorted(np.cumsum(sorted_weights[i]) / Q[i], gaps[100][i]) + 1)
                                                     for i in range(len(queries))])))


def match_mask(bits, query, positions):
    return (bits[:, positions] == (query[positions] >= 0)).all(1)


def filters(queries, reference, bits, route):
    """How many documents and how many reference neighbours each kind of filter keeps."""
    absmean = np.abs(queries).mean(0)
    largest_mean = np.argsort(-absmean)
    keys = {}
    kept = np.zeros(24); inside = np.zeros(24)
    for i in range(len(queries)):
        order = np.argsort(-np.abs(queries[i]))[:24]
        agree = bits[:, order] == (queries[i][order] >= 0)      # one million rows, 24 positions
        still_in = np.cumprod(agree, axis=1, dtype=bool)          # in the group after j splits
        kept += still_in.sum(0); inside += still_in[reference[i]].sum(0)
    splits = [dict(splits=j + 1, docs_kept=float(kept[j] / len(queries)), neighbours_kept=float(inside[j] / len(queries)))
              for j in range(24)]
    for h in (4, 8, 12):
        for name, positions in (("first", np.arange(h)), ("largest-mean", largest_mean[:h])):
            kept, inside = [], []
            for i in range(len(queries)):
                hit = match_mask(bits, queries[i], positions)
                kept.append(hit.sum()); inside.append(hit[reference[i]].sum())
            share = (np.abs(queries[:, positions]).sum(1) / np.abs(queries).sum(1)).mean()
            keys[f"{name}-{h}"] = dict(positions=[int(p) for p in positions], weight_share=float(share),
                                       docs_kept=float(np.mean(kept)), neighbours_kept=float(np.mean(inside)))
    # Inside the clusters opened for 99% recall with 4096 clusters: do splits still remove little?
    facts = route[4096]
    folder = ROUTERS / "ivf-4096-seed42"
    centroids = np.load(folder / "centroids.npy"); assignments = np.load(folder / "assignments.npy")
    probe = facts["probes"]["0.99"]
    inside_docs, inside_after12, inside_neighbours12 = [], [], []
    for i in range(len(queries)):
        top = np.argsort(-(centroids @ queries[i]))[:probe]
        inside = np.isin(assignments, top)
        keep = match_mask(bits, queries[i], np.argsort(-np.abs(queries[i]))[:12]) & inside
        inside_docs.append(inside.sum()); inside_after12.append(keep.sum()); inside_neighbours12.append(keep[reference[i]].sum())
    return dict(splits=splits, keys=keys,
                inside_opened=dict(probes=probe, docs=float(np.mean(inside_docs)), after_12_splits=float(np.mean(inside_after12)),
                                   neighbours_after_12_splits=float(np.mean(inside_neighbours12))))


def facts(recompute=False):
    cached = EVIDENCE / "dataset-facts.json"
    if cached.exists() and not recompute:
        data = json.loads(cached.read_text())
    else:
        queries, reference, bits = load_dataset()
        route = routing(queries, reference)
        data = dict(routing={str(c): v for c, v in route.items()}, weights=weights_and_gap(queries, reference, bits),
                    filters=filters(queries, reference, bits, route))
        save_json(cached, data)
    draw_facts(data)
    tables_facts(data)
    print("dataset facts written")


def draw_facts(data):
    route = data["routing"]
    # Figure: recall against documents opened, one line per cluster count.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for colour, clusters in zip(BLUES, CLUSTER_COUNTS):
        r = route[str(clusters)]
        docs, recall = np.array(r["curve_docs"]), np.array(r["curve_recall"])
        ax.plot(docs, 100 * recall, color=colour, label=f"{clusters:,} clusters")
    for t in TARGETS:
        ax.axhline(100 * t, color=GRID, linewidth=0.8)
    ax.set_xscale("log"); ax.set_xlim(100, 1.2e6); ax.set_ylim(0, 102)
    ax.set_xlabel("documents in the opened clusters (average per query)")
    ax.set_ylabel("share of the true top-100 inside (%)")
    ax.legend(loc="lower right", ncol=1)
    fig.savefig(FIGURES / "routing.pdf"); plt.close(fig)

    # Figure: how much a split removes, by weight rank.
    agree = np.array(data["weights"]["agree_all_by_rank"])
    ranks = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(ranks, 100 * (1 - agree[:24]), color=COLOR["branch"], width=0.6)
    ax.axhline(50, color=MUTED, linewidth=1)
    ax.text(24.4, 51.5, "50%: what a balanced bit would remove", ha="right", fontsize=8, color=INK2)
    ax.annotate("0.1%", (1, 100 * (1 - agree[0])), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("position, ordered by the query's weight (1 = largest weight)")
    ax.set_ylabel("documents removed by\nkeeping the preferred sign (%)")
    ax.set_xticks(ranks); ax.set_ylim(0, 60)
    fig.savefig(FIGURES / "split-removes.pdf"); plt.close(fig)

    # Figure: filter quality, documents kept against neighbours kept.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    r = route["4096"]
    docs, recall = np.array(r["curve_docs"]), np.array(r["curve_recall"])
    keep = np.unique(np.clip(np.geomspace(1, 4096, 40).astype(int), 1, 4096)) - 1
    ax.plot(100 * docs[keep] / N, 100 * recall[keep], color=COLOR["scan"], marker="o", markersize=3.5,
            label="clusters: open the P closest of 4,096")
    s = data["filters"]["splits"]
    ax.plot([100 * x["docs_kept"] / N for x in s], [x["neighbours_kept"] for x in s], color=COLOR["branch"],
            marker="o", markersize=3.5, label="sign splits: keep the query's j preferred signs")
    for x in s:
        if x["splits"] in (4, 12, 24):
            ax.annotate(f"j={x['splits']}", (100 * x["docs_kept"] / N, x["neighbours_kept"]), textcoords="offset points",
                        xytext=(6, -9), fontsize=7.5, color=INK2)
    for name, x in data["filters"]["keys"].items():
        ax.plot(100 * x["docs_kept"] / N, x["neighbours_kept"], color=COLOR["keys"], marker="s", markersize=5, linestyle="none",
                label="fixed keys: the query's own key" if name == "first-4" else None)
        ax.annotate(name.replace("largest-mean", "heaviest").replace("-", " ") + " bits", (100 * x["docs_kept"] / N, x["neighbours_kept"]),
                    textcoords="offset points", xytext=(5, -9 if "first" in name else 4), fontsize=7, color=INK2)
    for p in (r["probes"]["0.8"], r["probes"]["0.99"]):
        ax.annotate(f"P={p}", (100 * docs[p - 1] / N, 100 * recall[p - 1]), textcoords="offset points", xytext=(-6, -12), fontsize=7.5, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_xlim(0.01, 120); ax.set_ylim(0, 102)
    ax.set_xlabel("documents kept (% of one million, log scale)")
    ax.set_ylabel("true top-100 neighbours kept (%)")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES / "filter-quality.pdf"); plt.close(fig)


def tables_facts(data):
    route = data["routing"]
    lines = [r"\begin{tabular}{@{}rrrrrrrrr@{}}\toprule",
             r"Clusters $C$ & \multicolumn{2}{c}{80\%} & \multicolumn{2}{c}{90\%} & \multicolumn{2}{c}{95\%} & \multicolumn{2}{c}{99\%}\\",
             r" & $P$ & docs & $P$ & docs & $P$ & docs & $P$ & docs\\\midrule"]
    for clusters in CLUSTER_COUNTS:
        r = route[str(clusters)]
        cells = [f"{clusters:,}"]
        for t in TARGETS:
            cells += [f"{r['probes'][str(t)]:,}", fmt(r["docs_at_target"][str(t)])]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}"]
    write_tex(EVIDENCE / "routing-table.tex", "\n".join(lines) + "\n")

    w, f = data["weights"], data["filters"]
    top = w["top_share"]
    numbers = {
        "gapShare": f"{100 * w['gap']['100']['mean']:.0f}",
        "gapShareMin": f"{100 * w['gap']['100']['min']:.0f}", "gapShareMax": f"{100 * w['gap']['100']['max']:.0f}",
        "gapOneShare": f"{100 * w['gap']['1']['mean']:.0f}",
        "topOneShare": f"{100 * top[0]:.1f}", "topTwelveShare": f"{100 * top[11]:.0f}", "topTwentyFourShare": f"{100 * top[23]:.0f}",
        "topSixtyFourShare": f"{100 * top[63]:.0f}",
        "splitsBeforePruning": f"{w['splits_before_pruning']:.0f}",
        "agreeRankOne": f"{100 * w['agree_all_by_rank'][0]:.1f}", "agreeRankTwo": f"{100 * w['agree_all_by_rank'][1]:.0f}",
        "agreeRankTwelve": f"{100 * w['agree_all_by_rank'][11]:.0f}",
        "removedTwelve": f"{100 * (1 - w['agree_all_by_rank'][11]):.0f}",
        "neighbourAgreeTwelve": f"{100 * w['agree_neighbours_by_rank'][11]:.0f}",
        "groupTwelveDocs": fmt(f["splits"][11]["docs_kept"]), "groupTwelveShare": f"{100 * f['splits'][11]['docs_kept'] / N:.1f}",
        "groupTwelveNeighbours": f"{f['splits'][11]['neighbours_kept']:.0f}",
        "groupTwentyFourDocs": fmt(f["splits"][23]["docs_kept"]), "groupTwentyFourNeighbours": f"{f['splits'][23]['neighbours_kept']:.0f}",
        "insideDocs": fmt(f["inside_opened"]["docs"]), "insideAfterTwelve": fmt(f["inside_opened"]["after_12_splits"]),
        "insideShareAfterTwelve": f"{100 * f['inside_opened']['after_12_splits'] / f['inside_opened']['docs']:.0f}",
        "insideNeighboursAfterTwelve": f"{f['inside_opened']['neighbours_after_12_splits']:.0f}",
        "keyFirstTwelveShare": f"{100 * f['keys']['first-12']['weight_share']:.1f}",
        "keyFirstTwelveDocs": fmt(f["keys"]["first-12"]["docs_kept"]), "keyFirstTwelveNeighbours": f"{f['keys']['first-12']['neighbours_kept']:.0f}",
        "keyHeavyTwelveShare": f"{100 * f['keys']['largest-mean-12']['weight_share']:.1f}",
        "keyHeavyTwelveDocs": fmt(f["keys"]["largest-mean-12"]["docs_kept"]), "keyHeavyTwelveNeighbours": f"{f['keys']['largest-mean-12']['neighbours_kept']:.0f}",
        "keyFirstEightShare": f"{100 * f['keys']['first-8']['weight_share']:.1f}",
    }
    for clusters, word in ((4096, "Cfour"), (8192, "Ceight")):  # macro names may not contain digits
        r = route[str(clusters)]
        for t, name in zip(TARGETS, ("Eighty", "Ninety", "NinetyFive", "NinetyNine")):
            numbers[f"probes{name}{word}"] = f"{r['probes'][str(t)]:,}"
            numbers[f"docs{name}{word}"] = fmt(r["docs_at_target"][str(t)])
            numbers[f"docsShare{name}{word}"] = f"{100 * r['docs_at_target'][str(t)] / N:.1f}"
    write_tex(EVIDENCE / "numbers-facts.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items()))


# ----- Results of the sweep -----

def read_rows(run):
    with (run / "settings.csv").open() as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if value in ("", None):
                continue
            try:
                row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                pass
    return [r for r in rows if r["status"] == "complete"]


def setting_text(row):
    base = "global" if row["clusters"] == 1 else f"$C={row['clusters']:,}$, $P={row['probes']:,}$"
    if row["method"] == "branch":
        leaf = "no splits" if row["leaf_size"] >= N else f"leaf {row['leaf_size']}"
        budget = "" if row["node_budget"] == 0 else f", budget {row['node_budget']}"
        return f"{base}; {leaf}{budget}"
    if row["method"] == "keys":
        return f"{base}; $h={row['key_bits']}$"
    return base


def results(run):
    rows = read_rows(run)
    selected = json.loads((run / "summary.json").read_text())["selected"]
    checks = json.loads((run / "checks.json").read_text())
    units = json.loads((run / "unit-costs.json").read_text())
    draw_results(rows, selected, units)
    tables_results(rows, selected, checks, units)
    print(f"results written from {len(rows)} complete settings")


def draw_results(rows, selected, units):
    # Figure: fastest setting per method at each recall target.
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.26
    for k, method in enumerate(("scan", "branch", "keys")):
        xs, ys = [], []
        for t_index, target in enumerate(TARGETS):
            match = [s for s in selected if s["method"] == method and s["recall_target"] == target]
            if match:
                xs.append(t_index + (k - 1) * width); ys.append(match[0]["query_ms_p50"])
        bars = ax.bar(xs, ys, width=width - 0.03, color=COLOR[method], label=LABEL[method])
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.2f}" if y < 10 else f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels([f"{int(100 * t)}% recall" for t in TARGETS])
    ax.set_ylabel("median time per query (ms)")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES / "results.pdf"); plt.close(fig)

    # Figure: every measured setting, time against recall.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method, z in (("keys", 3), ("branch", 4), ("scan", 5)):
        sample = [r for r in rows if r["method"] == method]
        ax.scatter([r["query_ms_p50"] for r in sample], [100 * r["recall"] for r in sample], s=22, color=COLOR[method],
                   alpha=0.85, edgecolors="white", linewidths=0.8, label=LABEL[method], zorder=z)
    best = sorted([s for s in selected if s["method"] == "scan"], key=lambda s: s["query_ms_p50"])
    ax.plot([s["query_ms_p50"] for s in best], [100 * s["recall"] for s in best], color=COLOR["scan"], linewidth=1.2, zorder=2)
    ax.set_xscale("log"); ax.set_ylim(0, 102)
    ax.set_xlabel("median time per query (ms, log scale)")
    ax.set_ylabel("recall of the true top-100 (%)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "all-settings.pdf"); plt.close(fig)

    # Figure: search time predicted from the counts against measured search time.
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    low, high = 1e9, 0
    for method in ("scan", "branch", "keys"):
        sample = [r for r in rows if r["method"] == method and "predicted_search_ms" in r and r["predicted_search_ms"] != ""]
        x = [r["search_ms_p50"] for r in sample]; y = [r["predicted_search_ms"] for r in sample]
        low, high = min(low, min(x)), max(high, max(x))
        ax.scatter(x, y, s=22, color=COLOR[method], alpha=0.8, edgecolors="white", linewidths=0.8, label=LABEL[method], zorder=3)
    ax.plot([low * 0.8, high * 1.2], [low * 0.8, high * 1.2], color=MUTED, linewidth=1, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("measured search time (ms)"); ax.set_ylabel("time from the formula (ms)")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES / "predicted-vs-measured.pdf"); plt.close(fig)


def tables_results(rows, selected, checks, units):
    lines = [r"\begin{tabularx}{\linewidth}{@{}llYrrr@{}}\toprule",
             r"Target & Method & Setting & Recall & Median ms & Docs scored\\\midrule"]
    for target in TARGETS:
        first = True
        for method in ("scan", "branch", "keys"):
            match = [s for s in selected if s["method"] == method and s["recall_target"] == target]
            if not match:
                continue
            s = match[0]
            lines.append(f"{int(100 * target)}\\% & {LABEL[method]} & {setting_text(s)} & {100 * s['recall']:.2f}\\% & "
                         f"{s['query_ms_p50']:.3f} & {fmt(s['documents_scored'])}\\\\" if first else
                         f" & {LABEL[method]} & {setting_text(s)} & {100 * s['recall']:.2f}\\% & {s['query_ms_p50']:.3f} & {fmt(s['documents_scored'])}\\\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule\end{tabularx}"]
    write_tex(EVIDENCE / "selected-table.tex", "\n".join(lines) + "\n")

    # Formula checks: for each checked quantity, how many settings and the largest deviation.
    summary = {}
    for entry in checks:
        for name, value in entry.items():
            if isinstance(value, dict):
                deviation = abs(value["predicted"] - value["measured"]) / max(abs(value["measured"]), 1e-9)
                item = summary.setdefault(name, dict(settings=0, worst=0.0))
                item["settings"] += 1; item["worst"] = max(item["worst"], deviation)
    pretty = dict(documents_scored="Documents scored, $F$ (exact settings)", recall="Recall $=r_{\\rm route}$ (exact settings)",
                  leaf_words="Leaf words $V_l$ (B without splits)", split_words="Split words $V_s$ (B without splits)",
                  keys="Keys visited $E=2^h$", key_lookups="Lookups $H=2^hP$", directory_bytes="Directory bytes $=20U$",
                  codes_bytes="Code bytes $=32N$", row_ids_bytes="ID bytes $=8N$", bitplanes_bytes="Bitplane bytes $=8d\\sum_cW_c$")
    lines = [r"\begin{tabular}{@{}lrr@{}}\toprule", r"Quantity (formula) & Settings checked & Largest difference\\\midrule"]
    for name in ("documents_scored", "recall", "leaf_words", "split_words", "keys", "key_lookups", "codes_bytes", "row_ids_bytes", "bitplanes_bytes", "directory_bytes"):
        if name in summary:
            worst = summary[name]["worst"]
            lines.append(f"{pretty[name]} & {summary[name]['settings']} & {'none' if worst < 1e-9 else f'{100 * worst:.2f}\\%'}\\\\")
    lines += [r"\bottomrule\end{tabular}"]
    write_tex(EVIDENCE / "checks-table.tex", "\n".join(lines) + "\n")

    names = dict(call="per call, $c_{\\rm call}$", documents_scored="per scored document, $c_{\\rm score}$",
                 split_words="per split word, $c_{\\rm split}$", leaf_words="per leaf word, $c_{\\rm leaf}$",
                 nodes="per group taken from the queue, $c_{\\rm group}$", key_lookups="per directory lookup, $c_{\\rm lookup}$",
                 keys="per key generated, $c_{\\rm key}$")
    lines = [r"\begin{tabular}{@{}llrrr@{}}\toprule", r"Method & Cost & Microseconds & Settings & Median error\\\midrule"]
    for method in ("scan", "branch", "keys"):
        if method not in units:
            continue
        u = units[method]
        error = f"{u['median_abs_error_percent']:.1f}\\%"
        first = True
        for feature, value in u["ms_per_unit"].items():
            head = f"{LABEL[method]} & " if first else " & "
            tail = f" & {u['settings']} & {error}" if first else " & & "
            lines.append(f"{head}{names[feature]} & {1000 * value:.4f}{tail}\\\\")
            first = False
    lines += [r"\bottomrule\end{tabular}"]
    write_tex(EVIDENCE / "unit-costs-table.tex", "\n".join(lines) + "\n")

    # Numbers for the prose.
    numbers = {}
    for target, name in zip(TARGETS, ("Eighty", "Ninety", "NinetyFive", "NinetyNine")):
        for method in ("scan", "branch", "keys"):
            match = [s for s in selected if s["method"] == method and s["recall_target"] == target]
            if match:
                numbers[f"ms{name}{method.capitalize()}"] = f"{match[0]['query_ms_p50']:.2f}"
    a99 = [s for s in selected if s["method"] == "scan" and s["recall_target"] == 0.99]
    b99 = [s for s in selected if s["method"] == "branch" and s["recall_target"] == 0.99]
    c99 = [s for s in selected if s["method"] == "keys" and s["recall_target"] == 0.99]
    if a99 and b99 and c99:
        numbers["ratioBoverA"] = f"{b99[0]['query_ms_p50'] / a99[0]['query_ms_p50']:.2f}"
        numbers["ratioCoverA"] = f"{c99[0]['query_ms_p50'] / a99[0]['query_ms_p50']:.2f}"
    for method in units:
        numbers[f"unitError{method.capitalize()}"] = f"{units[method]['median_abs_error_percent']:.1f}"
        numbers[f"unitWorst{method.capitalize()}"] = f"{units[method]['max_abs_error_percent']:.0f}"
    if "scan" in units:
        numbers["scoreMicroseconds"] = f"{1000 * units['scan']['ms_per_unit']['documents_scored']:.3f}"
    if "branch" in units:
        numbers["splitNanoseconds"] = f"{1e6 * units['branch']['ms_per_unit']['split_words']:.1f}"
    if "keys" in units:
        numbers["lookupNanoseconds"] = f"{1e6 * units['keys']['ms_per_unit']['key_lookups']:.0f}"
    numbers["settingsMeasured"] = str(len(rows))
    exact_global = [r for r in rows if r["method"] == "scan" and r["clusters"] == 1]
    if exact_global:
        numbers["globalScanMs"] = f"{exact_global[0]['query_ms_p50']:.1f}"
    global_b = [r for r in rows if r["method"] == "branch" and r["clusters"] == 1 and r["leaf_size"] == 128 and r["node_budget"] == 0]
    if global_b:
        numbers["globalBranchMs"] = f"{global_b[0]['query_ms_p50']:.0f}"
        numbers["globalBranchSplitWords"] = fmt(global_b[0]["split_words"] / 1e6, 0)
        numbers["globalBranchGroups"] = fmt(global_b[0]["nodes"])
    write_tex(EVIDENCE / "numbers-results.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("facts", "results"))
    parser.add_argument("--run", type=Path, default=ROOT / "results/fixed-data-2026-09-12")
    parser.add_argument("--recompute", action="store_true", help="recompute the dataset facts instead of reusing the cache")
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True); EVIDENCE.mkdir(exist_ok=True)
    if args.stage == "facts":
        facts(args.recompute)
    else:
        results(args.run.resolve())


if __name__ == "__main__":
    main()
