"""Make readable tables and plots from the K sweep and prefix depth measurements."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fixed_data_comparison import save_json
from experiments.k_prefix_study import KS, OUTPUT, TARGETS
from experiments.prefix_study_v2 import read_json
from experiments.prefix_depth_study import FOLDER, STARTS, write_csv, allows_stop

METHODS = [("scan", "A: scan", "#286493"),
           ("branch", "B: Bitplanes", "#C36620"),
           ("prefix", "C: Backward Walk", "#7850A4")]


def comparisons(folders):
    # Selection is made from the initial measurement. Repeat timings do not pick a new winner.
    initial, repeated = {}, {}
    for folder in folders:
        for phase, target in [("main", initial), ("repeat", repeated)]:
            path = folder / phase / "settings.json"
            if path.exists():
                for row in read_json(path):
                    target[(str(folder), row["setting_id"])] = dict(row, study=str(folder))
    comparisons = []
    for k in sorted({row["top_k"] for row in initial.values()}):
        for target in TARGETS:
            for method, _, _ in METHODS:
                eligible = [(key,row) for key,row in initial.items()
                            if row["method"]==method and row["top_k"]==k
                            and row["qualified"] and row["recall"]>=target]
                if not eligible:
                    continue
                # Match the runner when two measured medians are exactly equal.
                key, selected = min(eligible, key=lambda item:(item[1]["p50_ms"], item[0]))
                result = repeated.get(key)
                if result is None or not result["qualified"] or result["recall"]<target:
                    comparisons.append(dict(top_k=k, target=target, method=method,
                        status="selected setting needs a qualified repeat", setting_id=key[1],study=key[0]))
                    continue
                comparisons.append(dict(result, target=target, status="complete",
                    selection_p50_ms=selected["p50_ms"]))
    save_json(OUTPUT / "comparison.json", comparisons)
    return comparisons


def plot_k(rows):
    ks = sorted({row["top_k"] for row in rows})
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,target in zip(axes.flat,TARGETS):
        for method,label,color in METHODS:
            own={r["top_k"]:r for r in rows if r["method"]==method and r["target"]==target and r["status"]=="complete"}
            y=[own[k]["process_p50_median_ms"] if k in own else np.nan for k in ks]
            low=[own[k]["process_p50_min_ms"] if k in own else np.nan for k in ks]
            high=[own[k]["process_p50_max_ms"] if k in own else np.nan for k in ks]
            axis.plot(range(len(ks)),y,"o-",label=label,color=color)
            axis.fill_between(range(len(ks)),low,high,color=color,alpha=.12)
            # A fast fallback must not look like a successful pruning strategy.
            plain = [i for i,k in enumerate(ks) if k in own and (
                (method == "branch" and own[k]["mean_bitplane_words"] == 0)
                or (method == "prefix" and own[k]["start_depth"] == 0))]
            axis.plot(plain,[y[i] for i in plain],"o",color=color,markerfacecolor="white")
        axis.set(title=f"At least {target:.0%} average recall",xlabel="Results requested per query (K)",
                 ylabel="Median query time (ms)",xticks=range(len(ks)),xticklabels=ks)
        axis.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle("Same 522,931 documents and 1,000 queries, Qwen 32-bit binary index\n"
                 "Fastest tested settings meeting each target; bands show three repeats\n"
                 "Hollow B points: no bit splits. Hollow C points: depth 0, a full scan.",fontsize=12)
    for extension in ["png","pdf"]:
        fig.savefig(OUTPUT / f"k-latency.{extension}",dpi=170)
    plt.close(fig)


def summarize_local():
    groups=defaultdict(list)
    for folder in sorted(FOLDER.glob("timing-*")):
        assert (folder/"complete.json").exists(),folder
        with (folder/"queries.jsonl").open() as source:
            for line in source:
                row=json.loads(line)
                groups[(row["method"],row["name"])].append(row)
    summary=[]
    for (method,name),rows in groups.items():
        assert len(rows)==3000,(method,name,len(rows))
        times=[r["local_ms"] for r in rows]
        medians=[float(np.median([r["local_ms"] for r in rows if r["repetition"]==rep])) for rep in range(3)]
        first=rows[0]
        item={key:first[key] for key in ["top_k","start_depth","candidate_target","leaf_size","node_budget"] if key in first}
        item.update(method=method,name=name,local_ms=float(np.median(medians)),
                    min_ms=min(medians),max_ms=max(medians),p95_ms=float(np.percentile(times,95)))
        for field in ["local_recall","global_recall","documents_scored","prefix_lookups","prefix_levels","final_depth","nodes","bitplane_words","leaf_words"]:
            if "recall" in field:
                item[field]=sum(round(r[field]*first["top_k"]) for r in rows)/(len(rows)*first["top_k"])
            else:
                item[field]=sum(r.get(field,0) for r in rows)/len(rows)
        summary.append(item)
    save_json(FOLDER/"timing-summary.json",summary)
    return summary


def numeric_csv(path):
    with path.open() as source:
        return [{key:float(value) for key,value in row.items()} for row in csv.DictReader(source)]


def plot_depth(local):
    depths=numeric_csv(FOLDER/"depth-summary.csv")
    gaps=numeric_csv(FOLDER/"query-stopping-gaps.csv")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,k in zip(axes.flat,[1,5,10,50]):
        own=sorted([r for r in depths if r["top_k"]==k],key=lambda r:-r["depth"])
        for field,label,color in [("local_recall","Within opened clusters","#286493"),
                                   ("global_recall","Across all documents","#7850A4")]:
            axis.plot([r["depth"] for r in own],[100*r[field] for r in own],label=label,color=color)
        axis.set(title=f"K = {k}",xlabel="Prefix depth, from specific to broad",ylabel="Average recall (%)",
                 xlim=(32,0),ylim=(0,102),xticks=[32,24,16,12,8,4,2,0])
        axis.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle("C after every prefix depth, same 1,000 queries at every point\n"
                 "Same 78 of 1,024 clusters, about 41,147 opened documents per query",fontsize=13)
    fig.savefig(FOLDER/"recall-by-depth.png",dpi=170)
    fig.savefig(FOLDER/"recall-by-depth.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    geometry=sorted([r for r in depths if r["top_k"]==1],key=lambda r:-r["depth"])
    axes[0].plot([r["depth"] for r in geometry],[r["documents_scored"] for r in geometry],"o-",color="#7850A4")
    axes[0].set(xlim=(32,0),xlabel="Prefix depth",ylabel="Mean cumulative documents scored",
                title="Documents exposed by the same prefix ranges")
    gap_summary=[]
    for k in KS:
        own=[r for r in gaps if r["top_k"]==k]
        extra=np.array([r["extra_documents_scored"] for r in own])
        gap_summary.append(dict(top_k=k,fraction_extra_scoring=float(np.mean(extra>0)),
            mean_extra_scoring=float(extra.mean()),median_extra_scoring=float(np.median(extra)),
            p90_extra_scoring=float(np.percentile(extra,90)),
            mean_correct_depth=float(np.mean([r["correct_depth"] for r in own])),
            mean_stop_depth=float(np.mean([r["stop_depth"] for r in own])),
            fraction_correct_only_at_root=float(np.mean([r["correct_depth"]==0 for r in own]))))
    axes[1].bar(range(len(KS)),[r["mean_extra_scoring"] for r in gap_summary],color="#C36620")
    axes[1].set(xticks=range(len(KS)),xticklabels=KS,xlabel="Results requested (K)",
                ylabel="Mean additional documents scored",
                title="Scoring after all exact local answers were present")
    for axis in axes: axis.grid(axis="y",alpha=.2)
    fig.savefig(FOLDER/"prefix-work.png",dpi=170)
    plt.close(fig)
    write_csv(FOLDER/"stopping-gap-summary.csv",gap_summary)

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,k in zip(axes.flat,[1,5,10,50]):
        own=sorted([r for r in local if r["method"]=="prefix" and r["top_k"]==k and r["candidate_target"]==0],
                   key=lambda r:r["start_depth"])
        axis.plot([r["start_depth"] for r in own],[r["local_ms"] for r in own],"o-",color="#7850A4",label="C, exact within opened clusters")
        a=next(r for r in local if r["method"]=="scan" and r["top_k"]==k)
        axis.axhline(a["local_ms"],color="#286493",label="A, scan same clusters")
        axis.set(title=f"K = {k}",xlabel="Starting prefix depth",ylabel="Median local search time (ms)",xticks=STARTS)
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Does starting C deeper actually save time?\nNormal searches without diagnostic recording",fontsize=13)
    fig.savefig(FOLDER/"starting-depth-latency.png",dpi=170)
    plt.close(fig)


def write_query_examples():
    gaps=numeric_csv(FOLDER/"query-stopping-gaps.csv")
    own=[r for r in gaps if r["top_k"]==1]
    nonzero=[r for r in own if r["extra_documents_scored"]>0]
    median=float(np.median([r["extra_documents_scored"] for r in nonzero]))
    delayed=min(nonzero,key=lambda r:(abs(r["extra_documents_scored"]-median),r["query"]))
    immediate=min((r for r in own if r["stop_depth"]==32),key=lambda r:r["query"])
    chosen={int(r["query"]):r for r in [immediate,delayed]}
    text=["# Two measured prefix paths", "",
          "Both examples use K = 1 and the same fixed 78 clusters. The correct-answer column is calculated afterward; it is not available to the live search.", "",
          "The first example is the lowest query ID that permits an exact stop at depth 32. The second is the query nearest the median nonzero extra-scoring count. Neither is selected for winning a timing comparison.", ""]
    empty_work=[]
    with gzip.open(FOLDER/"query-traces.jsonl.gz","rt") as source:
        for line in source:
            row=json.loads(line)
            query=np.array(row["query_values"],dtype=np.float64)
            l1=float(np.abs(query).sum())
            steps=row["steps"]
            for k in KS:
                stop=next(s for s in steps if allows_stop(s,l1,len(query),k))
                visited=[s for s in steps if s["depth"]>=stop["depth"]]
                empty_work.append(dict(query=row["query"],top_k=k,
                    empty_depths=sum(s["new_documents_scored"]==0 for s in visited),
                    lookups_at_empty_depths=sum(s["new_prefix_lookups"] for s in visited if s["new_documents_scored"]==0),
                    total_lookups=stop["prefix_lookups"]))
            if row["query"] not in chosen:
                continue
            gap=chosen[row["query"]]
            keep={32,24,16,8,4,3,2,1,0,int(gap["correct_depth"]),int(gap["stop_depth"])}
            pattern="".join("1" if q>=0 else "0" for q in query)
            text += [f"## Query {row['query']}","",
                     f"Ideal code: `{pattern}`. Q = {l1:.6f}. The exact local best document ID is {row['local_reference'][0]}.","",
                     "Selected depths are shown below. The normal search stops at the final row shown; the diagnostic continued to the root only to check the answers.","",
                     "| Depth | New rows at this depth | Total scored | Correct / 1 | Best score so far | Unseen score bound | Decision |",
                     "|---:|---:|---:|---:|---:|---:|---|"]
            for step in steps:
                if step["depth"] not in keep or step["depth"]<gap["stop_depth"]:
                    continue
                correct=int(step["rows"][:1]==row["local_reference"][:1])
                score=f"{step['scores'][0]:.6f}" if step["scores"] else "No result"
                bound=f"{step['upper_bound']:.6f}" if step["upper_bound"] is not None else "No unseen rows"
                decision="Stop" if allows_stop(step,l1,len(query),1) else "Broaden"
                text.append(f"| {step['depth']} | {step['new_documents_scored']} | {step['documents_scored']} | {correct} | {score} | {bound} | {decision} |")
            text += ["",f"The correct answer first appeared at depth {int(gap['correct_depth'])}; the normal exact search stopped at depth {int(gap['stop_depth'])}. It scored {int(gap['extra_documents_scored']):,} additional documents between those points.",""]
    (FOLDER/"QUERY_EXAMPLES.md").write_text("\n".join(text)+"\n")
    write_csv(FOLDER/"empty-depth-work.csv",empty_work)


def results_note(rows):
    ks = sorted({row["top_k"] for row in rows})
    text=["# K and prefix experiment results", "", "Same Quora documents, queries and Qwen 32-bit binary score. These results are separate from the paper.", ""]
    for target in TARGETS:
        text += [f"## Required average recall: {target:.0%}","",
                 "Each cell is median ms / achieved recall. Times include routing. Selected configurations were repeated three times.","",
                 "| K | A: scan | B: Bitplanes | C: Backward Walk |", "|---:|---:|---:|---:|"]
        for k in ks:
            cells=[]
            for method,_,_ in METHODS:
                r=next((r for r in rows if r["top_k"]==k and r["target"]==target and r["method"]==method and r["status"]=="complete"),None)
                cells.append(f"{r['process_p50_median_ms']:.4f} / {r['recall']:.2%}" if r else "No qualified repeat")
            text.append(f"| {k} | "+" | ".join(cells)+" |")
        text.append("")
    text += ["## Files", "", "- `comparison.json`: selected parameters, achieved recall, times, repeat spread and memory.",
             "- `k-latency.png`: the K comparison.", "- `depth/`: same-cluster traces, normal local timings and plots.",
             "", "A small difference between overlapping timing repeats is not a clear win. A short-code binary recall result is not a claim about full-embedding or human relevance quality."]
    (OUTPUT/"RESULTS.md").write_text("\n".join(text)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies",nargs="+",type=Path,default=[OUTPUT/"sweep"])
    parser.add_argument("--depth",action="store_true")
    args=parser.parse_args()
    rows=comparisons(args.studies)
    plot_k(rows)
    results_note(rows)
    if args.depth:
        plot_depth(summarize_local())
        write_query_examples()


if __name__=="__main__":
    main()
