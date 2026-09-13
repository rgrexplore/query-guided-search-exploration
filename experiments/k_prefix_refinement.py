"""Try nearby search settings after the first K sweep, without changing the data."""

import argparse
from collections import defaultdict

import numpy as np

from experiments.fixed_data_comparison import save_json
from experiments.k_prefix_study import OUTPUT, KS, TARGETS
from experiments.prefix_exploration_v2 import split_schedule
from experiments.prefix_study_v2 import read_json, run, summarize, repeat

FOLDER = OUTPUT / "refinement"


def prepare():
    source = OUTPUT / "sweep"
    config = read_json(source / "configuration.json")
    rows = read_json(source / "main/settings.json")
    chosen = defaultdict(set)
    reasons = []
    # Select routing layouts around the 90% and 99% comparisons. The lower-target
    # baseline settings remain available from the original sweep.
    for k in KS:
        for target in [.9, .99]:
            for method in ["scan", "branch", "prefix"]:
                eligible = [r for r in rows if r["top_k"]==k and r["method"]==method
                            and r["qualified"] and r["recall"]>=target]
                if not eligible:
                    continue
                best = min(eligible, key=lambda r:r["p50_ms"])
                clusters, probes = best["clusters"], best["probes"]
                for nearby in [probes, min(clusters, max(probes+1, int(np.ceil(probes*1.5))))]:
                    chosen[(clusters,best["router"],k)].add(nearby)
                reasons.append(dict(top_k=k,target=target,method=method,
                                    source_setting_id=best["setting_id"]))
    FOLDER.mkdir()
    (FOLDER / "main").mkdir()
    save_json(FOLDER / "configuration.json",dict(config,global_seconds=7200,job_seconds=240,
        scope="Selected routing choices and 1.5x probes, shared by all methods. C also tests approximate count stops."))
    for name in ["inputs-manifest.json","layouts.json"]:
        save_json(FOLDER/name,read_json(source/name))
    save_json(FOLDER/"selection-reasons.json",reasons)
    jobs=[]
    number=0
    for (clusters,router,k),probes in sorted(chosen.items()):
        for method in ["scan","branch","prefix"]:
            if method=="scan":
                policies=[{}]
            elif method=="branch":
                policies=[dict(leaf_size=leaf,node_budget=budget,prefer_deeper_ties=deeper,
                               exploration=0,seed=42)
                          for leaf,budget,deeper in [(64,2048,True),(64,0,True),(128,0,True),
                                                     (512,8192,True),(2048,8192,False)]]
            else:
                policies=[dict(start_depth=d,candidate_target=0,stop_when_exact=True)
                          for d in [8,12,24]]
                targets=sorted({k,4*k,16*k,64*k,1000,10000})
                policies += [dict(start_depth=d,candidate_target=count,stop_when_exact=True)
                             for d in [4,16,32] for count in targets]
            variants=[]
            for count in sorted(probes):
                for policy in policies:
                    variants.append(dict(setting_id=f"refine-s{number:05d}",top_k=k,
                                         probes=count,repetitions=1,**policy))
                    number+=1
            np.random.default_rng(number).shuffle(variants)
            jobs.append(dict(job_id=f"refine-j{len(jobs):04d}",phase="main",pool=config["pool"],
                router=router,clusters=clusters,method=method,dimensions=32,documents=522931,
                query_rows=list(range(1000)),max_prefix_bits=32,
                ram_budget_bytes=config["ram_budget_bytes"],variants=variants))
    batches=split_schedule(jobs,batch_size=6)
    save_json(FOLDER/"main/schedule.json",batches)
    save_json(FOLDER/"planned-work.json",dict(settings=number,jobs=len(batches)))
    print(f"Prepared {number} refinement settings in {len(batches)} batches",flush=True)


def execute():
    if not FOLDER.exists():
        prepare()
    run(FOLDER)
    rows=summarize(FOLDER)
    repetitions=repeat(FOLDER)
    save_json(FOLDER/"complete.json",dict(qualified=sum(r["qualified"] for r in rows),
        scheduled=len(rows),repeated=sum(r["qualified"] for r in repetitions)))
    print(read_json(FOLDER/"complete.json"),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","run"])
    args=parser.parse_args()
    {"prepare":prepare,"run":execute}[args.stage]()


if __name__=="__main__":
    main()
