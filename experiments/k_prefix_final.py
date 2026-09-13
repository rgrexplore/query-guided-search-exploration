"""Measure the final selected configurations together in three fresh timing blocks."""

import json
from collections import defaultdict
from pathlib import Path

from experiments.fixed_data_comparison import save_json
from experiments.k_prefix_study import OUTPUT
from experiments.prefix_exploration_v2 import split_schedule
from experiments.prefix_study_v2 import read_json, repeat
from experiments.k_prefix_results import plot_k, results_note, plot_depth, summarize_local, write_query_examples

FOLDER = OUTPUT / "final"
SEARCH_FIELDS = ["top_k","probes","leaf_size","node_budget","prefer_deeper_ties",
                 "exploration","seed","start_depth","candidate_target","stop_when_exact"]


def prepare():
    points = read_json(OUTPUT / "comparison.json")
    assert all(row["status"] == "complete" for row in points)
    source = OUTPUT / "sweep"
    config = read_json(source / "configuration.json")
    manifest = read_json(source / "inputs-manifest.json")
    shape = manifest["identity"]
    layouts = {}
    for study in {row["study"] for row in points}:
        assert read_json(Path(study)/"inputs-manifest.json") == manifest
        for layout in read_json(Path(study)/"layouts.json"):
            layouts[layout["path"]] = layout
    FOLDER.mkdir()
    (FOLDER/"main").mkdir()
    save_json(FOLDER/"configuration.json",dict(config,
        top_ks=sorted({r["top_k"] for r in points}), schedule_seed=73,global_seconds=1800,
        scope="main contains selection records from completed studies; repeat contains the new shared timing round"))
    save_json(FOLDER/"inputs-manifest.json",manifest)
    save_json(FOLDER/"layouts.json",list(layouts.values()))
    unique = {}
    selections = []
    grouped = defaultdict(list)
    for point in points:
        options = {name:point[name] for name in SEARCH_FIELDS if name in point}
        key = (point["method"],point["router"],json.dumps(options,sort_keys=True))
        if key not in unique:
            identifier=f"final-s{len(unique):04d}"
            unique[key]=identifier
            grouped[(point["method"],point["router"],point["clusters"])].append(
                dict(setting_id=identifier,repetitions=1,**options))
        selections.append(dict(point,setting_id=unique[key],
            source_study=point["study"],source_setting_id=point["setting_id"],study=str(FOLDER)))
    jobs=[]
    for (method,router,clusters),variants in grouped.items():
        jobs.append(dict(job_id=f"final-j{len(jobs):03d}",phase="main",pool=config["pool"],
            router=router,clusters=clusters,method=method,dimensions=shape["dimensions"],
            documents=shape["documents"],query_rows=list(range(shape["queries"])),
            max_prefix_bits=32,ram_budget_bytes=config["ram_budget_bytes"],variants=variants))
    save_json(FOLDER/"main/schedule.json",split_schedule(jobs,batch_size=6))
    save_json(FOLDER/"selected-settings.json",selections)
    print(f"Selected {len(unique)} distinct configurations for the final timing round",flush=True)


def run():
    if not FOLDER.exists():
        prepare()
    repeated={row["setting_id"]:row for row in repeat(FOLDER)}
    output=[]
    for choice in read_json(FOLDER/"selected-settings.json"):
        row=repeated[choice["setting_id"]]
        assert row["qualified"] and row["recall"] >= choice["target"],(choice,row)
        output.append(dict(row,target=choice["target"],status="complete",study=str(FOLDER),
            source_study=choice["source_study"],source_setting_id=choice["source_setting_id"],
            selection_p50_ms=choice["selection_p50_ms"]))
    save_json(OUTPUT/"comparison.json",output)
    plot_k(output)
    results_note(output)
    plot_depth(summarize_local())
    write_query_examples()
    save_json(FOLDER/"complete.json",dict(selected=len(repeated),points=len(output),
        all_qualified=True,all_targets_met=True))
    print(f"Final comparison complete: {len(output)} target/method/K points",flush=True)


if __name__ == "__main__":
    run()
