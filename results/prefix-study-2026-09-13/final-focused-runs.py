"""Run the remaining targeted controls after the exact-stop build is ready."""

import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/prefix-study-2026-09-13"
CONFIGS = ROOT / "configs/prefix-study-v2"
records = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(*arguments):
    started = monotonic()
    print("START", *map(str, arguments), flush=True)
    subprocess.run([sys.executable, "-m", *map(str, arguments)], cwd=ROOT, check=True)
    records.append(dict(arguments=list(map(str, arguments)), seconds=monotonic() - started))
    save(RESULTS / "final-focused-execution.json", records)
    print("DONE", round(records[-1]["seconds"], 2), "seconds", flush=True)


def run_and_repeat(output, main_seconds, total_seconds):
    path = output / "configuration.json"
    config = json.loads(path.read_text())
    config["global_seconds"] = main_seconds
    save(path, config)
    command("experiments.prefix_study_v2", "run", output)
    command("experiments.prefix_study_v2", "summarize", output)
    config["global_seconds"] = total_seconds
    save(path, config)
    command("experiments.prefix_study_v2", "repeat", output)


marker = RESULTS / "native-exact-stop-build.json"
while not marker.exists():
    sleep(3)
assert json.loads(marker.read_text())["status"] == "ready"

for name, main_seconds, total_seconds in [
    ("qwen-quora-d32-global-exact", 120, 300),
    ("qwen-quora-d32-binary", 900, 1200),
    ("nomic-msmarco-d64-binary", 900, 1200),
]:
    output = RESULTS / name
    command("experiments.prefix_exploration_v2", CONFIGS / f"{name}.json", output)
    schedule_path = output / "main/schedule.json"
    schedule = json.loads(schedule_path.read_text())
    save(output / "main/schedule-before-labels.json", schedule)
    # These runs reuse a pool. Distinct IDs let the final table combine all
    # tested routing choices without confusing two settings called main-s00000.
    for job in schedule:
        job["job_id"] = f"{name}-{job['job_id']}"
        for variant in job["variants"]:
            variant["setting_id"] = f"{name}-{variant['setting_id']}"
    save(schedule_path, schedule)
    run_and_repeat(output, main_seconds, total_seconds)

for name in ["qwen-quora-d32", "nomic-msmarco-d64"]:
    output = RESULTS / f"{name}-probability"
    command("experiments.prefix_probability_v2", "prepare", RESULTS / name, output)
    run_and_repeat(output, 180, 300)

save(RESULTS / "final-focused-finished.json", dict(status="finished", stages=len(records)))
print("FINAL FOCUSED RUNS FINISHED", flush=True)
