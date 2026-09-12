"""Continue the dated exploration after reference preparation releases the CPU."""

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
    save(RESULTS / "exploration-execution.json", records)
    print("DONE", round(records[-1]["seconds"], 2), "seconds", flush=True)


def run_with_repeat_allowance(output, main_seconds, repeat_seconds):
    config_path = output / "configuration.json"
    config = json.loads(config_path.read_text())
    save(output / "stage-allowances.json", dict(main_seconds=main_seconds,
        repeat_seconds=repeat_seconds, total_seconds=main_seconds + repeat_seconds))
    config["global_seconds"] = main_seconds
    save(config_path, config)
    command("experiments.prefix_study_v2", "run", output)
    command("experiments.prefix_study_v2", "summarize", output)
    config["global_seconds"] = main_seconds + repeat_seconds
    save(config_path, config)
    command("experiments.prefix_study_v2", "repeat", output)


while not (RESULTS / "short-pools-ready.json").exists():
    sleep(3)

# Each method sees all 1,000 queries. The allowance changes how many settings
# finish, never which queries are retained from a completed setting.
for name, main_seconds, repeat_seconds in [
    ("qwen-quora-d32", 1200, 600),
    ("nomic-msmarco-d64", 1800, 900),
    ("nomic-msmarco-d768-focused", 1800, 600),
]:
    output = RESULTS / name
    command("experiments.prefix_exploration_v2", CONFIGS / f"{name}.json", output)
    run_with_repeat_allowance(output, main_seconds, repeat_seconds)

for name in ["qwen-quora-d256", "nomic-msmarco-d256", "qwen-quora-d1024"]:
    output = RESULTS / f"{name}-followup"
    command("experiments.prefix_followups_v2", "prepare", RESULTS / name, output, "--seconds", 900)
    run_with_repeat_allowance(output, 600, 300)

save(RESULTS / "exploration-finished.json", dict(status="finished", stages=len(records)))
print("EXPLORATION FINISHED", flush=True)
