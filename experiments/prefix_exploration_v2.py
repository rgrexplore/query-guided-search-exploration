"""Prepare small search batches so a slow setting does not hold up the whole study."""

import argparse
from collections import Counter
from pathlib import Path

from experiments.fixed_data_comparison import save_json
from experiments.prefix_study_v2 import prepare, read_json


def split_schedule(jobs, batch_size=3):
    """Keep every setting and query; only change process boundaries and run order."""
    batches = []
    for job in jobs:
        # A global branch can be much slower than a clustered search. Give each
        # of those settings its own checkpoint instead of grouping several.
        size = 1 if job["clusters"] == 1 else batch_size
        for start in range(0, len(job["variants"]), size):
            batches.append(dict(job, job_id=f"{job['job_id']}-part{start // size:03d}",
                                variants=job["variants"][start:start + size]))

    # Finish the scan references first. Then alternate small batches from the
    # different indexes. Try finer layouts first within each round: the earlier
    # sweeps showed that global and coarse scans can consume most of the allowance.
    batches.sort(key=lambda job: (job["method"] != "scan",
                                  int(job["job_id"].rsplit("part", 1)[1]),
                                  -job["clusters"],
                                  job["job_id"]))
    before = Counter(v["setting_id"] for job in jobs for v in job["variants"])
    after = Counter(v["setting_id"] for job in batches for v in job["variants"])
    assert before == after and all(count == 1 for count in after.values())
    return batches


def prepare_exploration(config_path, output, batch_size=3):
    output = Path(output)
    jobs = prepare(read_json(config_path), output)
    save_json(output / "main/schedule-before-batching.json", jobs)
    batches = split_schedule(jobs, batch_size)
    save_json(output / "main/schedule.json", batches)
    save_json(output / "batching.json", dict(
        original_jobs=len(jobs), jobs=len(batches),
        settings=sum(len(job["variants"]) for job in batches),
        maximum_clustered_settings_per_job=batch_size,
        scope="Same settings, documents and queries; smaller independent processes. "
              "A time-limited process is retained as incomplete, not a measured loss."))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    jobs = prepare_exploration(args.configuration, args.output)
    print(f"Prepared {len(jobs)} small batches", flush=True)


if __name__ == "__main__":
    main()
