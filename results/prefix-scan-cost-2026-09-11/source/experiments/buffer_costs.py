"""Measure bitmap paths and the shared scorer at explicitly chosen buffer sizes.

This macOS profile records whole-process peak RAM. Its kernel timings are not
end-to-end retrieval measurements and do not estimate recall.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess

from experiments.worker import power_state

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'build/cost-kernels'
    compile_command = ['clang++', '-std=c++20', '-O3', '-DNDEBUG', '-Wall', '-Wextra',
                       '-Wpedantic', '-Icpp', 'experiments/cost_kernels.cpp', '-o', str(binary)]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    check = subprocess.run([str(binary), '--self-test'], capture_output=True, text=True, check=True)
    (output / 'self-test.txt').write_text(check.stdout)
    sources = ['experiments/cost_kernels.cpp', 'experiments/buffer_costs.py', 'cpp/score.hpp', 'cpp/index.hpp']
    for name in sources:
        saved = output / 'source' / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((ROOT / name).read_bytes())
    config = dict(words=args.words, depth=args.depth, rows=args.rows, dimensions=args.dimensions,
                  top_k=args.top_k, repeats=args.repeats, seed=args.seed,
                  gather_pools=args.gather_pools, gather_depths=args.gather_depths, fresh_subsets=args.fresh_subsets)
    metadata = dict(config=config, source_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(), compile_command=compile_command, source_sha256={p:digest(ROOT/p) for p in sources},
                    binary_sha256=digest(binary),
                    compiler=subprocess.run(['clang++', '--version'], capture_output=True, text=True).stdout,
                    hardware=subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], capture_output=True, text=True).stdout,
                    scope='Kernel-only costs. Timed path retains alternative masks and includes parent release. Leaf measures enumeration/checksum only. Scorer includes table preparation, full scores and top-K selection. Gather includes bitmap enumeration and scattered document/ID reads. gather_fresh selects a different bitmap before every trial; plain gather reuses the warmed bitmap. Input generation and partition verification are untimed. Warmup path omitted; independent process confidence intervals are not claimed.')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    cases = [['path', str(words), str(args.depth), str(args.repeats), str(args.seed)] for words in args.words]
    cases += [['score', str(rows), str(dim), str(args.top_k), str(args.repeats), str(args.seed)]
              for rows in args.rows for dim in args.dimensions]
    cases += [['gather_fresh' if args.fresh_subsets else 'gather', str(rows), str(dim), str(depth), str(args.top_k), str(args.repeats), str(args.seed)]
              for rows in args.gather_pools for dim in args.dimensions for depth in args.gather_depths]
    random.Random(args.seed).shuffle(cases)
    (output / 'schedule.json').write_text(json.dumps(cases, indent=2) + '\n')
    summaries = []
    for number, case in enumerate(cases):
        folder = output / f'{number:03d}'
        folder.mkdir()
        before = power_state()
        command = ['/usr/bin/time', '-l', str(binary), *case]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (folder / 'measurements.csv').write_text(completed.stdout)
        (folder / 'process.txt').write_text(completed.stderr)
        after = power_state()
        match = re.search(r'(\d+)\s+maximum resident set size', completed.stderr)
        status = dict(command=command, returncode=completed.returncode,
                      peak_rss_bytes=int(match.group(1)) if match else None,
                      power_before=before, power_after=after)
        (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        summaries.append(dict(case=number, **status))
        (output / 'processes.json').write_text(json.dumps(summaries, indent=2) + '\n')
        if completed.returncode != 0 or match is None:
            raise RuntimeError(f'case {number} did not produce valid OS statistics; see {folder}')
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        print(f"{number+1}/{len(cases)} {' '.join(case)}: {len(rows)} measurements, peak {status['peak_rss_bytes']/1e9:.3f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--words', nargs='+', type=int, default=[64, 1024, 15625, 156250, 1562500, 15625000])
    parser.add_argument('--depth', type=int, default=13)
    parser.add_argument('--rows', nargs='+', type=int, default=[128, 512, 2048, 8192, 65536, 1000000, 4000000])
    parser.add_argument('--dimensions', nargs='+', type=int, default=[128, 256, 512, 768])
    parser.add_argument('--fresh-subsets', action='store_true')
    parser.add_argument('--gather-pools', nargs='+', type=int, default=[])
    parser.add_argument('--gather-depths', nargs='+', type=int, default=[8, 10, 12, 13])
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=79)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
