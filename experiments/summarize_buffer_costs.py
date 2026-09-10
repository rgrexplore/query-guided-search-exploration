"""Summarize kernel costs and check a leaf-cost fit on a larger document buffer."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls


def summarize(folders, output, operation="gather"):
    output.mkdir(parents=True, exist_ok=False)
    all_rows, processes = [], []
    for folder in folders:
        metadata = json.loads((folder / 'metadata.json').read_text())
        for process in json.loads((folder / 'processes.json').read_text()):
            assert process['returncode'] == 0 and process['peak_rss_bytes'] is not None
            assert process['power_before']['available'] and process['power_before'] == process['power_after']
            number = process['case']
            processes.append(dict(source=str(folder), **process))
            with (folder / f'{number:03d}' / 'measurements.csv').open() as file:
                for row in csv.DictReader(file):
                    all_rows.append(dict(source=str(folder), case=number, binary=metadata['binary_sha256'],
                                         operation=row['operation'], size=int(row['size']),
                                         dimensions=int(row['dimensions']), repetition=int(row['repetition']),
                                         step=int(row['step']), milliseconds=float(row['milliseconds']),
                                         count=int(row['count']), live_payload_bytes=int(row['live_payload_bytes'])))
    groups = {}
    for row in all_rows:
        key = (row['source'], row['case'], row['operation'], row['step'])
        groups.setdefault(key, []).append(row)
    rows = []
    for values in groups.values():
        first = values[0]
        times = [r['milliseconds'] for r in values]
        rows.append(dict(source=first['source'],case=first['case'],operation=first['operation'],
                         size=first['size'], dimensions=first['dimensions'],step=first['step'],
                         count=statistics.median(r['count'] for r in values),p50_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                         live_payload_bytes=first['live_payload_bytes']))
    with (output / 'kernels.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Fit only the 1M/4M document buffers. The 16M observations stay excluded.
    gather = [r for r in all_rows if r['operation'] == operation and r['dimensions'] == 256]
    train = [r for r in gather if r['size'] in (1000000, 4000000)]
    check = [r for r in rows if r['operation'] == operation and r['size'] == 16000000]
    features = np.array([[1, (r['size']+63)//64, r['count']] for r in train], dtype=float)
    observed = np.array([r['milliseconds'] for r in train])
    scales = np.linalg.norm(features/observed[:,None], axis=0)
    coefficients, _ = nnls(features/scales/observed[:,None], np.ones(len(observed)))
    coefficients /= scales
    predictions = []
    for row in check:
        predicted = float(np.array([1, (row['size']+63)//64, row['count']]) @ coefficients)
        predictions.append(dict(**row, predicted_ms=predicted,
                                error_percent=100*(predicted/row['p50_ms']-1)))
    result = dict(operation=operation, training_sizes=[1000000,4000000],check_size=16000000,
                  coefficients=dict(zip(['fixed_ms','per_bitmap_word_ms','per_scored_row_ms'], coefficients.tolist())),
                  matrix_rank=int(np.linalg.matrix_rank(features/scales)), predictions=predictions,
                  scope='Combined leaf cost: preparation, bitmap traversal, scattered scoring and top-K. Repeated timings are not independent queries. This is not a 40GB-buffer validation or a full-search latency model.')
    (output / 'leaf-model.json').write_text(json.dumps(result, indent=2)+'\n')
    (output / 'processes.json').write_text(json.dumps(processes, indent=2)+'\n')
    lines = ['# Buffer-size cost measurements', '',
             'These are kernel measurements on constructed input buffers, not a retrieval-quality benchmark.',
             'Raw elapsed times include the work named by each operation; input generation is excluded.', '',
             '## Leaf model checked on a larger buffer', '',
             'Fit: fixed cost + physical bitmap words × word cost + scored rows × gathered-row cost.',
             'Only 1M/4M buffers fit the coefficients; the 16M buffer checks their transfer.', '',
             '| Document pool | Remaining rows | Measured ms | Predicted ms | Error |',
             '|---:|---:|---:|---:|---:|']
    for row in predictions:
        lines.append(f"| {row['size']:,} | {row['count']:,} | {row['p50_ms']:.4f} | {row['predicted_ms']:.4f} | {row['error_percent']:+.1f}% |")
    lines += ['', 'All source tables, including split/root/release times and OS peaks, are retained.',
              'A path profile holds only the measured planes and masks. Its RAM is not the full d-dimensional index RAM.',
              'The 32GB case cannot hold 1B 256-bit codes plus 64-bit document IDs: that payload alone needs 40GB.',
              'Larger-RAM scenarios still require a separate full-query model and explicit extrapolation limits.']
    (output / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--operation', choices=['gather', 'gather_fresh'], default='gather')
    args=parser.parse_args()
    summarize(args.folders,args.output,args.operation)


if __name__ == '__main__':
    main()
