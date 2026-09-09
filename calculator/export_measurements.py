"""Keep the browser's evidence small. Run from the project root."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/scaling-msmarco-1m/analysis/settings.csv'
FIELDS = ['pool_size', 'node_budget', 'mean_recall', 'api_p50_ms', 'api_p95_ms',
          'mean_documents_scored', 'mean_nodes', 'mean_bitplane_words', 'native_peak_rss_bytes']


def positive_fit(features, target):
    # Only two coefficients here. Try each nonnegative combination; no optimizer needed.
    best = None
    for count in range(1, features.shape[1] + 1):
        for columns in itertools.combinations(range(features.shape[1]), count):
            values = np.linalg.lstsq(features[:, columns], target, rcond=None)[0]
            if np.any(values < 0):
                continue
            coefficients = np.zeros(features.shape[1])
            coefficients[list(columns)] = values
            error = np.sum((features @ coefficients - target) ** 2)
            if best is None or error < best[0]:
                best = (error, coefficients)
    return best[1]


rows = []
for raw in csv.DictReader(SOURCE.open()):
    if raw['phase'] == 'pilot' or raw['status'] != 'complete':
        continue
    row = {key: float(raw[key]) for key in FIELDS}
    row.update(phase=raw['phase'], method=raw['method'])
    rows.append(row)

# Fit on validation only. Held-out test rows stay available for an independent check.
scan = [r for r in rows if r['phase'] == 'development' and r['method'] == 'scan']
x = np.array([[1, r['pool_size'] * 64 / 1e6] for r in scan])
scan_times = np.array([r['api_p50_ms'] for r in scan])
fixed_ms, score_ns = positive_fit(x / scan_times[:, None], np.ones(len(scan)))
branch = [r for r in rows if r['phase'] == 'development' and r['method'] == 'branch']
x = np.array([[r['mean_nodes'] * np.ceil(r['pool_size'] / 64) / 1e6,
               r['mean_nodes'] / 1e6] for r in branch])
y = np.array([r['api_p50_ms'] - fixed_ms - r['mean_documents_scored'] * 64 * score_ns / 1e6
              for r in branch])
branch_times = np.array([r['api_p50_ms'] for r in branch])
word_ns, node_ns = positive_fit(x / branch_times[:, None], y / branch_times)
output = {
    'source': '../results/scaling-msmarco-1m/analysis/settings.csv',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'scope': 'MS MARCO, Nomic v1.5, 256 bits, top 100, leaf 32, exploration 0, no routing, one CPU thread',
    'hardware': 'Apple M5 Max · 128 GiB RAM',
    'calibration': {'fixedMs': float(fixed_ms), 'scoreNs': float(score_ns),
                    'wordNs': float(word_ns), 'nodeNs': float(node_ns)},
    'rows': rows,
}
(ROOT / 'calculator/measurements.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output['calibration'], indent=2))
print(f'Exported {len(rows)} rows from {SOURCE.relative_to(ROOT)}')
