"""Exercise the real subprocess boundary and inspect the evidence it writes."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.components import pack_signs, reference_rows
from experiments.isolated import run_isolated


@pytest.mark.parametrize('method', ['scan', 'branch', 'keys'])
def test_worker_runs_one_method_in_a_fresh_process(tmp_path, method):
    rng = np.random.default_rng(6)
    signs = rng.integers(0, 2, (65, 5), dtype=np.uint8)
    queries = rng.normal(size=(3, 5)).astype(np.float32)
    pool = tmp_path/'pool'
    pool.mkdir()
    np.save(pool/'codes.npy', pack_signs(signs))
    np.save(pool/'queries.npy', queries)
    np.save(pool/'reference.npy', reference_rows(signs, queries, 3))
    case = dict(method=method, pool=str(pool), dimensions=5, documents=65, top_k=3,
                router=None, probes=1, repetitions=1, query_rows=[0, 1, 2],
                node_budget=0, leaf_size=4, key_bits=3, key_offset=0,
                candidate_target=0, key_limit=0, ram_budget_bytes=32_000_000_000)
    case_path = tmp_path/'case.json'
    case_path.write_text(json.dumps(case))
    output = tmp_path/'case-output'
    run = subprocess.run([sys.executable, '-m', 'experiments.worker', str(case_path), str(output)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'complete'
    assert result['pid'] != os.getpid()
    assert result['memory']['rss_before_queries'] > result['storage']['logical_bytes']
    assert result['memory']['lifetime_peak_bytes'] > 0
    assert result['memory']['budget_status'] == 'fits_by_lifetime_peak'
    rows = [json.loads(line) for line in (output/'queries.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and all(row['recall'] == 1 for row in rows)
    assert all(row['query_ms'] >= row['search_ms'] for row in rows)
    assert result['storage']['bitplanes_bytes'] == (80 if method=='branch' else 0)


def test_parent_preserves_configuration_and_each_worker_result(tmp_path):
    config = {
        'data': {'kind': 'random_signs', 'cache_dir': str(tmp_path/'pools'),
                 'sizes': [16], 'dimensions': 5, 'queries': 2, 'seed': 9},
        'search': {'top_k': 3},
        'sweep': {'node_budgets': [0], 'leaf_sizes': [4], 'key_bits': [2],
                  'candidate_targets': [0], 'key_limit': 0},
        'measurement': {'repetitions': 1, 'schedule_seed': 3},
        'limits': {'ram_budget_bytes': 32_000_000_000, 'worker_stop_bytes': 34_359_738_368,
                   'case_seconds': 20},
    }
    output = tmp_path/'study'
    run_isolated(config, output)
    assert json.loads((output/'configuration.json').read_text()) == config
    results = [json.loads(path.read_text()) for path in output.glob('cases/*/run/result.json')]
    assert len(results) == 3
    assert len({result['pid'] for result in results}) == 3
    assert all(result['status']=='complete' and result['exact_checks']==2 for result in results)
