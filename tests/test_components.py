"""Run the actual component study and inspect its saved numerical evidence."""
import csv

from experiments.components import run_components


def test_small_study_saves_exact_results_and_independent_storage_counts(tmp_path):
    config = {
        'data': {'kind': 'random_signs', 'sizes': [5, 65], 'dimensions': 5, 'queries': 2, 'seed': 7},
        'search': {'top_k': 3, 'key_bits': 2, 'key_offset': 0, 'candidate_target': 0,
                   'key_limit': 0, 'node_budget': 0, 'leaf_size': 4},
        'measurement': {'repetitions': 1},
    }
    output = tmp_path/'observations'
    run_components(config, output)
    with (output/'measurements.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert all(float(row['recall']) == 1 for row in rows)
    with (output/'indexes.csv').open() as file:
        indexes = list(csv.DictReader(file))
    scan = next(row for row in indexes if row['method']=='scan' and row['documents']=='65')
    branch = next(row for row in indexes if row['method']=='branch' and row['documents']=='65')
    assert int(scan['logical_bytes']) == 1040
    assert int(branch['bitplanes_bytes']) == 80
    assert (output/'README.md').exists()
