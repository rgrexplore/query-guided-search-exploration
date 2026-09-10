"""Check the standalone measurement code separately from the Python extension."""
import csv
import io
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    binary = tmp_path_factory.mktemp('cost-kernel') / 'cost-kernels'
    subprocess.run(['clang++', '-std=c++20', '-O3', '-DNDEBUG', '-Icpp',
                    'experiments/cost_kernels.cpp', '-o', str(binary)], cwd=ROOT, check=True)
    return binary


def records(kernel, *arguments):
    result = subprocess.run([str(kernel), *map(str, arguments)], capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def test_kernel_checks_against_bitwise_and_direct_score_references(kernel):
    result = subprocess.run([str(kernel), '--self-test'], capture_output=True, text=True, check=True)
    assert 'checks passed' in result.stdout


def test_path_keeps_alternatives_and_prices_parent_child_overlap(kernel):
    rows = records(kernel, 'path', 16, 3, 2, 79)
    assert len(rows) == 12
    splits = [r for r in rows if r['operation'] == 'split' and r['repetition'] == '0']
    assert [int(r['live_payload_bytes']) for r in splits] == [768, 896, 1024]
    counts = [int(r['count']) for r in splits]
    assert counts == sorted(counts, reverse=True)
    assert [int(r['size']) for r in splits] == [16, 16, 16]


def test_fresh_candidates_change_while_warm_control_reuses_them(kernel):
    warm = records(kernel, 'gather', 129, 256, 4, 100, 3, 79)
    fresh = records(kernel, 'gather_fresh', 129, 256, 4, 100, 3, 79)
    assert len({r['checksum'] for r in warm}) == 1
    assert len({r['checksum'] for r in fresh}) > 1
    assert all(0 <= int(r['count']) <= 129 for r in warm + fresh)
