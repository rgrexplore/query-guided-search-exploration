"""Independent small checks for the equations and numerical examples in the report."""
import csv
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def check_scores():
    q = [.7, -.4, .2, -.1]
    codes = ['1010', '1011', '1000', '0010', '1110', '0101', '1001', '0011']
    expected = [1.4, 1.2, 1.0, 0, .6, -1.4, .8, -.2]
    for code, value in zip(codes, expected):
        signs = [1 if bit == '1' else -1 for bit in code]
        direct = sum(x*y for x, y in zip(q, signs))
        penalty = sum(abs(x) for x, y in zip(q, signs) if (x >= 0) != (y > 0))
        assert math.isclose(direct, sum(map(abs, q))-2*penalty, abs_tol=1e-12)
        assert math.isclose(direct, value, abs_tol=1e-12)
        # A grouped lookup has the same score as the direct sum.
        grouped = sum(sum(q[i]*signs[i] for i in range(start, min(start+2, len(q))))
                      for start in range(0, len(q), 2))
        assert math.isclose(grouped, direct, abs_tol=1e-12)
    order = sorted(range(8), key=lambda i: (-expected[i], i))
    assert order[:2] == [0, 1]
    selected = [1, 4, 6, 7]
    local = sorted(selected, key=lambda i: (-expected[i], i))[:2]
    assert local == [1, 6]
    assert len(set(local) & set(order[:2]))/2 == .5
    planes = [''.join(codes[i][dim] for i in selected) for dim in range(4)]
    assert planes == ['1110', '0100', '1101', '1011']


def subset_stream(weights):
    yield (0, 0)
    if not weights:
        return
    queue = [(weights[0], 1, 0)]
    while queue:
        cost, mask, last = heapq.heappop(queue)
        yield cost, mask
        if last+1 < len(weights):
            nxt = last+1
            heapq.heappush(queue, (cost+weights[nxt], mask | (1 << nxt), nxt))
            heapq.heappush(queue, (cost-weights[last]+weights[nxt],
                                  (mask ^ (1 << last)) | (1 << nxt), nxt))


def check_subsets():
    for weights in [[], [0, 0, 1, 2], [3, 4, 5, 6], [1, 2, 2, 4, 7, 9]]:
        emitted = list(subset_stream(weights))
        assert len(emitted) == 2**len(weights)
        assert len({mask for _, mask in emitted}) == 2**len(weights)
        assert [cost for cost, _ in emitted] == sorted(cost for cost, _ in emitted)
        for cost, mask in emitted:
            assert cost == sum(w for i, w in enumerate(weights) if mask >> i & 1)
    # Full-key ordering including the final score tie equals exhaustive ranking.
    q = [7, 5, -4, -6, 3, -1, 1, -2]
    ids = sorted(range(len(q)), key=lambda i: abs(q[i]))
    weights = [abs(q[i]) for i in ids]
    ideal = sum((value >= 0) << i for i, value in enumerate(q))
    rows = []
    for penalty, mask in subset_stream(weights):
        key = ideal
        for position, i in enumerate(ids):
            if mask >> position & 1:
                key ^= 1 << i
        rows.append((penalty, key))
    got = [key for _, key in sorted(rows)[:100]]
    scores = [(sum(v*(1 if code >> i & 1 else -1) for i, v in enumerate(q)), code)
              for code in range(256)]
    exact = [code for _, code in sorted(scores, key=lambda r: (-r[0], r[1]))[:100]]
    assert got == exact


def check_binomial():
    d, h, full_radius, prefix_radius = 6, 2, 1, 0
    patterns = list(itertools.product([0, 1], repeat=d))
    truth = [p for p in patterns if sum(p) <= full_radius]
    candidates = [p for p in patterns if sum(p[:h]) <= prefix_radius]
    actual = len(set(truth) & set(candidates)) / len(truth)
    denominator = sum(math.comb(d, r) for r in range(full_radius+1)) / 2**d
    numerator = sum(math.comb(h, a)/2**h *
                    sum(math.comb(d-h, b) for b in range(max(-1, full_radius-a)+1)) / 2**(d-h)
                    for a in range(min(h, full_radius, prefix_radius)+1))
    assert len(truth) == 7 and len(candidates) == 16
    assert math.isclose(actual, 5/7)
    assert math.isclose(numerator/denominator, actual)


def check_memory_and_evidence():
    assert 10**9 * 256 // 8 == 32 * 10**9
    assert 10**10 * 256 // 8 == 320 * 10**9
    assert math.ceil(1e6/64) == 15625
    assert 15625*8 == 125000
    path = HERE/'evidence/scaling-settings.csv'
    rows = list(csv.DictReader(path.open()))
    branch = next(r for r in rows if r['phase']=='evaluation' and r['pool_size']=='1000000'
                  and r['method']=='branch' and r['node_budget']=='32768')
    scan = next(r for r in rows if r['phase']=='evaluation' and r['pool_size']=='1000000'
                and r['method']=='scan')
    assert math.isclose(float(branch['api_p50_ms']), 244.98483282513916)
    assert math.isclose(float(scan['api_p50_ms']), 25.13647940941155)
    split_words = float(branch['mean_bitplane_words'])
    total_words = float(branch['mean_nodes'])*15625
    assert split_words == 352466875
    assert total_words-split_words == 159533125
    assert math.isclose(float(branch['mean_recall']), .9926)
    study_path = HERE/'evidence/recall-validation.json'
    study = json.loads(study_path.read_text())
    assert sum(case['exact_reference_checks'] for case in study['cases']) == 96
    recalls = []
    for case in study['cases']:
        row = next(r for r in case['summaries'] if r['method']=='C' and r['target']==1000)
        recalls.append(row['measured_recall'])
    for actual, expected in zip(recalls, [.23, .95, .2525]):
        assert math.isclose(actual, expected)
    fiqa_path = HERE/'evidence/fiqa-summary.csv'
    fiqa = list(csv.DictReader(fiqa_path.open()))
    assert len(fiqa) == 583
    assert sum(int(r['measurements']) for r in fiqa) == 1133352
    example = next(r for r in fiqa if r['method']=='sign/branch' and r['probes']=='32' and r['node_budget']=='2048' and float(r['explore_probability'])==0)
    assert abs(float(example['float_recall'])-.7006) < .00005
    assert abs(float(example['retrieval_p50_ms'])-.6256) < .00005
    return {'fiqa_csv_sha256': hashlib.sha256(fiqa_path.read_bytes()).hexdigest(), 'scaling_csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'recall_study_sha256': hashlib.sha256(study_path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    check_scores()
    check_subsets()
    check_binomial()
    evidence = check_memory_and_evidence()
    result = {'score_identity_and_lookup': 'passed', 'complete_weighted_subset_order': 'passed',
              'finite_binary_recall_example': 'passed', 'memory_and_archived_numbers': 'passed', **evidence}
    (HERE/'evidence/math-checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
