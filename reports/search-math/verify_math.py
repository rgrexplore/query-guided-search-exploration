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


def binomial_cdf(bits, radius):
    """Fraction of all bit patterns with at most radius mismatches."""
    return sum(math.comb(bits, errors) for errors in range(radius + 1)) / 2**bits


def expected_clipped_count(documents, probability, top_k):
    """Expected number of top-K slots filled by an event with this probability.

    This direct sum is for the report's small examples. Larger experiments use
    SciPy's stable binomial survival function instead of powers and factorials.
    """
    return sum(
        min(top_k, count) * math.comb(documents, count)
        * probability**count * (1 - probability)**(documents - count)
        for count in range(documents + 1)
    )


def prefix_survival(bits, prefix_bits, radius, full_errors):
    """Given the total errors, count which placements pass the prefix filter."""
    accepted = 0
    for prefix_errors in range(min(prefix_bits, radius) + 1):
        tail_errors = full_errors - prefix_errors
        if 0 <= tail_errors <= bits - prefix_bits:
            accepted += (math.comb(prefix_bits, prefix_errors)
                         * math.comb(bits - prefix_bits, tail_errors))
    return accepted / math.comb(bits, full_errors)


def expected_prefix_recall(documents, bits, prefix_bits, radius, top_k):
    """Exact expectation for fixed-radius lookup in the iid balanced-bit model."""
    expected_hits = 0.0
    previous_slots = 0.0
    for errors in range(bits + 1):
        slots = expected_clipped_count(
            documents, binomial_cdf(bits, errors), top_k
        )
        # The difference counts top-K rows at this distance, including tied rows.
        rows_at_distance = slots - previous_slots
        expected_hits += rows_at_distance * prefix_survival(
            bits, prefix_bits, radius, errors
        )
        previous_slots = slots
    return expected_hits / top_k


def check_finite_recall():
    """Compare probability formulas with actual sorted tiny corpora.

    Enumerating ordered corpora includes repeated codes. Row IDs break ties;
    they are assigned by position, independently of the sampled code.
    """
    checked = 0
    for documents, bits in [(1, 3), (3, 3)]:
        for prefix_bits in (0, 1, bits):
            for radius in sorted({0, prefix_bits}):
                for top_k in sorted({1, min(2, documents), documents}):
                    observed_sum = 0.0
                    for codes in itertools.product(range(2**bits), repeat=documents):
                        truth = sorted(range(documents),
                                       key=lambda row: (bin(codes[row]).count('1'), row))[:top_k]
                        prefix_mask = (1 << prefix_bits) - 1
                        found = sum(bin(codes[row] & prefix_mask).count('1') <= radius
                                    for row in truth)
                        observed_sum += found / top_k
                    observed = observed_sum / (2**bits)**documents
                    predicted = expected_prefix_recall(
                        documents, bits, prefix_bits, radius, top_k
                    )
                    assert math.isclose(predicted, observed, abs_tol=1e-12)
                    checked += 1
    examples = []
    for radius in (0, 1, 2, 4):
        keys = sum(math.comb(4, errors) for errors in range(radius + 1))
        recall = expected_prefix_recall(256, 16, 4, radius, 1)
        examples.append(dict(radius=radius, keys=keys,
                             expected_candidates=256*keys/16, recall=recall))
    assert math.isclose(prefix_survival(16, 4, 1, 2), .95)
    assert math.isclose(examples[1]['recall'], .8959991329718401, abs_tol=1e-12)
    assert math.isclose(examples[-1]['recall'], 1, abs_tol=1e-12)
    return {'enumerated_settings': checked, 'worked_example': examples}


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
    model_path = HERE/'evidence/recall-model-checks.json'
    model = json.loads(model_path.read_text())
    assert len(model['controlled']) == 4
    for case in model['controlled']:
        assert case['observed']['samples'] == 10000
        exact_runs = [row for row in case['native_queries'] if row['budget'] == 0]
        assert len(exact_runs) == 12 and all(row['recall'] == 1 for row in exact_runs)
    assert model['archived']['source_sha256'] == hashlib.sha256(study_path.read_bytes()).hexdigest()
    return {'model_checks_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'fiqa_csv_sha256': hashlib.sha256(fiqa_path.read_bytes()).hexdigest(), 'scaling_csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'recall_study_sha256': hashlib.sha256(study_path.read_bytes()).hexdigest()}


def check_implemented_study():
    """Check quoted rows and conditional work against the frozen evidence copies."""
    tables = {}
    hashes = {}
    for kind in ('real', 'fixed', 'adaptive'):
        for name in ('targets', 'comparisons'):
            path = HERE / 'evidence' / f'{kind}-{name}.csv'
            with path.open() as file:
                tables[(kind, name)] = list(csv.DictReader(file))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    real = tables[('real', 'targets')]
    for selection, expected_passes in [('mean', 7), ('conservative', 36)]:
        rows = [r for r in real if r['selection'] == selection]
        assert len(rows) == 36
        assert sum(r['met_target'] == 'True' for r in rows) == expected_passes
    assert not any(r['supported_speedup'] == 'True' for r in tables[('real', 'comparisons')])

    # These are the N=1M, 99% conservative choices plotted in the report.
    expected_times = {'real': [9.5796, 9.9645, 11.0127],
                      'fixed': [.0145, .0156, .0127],
                      'adaptive': [24.9377, .1496, 26.8429]}
    for kind, times in expected_times.items():
        rows = {r['method']: r for r in tables[(kind, 'targets')]
                if r['documents'] == '1000000' and r['target'] == '0.99'
                and r['selection'] == 'conservative'}
        for method, expected in zip(('scan', 'branch', 'keys'), times):
            assert abs(float(rows[method]['p50_ms']) - expected) < .00005
            assert rows[method]['met_target'] == rows[method]['qualified_environment'] == 'True'
    for kind in ('real', 'fixed', 'adaptive'):
        for row in tables[(kind, 'comparisons')]:
            expected = row['both_qualified'] == 'True' and float(row['speedup_low']) > 1
            assert (row['supported_speedup'] == 'True') == expected

    checked = 0
    for kind in ('fixed', 'adaptive'):
        for phase in ('tuning', 'evaluation'):
            path = HERE / 'evidence' / f'{kind}-{phase}-work.json'
            work = json.loads(path.read_text())
            assert work['mismatches'] == []
            assert all(r['predicted'] == r['measured'] for r in work['checks'])
            assert work['checked'] == len(work['checks'])
            checked += work['checked']
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checked == 447
    example = json.loads((HERE / 'evidence' / 'ideal-key-example.json').read_text())
    top_k = example['top_k']
    recall = sum(min(z, top_k) for z in example['counts']) / (len(example['counts']) * top_k)
    assert recall == .9995
    assert float(next(r for r in tables[('fixed', 'targets')]
                      if r['method']=='scan' and r['documents']=='1000000')['recall']) == recall

    # The binomial tail-sum identity and direct clipped-count sum agree.
    for n in (1, 4, 8):
        for bits in (1, 2):
            p = 2**-bits
            for k in {1, n}:
                tail_sum = sum(sum(math.comb(n, z) * p**z * (1-p)**(n-z)
                                   for z in range(i, n+1)) for i in range(1, k+1))
                assert math.isclose(tail_sum, expected_clipped_count(n, p, k), abs_tol=1e-12)
    return dict(work_predictions=checked, ideal_key_recall=recall, source_sha256=hashes)


if __name__ == '__main__':
    check_scores()
    check_subsets()
    check_binomial()
    finite_recall = check_finite_recall()
    evidence = check_memory_and_evidence()
    implemented = check_implemented_study()
    result = {'score_identity_and_lookup': 'passed', 'complete_weighted_subset_order': 'passed',
              'finite_binary_recall_example': 'passed', 'finite_top_k_recall': finite_recall, 'memory_and_archived_numbers': 'passed', 'implemented_study': implemented, **evidence}
    (HERE/'evidence/math-checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
