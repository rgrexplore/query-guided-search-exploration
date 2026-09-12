"""Small quality study. C's candidate construction is not a timed hash implementation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits
import bitplane_index

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/scaling/1be0393b70430821be5a67e569bcc2df11106c0e17afdeb574350f4930bea7d3/embeddings/e867d7419fedbd667eaea8e1febe56b2fe16881be9570f6ae11c6bef51bff273'
D, H, QUERIES, TOP = 256, 24, 32, 100
TARGETS = [100, 1000, 5000, 10000]
BUDGETS = [512, 2048, 8192]


def best_rows(scores, rows, limit):
    """Use the same score and row-ID tie order everywhere."""
    return rows[np.lexsort((rows, -scores[rows]))[:limit]]


def subset_sums(weights):
    sums = np.zeros(1, dtype=np.float64)
    for weight in weights:
        sums = np.concatenate((sums, sums + weight))
    return sums


def keys_within(weights, penalty):
    """Exact weighted key-volume count via two halves, including final-score ties."""
    left = subset_sums(weights[:len(weights)//2])
    right = np.sort(subset_sums(weights[len(weights)//2:]))
    tolerance = 1e-12 * max(1, weights.sum())
    return int(np.searchsorted(right, penalty - left + tolerance, side='right').sum())


def predict_recall(rho, n, selected_fraction):
    if selected_fraction >= 1:
        return 1.0
    if selected_fraction <= 0:
        return 0.0
    # Integrate over the full top-100 tail. The normal-score assumption is being tested.
    z = ndtri(1 - (TOP / n) * (np.arange(256) + .5) / 256)
    threshold = ndtri(1 - selected_fraction)
    return float(ndtr((rho * z - threshold) / np.sqrt(max(1e-12, 1-rho*rho))).mean())


def validate_key_counter():
    weights = np.array([.7, .5, .4, .6, .3, .12, .08, .04])
    sums = subset_sums(weights)
    for threshold in [0, .04, .12, .4, 1, weights.sum()]:
        expected = int((sums <= threshold + 1e-12).sum())
        assert keys_within(weights, threshold) == expected
    assert keys_within(weights, -1) == 0


def dataset(name, signs, queries):
    n = len(signs)
    rows = np.arange(n, dtype=np.int64)
    full = np.where(signs, 1., -1.).astype(np.float64)
    prefix = np.ascontiguousarray(full[:, :H])
    packed = np.ascontiguousarray(np.packbits(signs, axis=1, bitorder='little').view(np.uint64))
    index = bitplane_index.Index(packed, np.zeros(n, dtype=np.int64), D)
    buckets = np.zeros((1, 1), dtype=np.int64)
    records, reference_checks = [], 0
    for qi, q in enumerate(queries):
        scores = full @ q.astype(np.float64)
        partial = prefix @ q[:H].astype(np.float64)
        truth = best_rows(scores, rows, TOP)
        truth_set = set(truth.tolist())
        rho = float(np.corrcoef(scores, partial)[0, 1])
        energy = float(np.dot(q[:H], q[:H]) / np.dot(q, q))
        baseline = index.scan(np.ascontiguousarray(q[None, :], dtype=np.float32), buckets, candidate_limit=1000)
        assert baseline['rows'][0, :TOP].tolist() == truth.tolist()
        reference_checks += 1
        for budget in BUDGETS:
            result = index.search(np.ascontiguousarray(q[None, :], dtype=np.float32), buckets,
                                  candidate_limit=1000, node_budget=budget, leaf_size=32)
            returned = result['rows'][0, :TOP]
            records.append(dict(query=qi, method='B', budget=budget,
                                recall=len(truth_set.intersection(returned.tolist()))/TOP,
                                scored=int(result['stats'][0]['documents_scored'])))
        for target in TARGETS:
            # Collect all prefix keys at or above the stopping score, not just target row IDs.
            cutoff = np.partition(partial, n-target)[n-target]
            tolerance = 1e-12 * max(1, np.abs(q[:H]).sum())
            candidates = rows[partial >= cutoff - tolerance]
            returned = best_rows(scores, candidates, TOP)
            penalty = (np.abs(q[:H]).sum(dtype=np.float64) - cutoff) / 2
            count = keys_within(np.abs(q[:H]).astype(np.float64), penalty)
            # Under uniform random signs, this is the exact expected candidate fraction.
            uniform_fraction = count / (2**H)
            records.append(dict(query=qi, method='C', target=target, candidates=len(candidates),
                                recall=len(truth_set.intersection(returned.tolist()))/TOP,
                                weighted_key_attempts=count, uniform_expected_candidates=n*uniform_fraction,
                                rho=rho, prefix_energy=energy))
    summaries = []
    # The first 16 queries estimate correlation; the next 16 check recall predictions.
    for target in TARGETS:
        c = [r for r in records if r['method']=='C' and r['target']==target]
        calibration, test = c[:16], c[16:]
        fitted_rho = float(np.mean([r['rho'] for r in calibration]))
        predictions = [predict_recall(fitted_rho, n, r['candidates']/n) for r in test]
        observed = [r['recall'] for r in test]
        summaries.append(dict(method='C', target=target, measured_recall=float(np.mean(observed)),
                              predicted_recall=float(np.mean(predictions)),
                              prediction_gap_points=100*float(np.mean(predictions)-np.mean(observed)),
                              mean_candidates=float(np.mean([r['candidates'] for r in test])),
                              median_key_attempts=float(np.median([r['weighted_key_attempts'] for r in test])),
                              fitted_score_correlation=fitted_rho,
                              mean_prefix_energy=float(np.mean([r['prefix_energy'] for r in test])),
                              mean_uniform_expected_candidates=float(np.mean([r['uniform_expected_candidates'] for r in test]))))
    for budget in BUDGETS:
        b = [r for r in records if r['method']=='B' and r['budget']==budget and r['query']>=16]
        summaries.append(dict(method='B', budget=budget, measured_recall=float(np.mean([r['recall'] for r in b])),
                              mean_documents_scored=float(np.mean([r['scored'] for r in b]))))
    print(name, json.dumps(summaries, indent=2), flush=True)
    return dict(name=name, documents=n, queries=QUERIES, dimensions=D, key_bits=H,
                exact_reference_checks=reference_checks, summaries=summaries, per_query=records)


def main():
    validate_key_counter()
    rng = np.random.default_rng(20260909)
    signs = rng.integers(0, 2, size=(20000, D), dtype=np.uint8).astype(bool)
    query = rng.standard_normal((QUERIES, D)).astype(np.float32)
    strong = query.copy()
    # This favors the prefix by construction. It is not a claim about Matryoshka training.
    strong[:, :H] *= np.sqrt((.8/H)/(.2/(D-H)))
    doc_vectors = np.load(CACHE/'full-documents.npy', mmap_mode='r')
    query_vectors = np.load(CACHE/'full-queries.npy', mmap_mode='r')
    real_signs = np.ascontiguousarray(doc_vectors[:100000, :D] >= 0)
    real_queries = np.ascontiguousarray(query_vectors[:QUERIES, :D], dtype=np.float32)
    with threadpool_limits(limits=1):
        cases = [dataset('Random independent signs', signs, query),
                 dataset('Random signs; 80% expected score variance in first 24 dims', signs, strong),
                 dataset('Cached Nomic / MS MARCO binary scores', real_signs, real_queries)]
    report = dict(seed=20260909, ground_truth='Exact top 100 by full 256-d float-query × binary-document score; ties by row ID',
                  candidate_limit_native=1000, queries_for_correlation=16, separate_test_queries=16,
                  scope='No routing. Quality study, not a measured hash-search latency benchmark.',
                  key_count='Meet-in-the-middle count of ALL weighted flip sets through the stopping tie, including empty keys',
                  cache_manifest_sha256=hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest(),
                  cases=cases)
    (ROOT/'calculator/recall-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved calculator/recall-validation.json', flush=True)


if __name__ == '__main__':
    main()
