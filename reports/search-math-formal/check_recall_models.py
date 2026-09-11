"""Check recall distributions and draw the report's figures.

These are quality calculations. Their execution time is not a search benchmark.
The native branch calls below only check recall and work on controlled small inputs.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import binom
from threadpoolctl import threadpool_limits
import bitplane_index

HERE = Path(__file__).resolve().parent


def clipped_count(n, top_k, probability):
    # For an integer X, min(K, X) = sum_{j=1}^K 1[X >= j].
    # Survival probabilities avoid forming huge binomial coefficients.
    return float(binom.sf(np.arange(top_k), n, probability).sum())


def recall_from_patterns(patterns, probabilities, weights, accepted, n, top_k):
    """Exact model expectation, including score ties, for a fixed candidate rule."""
    penalty = patterns @ weights
    previous_slots = 0.0
    cumulative_probability = 0.0
    found = 0.0
    for value in np.unique(penalty):
        group = penalty == value
        mass = probabilities[group].sum()
        if mass == 0:
            continue
        cumulative_probability += mass
        slots = clipped_count(n, top_k, min(1.0, cumulative_probability))
        # Within this tied score, row IDs do not favor one sampled pattern.
        survival = probabilities[group & accepted].sum() / mass
        found += (slots - previous_slots) * survival
        previous_slots = slots
    return found / top_k


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1)) / np.sqrt(len(values))
    # Recall is bounded in [0, 1]. This conservative bound stays nonzero even
    # when a finite simulation observes no misses. A zero sample variance
    # alone would otherwise produce a misleading [1, 1] interval.
    half_width = np.sqrt(np.log(2/.05) / (2*len(values)))
    return dict(mean=mean, low=max(0.0, mean-half_width), high=min(1.0, mean+half_width),
                standard_error=standard_error, samples=len(values),
                interval='95% Hoeffding bound across independent corpora; recall in [0,1]')


def controlled_checks(rng, trials):
    patterns = np.array(list(itertools.product([0, 1], repeat=8)), dtype=np.uint8)
    uniform = np.full(len(patterns), 1 / len(patterns))
    paired = np.all(patterns[:, :4] == patterns[:, 4:], axis=1).astype(float)
    paired /= paired.sum()
    cases = [
        ('Equal weights', np.ones(8), uniform),
        ('Strong prefix', np.array([4, 3, 2, 1, 1, 1, 1, 1]), uniform),
        ('Weak prefix', np.array([1, 1, 1, 4, 3, 2, 2, 2]), uniform),
        ('Paired bits', np.ones(8), paired),
    ]
    records = []
    for name, weights, distribution in cases:
        accepted = patterns[:, :3].sum(axis=1) <= 1
        expected = recall_from_patterns(patterns, distribution, weights, accepted, 64, 5)
        independent_prediction = recall_from_patterns(patterns, uniform, weights, accepted, 64, 5)
        sample = rng.choice(len(patterns), size=(trials, 64), p=distribution)
        penalties = patterns @ weights
        # Stable sorting keeps original row-ID order within ties.
        nearest = np.argsort(penalties[sample], axis=1, kind='stable')[:, :5]
        nearest_patterns = np.take_along_axis(sample, nearest, axis=1)
        observed = accepted[nearest_patterns].mean(axis=1)
        interval = mean_interval(observed)
        # This checks exact-model arithmetic. It does not require a wrong model to pass.
        standard_error = interval['standard_error']
        assert abs(interval['mean']-expected) <= 5*standard_error + 1/trials
        native = []
        for query_number, sampled_codes in enumerate(sample[:12]):
            errors = patterns[sampled_codes]
            packed = np.ascontiguousarray(
                np.packbits(errors == 0, axis=1, bitorder='little').astype(np.uint64)
            )
            index = bitplane_index.Index(packed, np.zeros(64, dtype=np.int64), 8)
            query = np.ascontiguousarray(weights[None, :], dtype=np.float32)
            buckets = np.zeros((1, 1), dtype=np.int64)
            truth = np.argsort(penalties[sampled_codes], kind='stable')[:5]
            baseline = index.scan(query, buckets, candidate_limit=5)
            assert baseline['rows'][0, :5].tolist() == truth.tolist()
            for budget, exploration in [(4, 0), (16, 0), (16, .1), (0, 0)]:
                result = index.search(query, buckets, candidate_limit=5,
                                      node_budget=budget, leaf_size=4,
                                      explore_probability=exploration, seed=query_number)
                recall = len(set(truth) & set(result['rows'][0, :5]))/5
                if budget == 0:
                    assert recall == 1
                native.append(dict(query=query_number, budget=budget, exploration=exploration,
                                   recall=recall, scored=int(result['stats'][0]['documents_scored'])))
            # Key grouping and a union of matching bitplane leaves must give the same set.
            from_keys = set(np.flatnonzero(errors[:, :3].sum(axis=1) <= 1).tolist())
            from_planes = set()
            for key in itertools.product([0, 1], repeat=3):
                if sum(key) <= 1:
                    mask = np.ones(64, dtype=bool)
                    for bit, value in enumerate(key):
                        mask &= errors[:, bit] == value
                    from_planes.update(np.flatnonzero(mask).tolist())
            assert from_keys == from_planes
        records.append(dict(name=name, documents=64, dimensions=8, top_k=5, prefix_bits=3,
                            radius=1, weights=weights.tolist(), exact_model_recall=expected,
                            independent_model_recall=independent_prediction,
                            observed=interval, native_queries=native,
                            expected_candidates=64*float(distribution[accepted].sum())))
    return records


def archived_error_intervals(rng, bootstrap_samples):
    """Reanalyse saved queries. These are not a newly collected test set."""
    path = HERE/'evidence/recall-validation.json'
    study = json.loads(path.read_text())
    rows = []
    for case in study['cases']:
        for summary in case['summaries']:
            if summary['method'] != 'C':
                continue
            selected = [r for r in case['per_query']
                        if r['method']=='C' and r['target']==summary['target'] and r['query']>=16]
            rho = summary['fitted_score_correlation']
            tail_points = ndtri(1-(100/case['documents'])*(np.arange(256)+.5)/256)
            errors = []
            for query in selected:
                cutoff = ndtri(1-query['candidates']/case['documents'])
                prediction = ndtr((rho*tail_points-cutoff)/np.sqrt(1-rho*rho)).mean()
                errors.append(100*(prediction-query['recall']))
            errors = np.asarray(errors)
            draws = rng.choice(errors, size=(bootstrap_samples, len(errors)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            rows.append(dict(case=case['name'], target=summary['target'], queries=len(errors),
                             error_points=float(errors.mean()), low=float(low), high=float(high),
                             within_two_points=abs(float(errors.mean()))<=2,
                             interval_within_two_points=bool(low>=-2 and high<=2)))
    return dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows,
                scope='Paired bootstrap over 16 previously inspected check queries. Correlation fit held fixed; not total model uncertainty or a fresh test.')


def make_figures(cases, archived):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.bbox':'tight'})
    distances = np.arange(9)
    cdf = binom.cdf(distances, 16, .5)
    previous = binom.cdf(distances-1, 16, .5)
    mass = (1-previous)**256 - (1-cdf)**256
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.bar(distances, mass*100, color='#32708d')
    ax.set(xlabel='Wrong bits in the nearest document', ylabel='Probability (%)',
           title='Nearest distance: 256 independent documents, 16 balanced bits')
    ax.set_xticks(distances)
    fig.savefig(HERE/'figures/nearest-distance.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(cases))
    ax.scatter(x-.13, [100*c['exact_model_recall'] for c in cases], label='Exact stated distribution', color='#17354b', marker='s')
    ax.scatter(x+.13, [100*c['independent_model_recall'] for c in cases], label='Assume independent bits', color='#bc652c', marker='x', s=65)
    means=np.array([100*c['observed']['mean'] for c in cases])
    half=np.array([[100*(c['observed']['mean']-c['observed']['low']) for c in cases],
                   [100*(c['observed']['high']-c['observed']['mean']) for c in cases]])
    ax.errorbar(x, means, yerr=half, fmt='o', color='#27876e', capsize=4, label='Sampled corpora, 95% interval')
    ax.set_xticks(x, [c['name'] for c in cases])
    ax.set(ylabel='Recall@5 (%)', title='Same prefix rule; different document and weight assumptions')
    ax.legend(fontsize=8, loc='lower left')
    fig.savefig(HERE/'figures/recall-model-checks.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args=parser.parse_args()
    rng=np.random.default_rng(args.seed)
    with threadpool_limits(limits=1):
        cases=controlled_checks(rng, args.trials)
        archived=archived_error_intervals(rng, args.bootstrap_samples)
    make_figures(cases, archived)
    result=dict(seed=args.seed, trials=args.trials, bootstrap_samples=args.bootstrap_samples,
                controlled=cases, archived=archived,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'evidence/recall-model-checks.json').write_text(json.dumps(result, indent=2)+'\n')
    for case in cases:
        print(case['name'], 'exact:', round(case['exact_model_recall'],6),
              'sampled:', round(case['observed']['mean'],6),
              'independent:', round(case['independent_model_recall'],6))
    print('Archived prediction errors:', json.dumps(archived['rows'], indent=2))


if __name__ == '__main__':
    main()
