"""Build the paper's tables and figures from completed, frozen experiments.

Run from the repository root after evaluation and the frozen model check:
    python reports/search-math-optima/make_results.py
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.analysis import load_cases
from experiments.evaluation_report import summarize_setting


REPORT = Path(__file__).resolve().parent
LETTERS = {'scan': 'A', 'branch': 'B', 'keys': 'C'}
COLORS = {'scan': '#286493', 'branch': '#C36620', 'keys': '#7850A4'}
CONDITIONS = {'fixed': 'Fixed', 'adaptive': 'Changing', 'real': 'Real data'}
WORK = ('documents_scored', 'bitplane_words', 'leaf_words', 'nodes',
        'key_attempts', 'keys_generated')
INPUTS = {}


def record(path):
    """Keep source paths and hashes beside the generated evidence."""
    resolved = path.resolve()
    try:
        name = str(resolved.relative_to(ROOT))
    except ValueError:
        name = str(resolved)
    INPUTS[name] = hashlib.sha256(resolved.read_bytes()).hexdigest()


def read_json(path):
    record(path)
    return json.loads(path.read_text())


def read_csv(path):
    record(path)
    with path.open() as source:
        return list(csv.DictReader(source))


def mean_work(items):
    """Repeated timings are not extra work-count samples."""
    unique = {}
    for item in items:
        seed = item['case'].get('seed', 0)
        for row in item['queries']:
            key = (row['query'], seed)
            counts = tuple(row[name] for name in WORK)
            if key in unique:
                assert unique[key] == counts, 'work changed for identical search settings'
            unique[key] = counts
    values = np.array(list(unique.values()), dtype=float)
    return {name: float(value) for name, value in zip(WORK, values.mean(axis=0), strict=True)}


def selected_results(folder, condition, bootstrap_samples=2000, seed=63):
    """Evaluate the earlier conservative choices, never select from final times."""
    cases, failures = load_cases(folder)
    if failures:
        raise ValueError(f'{folder} contains incomplete or failed cases: {failures}')
    for name in ('configuration.json', 'schedule.json', 'frozen-shortlist.json', 'evaluation-source.json'):
        record(folder / name)
    for item in cases:
        directory = folder / 'cases' / f'{item["case_id"]:04d}'
        for name in ('process.json', 'run/result.json', 'run/queries.jsonl'):
            record(directory / name)
    shortlist = read_json(folder / 'frozen-shortlist.json')
    choices = [choice for choice in shortlist
               if dict(selection='conservative', target=.99) in choice['uses']]
    assert {choice['case']['method'] for choice in choices} == set(LETTERS)
    assert len(choices) == 3, 'expect one previously selected setting per method'
    grouped = {choice['setting_id']: [item for item in cases
                                     if item['case']['setting_id'] == choice['setting_id']]
               for choice in choices}
    populations = {tuple(sorted({row['query'] for item in items for row in item['queries']}))
                   for items in grouped.values()}
    assert len(populations) == 1, 'paired comparison requires identical final query IDs'
    query_count = len(next(iter(populations)))
    blocks = sorted({item['case'].get('block', 0) for items in grouped.values() for item in items})
    rng = np.random.default_rng(seed)
    query_draws = rng.integers(0, query_count, size=(bootstrap_samples, query_count))
    block_draws = rng.integers(0, len(blocks), size=(bootstrap_samples, len(blocks)))
    settings, distributions = {}, {}
    for choice in choices:
        case = choice['case']
        items = grouped[choice['setting_id']]
        assert len(items) == len(blocks) * len(choice['evaluation_seeds'])
        summary, resampled = summarize_setting(items, choice['evaluation_seeds'], query_draws, block_draws)
        memory = [item['result']['memory'] for item in items]
        qualified = all(row['budget_status'] == 'fits_by_lifetime_peak' for row in memory)
        qualified = qualified and all(item['result']['power_before']['available']
                                      and item['result']['power_before'] == item['result']['power_after']
                                      for item in items)
        summary.update(
            condition=condition, method=case['method'], setting_id=choice['setting_id'],
            selection='conservative', target=.99, met_target=summary['recall'] >= .99,
            case=case, qualified_environment=qualified, mean_work=mean_work(items),
            logical_index_bytes=items[0]['result']['storage']['logical_bytes'],
            array_capacity_bytes=items[0]['result']['storage']['array_capacity_bytes'],
            process_peak_bytes=max(row['lifetime_peak_bytes'] for row in memory),
            query_start_rss_bytes=max(row['rss_before_queries'] for row in memory),
            sampled_query_peak_bytes=max(row['sampled_query_peak'] for row in memory),
        )
        settings[case['method']] = summary
        distributions[case['method']] = resampled
    comparisons = []
    for numerator, denominator in itertools.combinations(LETTERS, 2):
        first, second = settings[numerator], settings[denominator]
        ratios = distributions[numerator] / distributions[denominator]
        low, high = map(float, np.quantile(ratios, [.025, .975]))
        qualified = all(row['met_target'] and row['qualified_environment'] for row in (first, second))
        faster = None
        if qualified and low > 1:
            faster = LETTERS[denominator]
        elif qualified and high < 1:
            faster = LETTERS[numerator]
        comparisons.append(dict(
            condition=condition, ratio=f'{LETTERS[numerator]}/{LETTERS[denominator]}',
            numerator_setting=first['setting_id'], denominator_setting=second['setting_id'],
            ratio_of_medians=first['p50_ms'] / second['p50_ms'], low=low, high=high,
            both_qualified=qualified, supported_faster=faster,
        ))
    return list(settings.values()), comparisons


def table(path, alignment, header, rows):
    lines = [r'\begin{tabular}{@{}' + alignment + r'@{}}', r'\toprule',
             ' & '.join(header) + r' \\', r'\midrule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n')


def selected_table(path, settings):
    rows = []
    for row in sorted(settings, key=lambda row: (list(CONDITIONS).index(row['condition']), LETTERS[row['method']])):
        case = row['case']
        router = {'none': 'Global', 'direct': 'Prefix', 'sign': 'Prefix', 'ivf': 'IVF'}[case['router_kind']]
        leaf = ('--' if row['method'] != 'branch' else
                r'$N$' if case['leaf_size'] == case['documents'] else str(case['leaf_size']))
        rows.append([
            CONDITIONS[row['condition']], LETTERS[row['method']],
            f'{router} {case["clusters"]:,}/{case["probes"]:,}', leaf,
            str(case['key_bits']) if row['method'] == 'keys' else '--',
            f'{100 * row["recall"]:.2f}', f'{row["p50_ms"]:.4f}',
            f'{row["process_peak_bytes"] / 1e6:.1f}',
        ])
    table(path, 'lllrrrrr', ['Condition', 'Method', r'Route $C/P$', r'$L$', r'$h$',
                            r'Recall (\%)', 'ms', 'Peak MB'], rows)


def work_table(path, settings):
    rows = []
    for row in sorted(settings, key=lambda row: (list(CONDITIONS).index(row['condition']), LETTERS[row['method']])):
        counts = row['mean_work']
        rows.append([CONDITIONS[row['condition']], LETTERS[row['method']],
                     *[f'{counts[name]:,.1f}' for name in
                       ('documents_scored', 'bitplane_words', 'leaf_words', 'key_attempts')]])
    table(path, 'llrrrr', ['Condition', 'Method', 'Rows scored', 'Split words', 'Leaf words', 'Key lookups'], rows)


def ratio_table(path, comparisons):
    # Keep small ratios readable without rounding an interval to equal ends.
    def display(value):
        return f'{value:.5f}' if value < .01 else f'{value:.3f}'

    rows = [[CONDITIONS[row['condition']], row['ratio'], display(row['ratio_of_medians']),
             f'[{display(row["low"])}, {display(row["high"])}]', row['supported_faster'] or '--']
            for row in comparisons]
    table(path, 'llrrl', ['Condition', 'Time ratio', 'Estimate', r'95\% interval', 'Faster'], rows)


def model_table(path, rows):
    data = []
    for row in rows:
        if row['status'] != 'checked':
            raise ValueError('a model-selected setting has no final evaluation')
        data.append([CONDITIONS[row['support']], LETTERS[row['method']],
                     f'{row["predicted_p50_ms"]:.4f}', f'{row["measured_p50_ms"]:.4f}',
                     f'{row["p50_error_percent"]:+.1f}', f'{100 * row["recall"]:.2f}',
                     'Yes' if row['same_as_measured_best'] else 'No'])
    table(path, 'llrrrrl', ['Condition', 'Method', 'Predicted ms', 'Measured ms',
                          r'Error (\%)', r'Recall (\%)', 'Same setting'], data)


def tuning_plot(study, destination):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True, layout='constrained')
    plotted = []
    for axis, support in zip(axes, ('fixed', 'adaptive'), strict=True):
        rows = read_csv(study / support / 'frozen' / 'settings.csv')
        for method, letter in LETTERS.items():
            best = {}
            for row in rows:
                if row['method'] != method or not row['recall_low'] or float(row['recall_low']) < .99:
                    continue
                clusters = int(row['clusters'])
                if clusters not in best or float(row['p50_ms']) < float(best[clusters]['p50_ms']):
                    best[clusters] = row
            points = sorted(best.items())
            axis.plot([count for count, _ in points], [float(row['p50_ms']) for _, row in points],
                      marker='o', markersize=4, linewidth=1.4, color=COLORS[method], label=letter)
            plotted.extend(dict(row, condition=support) for _, row in points)
        axis.set(xscale='log', yscale='log', xlabel='Clusters', title=CONDITIONS[support] + ' positions')
        axis.set_xticks([1, 256, 16384], ['1', '256', '16,384'])
        axis.grid(alpha=.18)
        axis.legend(title='Method', ncol=3, frameon=False, handlelength=1, columnspacing=.6)
    axes[0].set_ylabel('Median query time (ms)')
    fig.savefig(destination)
    plt.close(fig)
    return plotted


def depth_plot(formulas, strong_bits, destination):
    fig, axis = plt.subplots(figsize=(6.8, 2.7), layout='constrained')
    for support, color, style in [('fixed', '#286493', '-'), ('adaptive', '#7850A4', '--')]:
        model = formulas[support]['global_branch_derivative']
        if not model['available']:
            raise ValueError(f'no identifiable global depth model for {support}')
        depths, times = model['candidate_depths'], model['predicted_search_ms']
        axis.plot(depths, times, color=color, linestyle=style, marker='o', markersize=3,
                  label=CONDITIONS[support] + ' positions')
        position = depths.index(strong_bits)
        axis.scatter([strong_bits], [times[position]], s=55, facecolors='white', edgecolors=color, zorder=4)
    axis.axvline(strong_bits, color='#61727F', linewidth=.8, linestyle=':')
    axis.set(yscale='log', xlabel='Split depth', ylabel='Modeled search time (ms)')
    axis.set_xticks(range(0, strong_bits + 1, 2))
    axis.grid(alpha=.18)
    axis.legend(frameon=False)
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'results/optima-2026-09-12')
    parser.add_argument('--real-folder', type=Path, help='Optional completed real-data replication folder.')
    args = parser.parse_args()
    study = args.study.resolve()
    real = args.real_folder.resolve() if args.real_folder else study / 'real-replication'
    evidence, figures = REPORT / 'evidence', REPORT / 'figures'
    evidence.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    settings, comparisons = [], []
    for condition in ('fixed', 'adaptive'):
        rows, pairs = selected_results(study / condition / 'evaluation', condition)
        settings.extend(rows)
        comparisons.extend(pairs)
    model_check = read_json(study / 'predictions' / 'check.json')
    formulas = read_json(study / 'predictions' / 'formulas.json')
    model_metadata = read_json(study / 'predictions' / 'provenance.json')
    record(study / 'predictions' / 'models.json')
    record(study / 'predictions' / 'shortlist.json')
    config = read_json(study / 'configuration.json')
    real_settings, real_comparisons = [], []
    if real.exists():
        real_settings, real_comparisons = selected_results(real, 'real')
        selected_table(evidence / 'real-settings.tex', real_settings)
    elif args.real_folder:
        raise FileNotFoundError(real)
    selected_table(evidence / 'measured-settings.tex', settings)
    work_table(evidence / 'work-counts.tex', settings + real_settings)
    ratio_table(evidence / 'pairwise-ratios.tex', comparisons)
    model_table(evidence / 'model-check.tex', model_check['rows'])
    tuning = tuning_plot(study, figures / 'tuning-clusters.pdf')
    depth_plot(formulas, config['data']['strong_bits'], figures / 'global-branch-depth.pdf')
    record(Path(__file__))
    record(ROOT / 'experiments/evaluation_report.py')
    result = dict(
        measured_settings=settings, pairwise_comparisons=comparisons,
        model_selected_checks=model_check, model_metadata=model_metadata,
        real_settings=real_settings, real_comparisons=real_comparisons,
        tuning_cluster_points=tuning, depth_models=formulas,
        bootstrap=dict(samples=2000, seed=63, paired_queries_and_blocks=True,
                       scope='Pointwise intervals over the observed queries and process blocks, not simultaneous guarantees.'),
        memory_scope='Process lifetime peak includes index construction and runtime. It is not index payload alone.',
        selection_scope='Final settings were selected on tuning queries. Tuning plots minimize within the declared family only.',
        sources=INPUTS,
    )
    (evidence / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    captions = {
        'tuning-clusters.pdf': 'Lowest measured tuning latency at each cluster count among settings whose tuning recall lower bound reaches 99%. Each method selects its own router, probes, and local settings. These points are tuning results, not independent evidence of a speedup.',
        'global-branch-depth.pdf': 'Frozen model prediction for one global bitplane path. Every integer depth is evaluated. The highlighted final depth matches all strong coordinates; this curve is a model prediction, not a measured latency curve.',
        'measured-settings.tex': 'Previously selected conservative 99% settings on final queries. C/P denotes clusters/probes, L the leaf limit, and h the local key width. N means a scan-size leaf. Peak MB is the maximum whole-process lifetime peak, including construction.',
        'pairwise-ratios.tex': 'Paired bootstrap intervals for ratios of median query times. A ratio above one favors the denominator. A faster method is identified only when both methods meet the target and environment checks and the interval excludes one.',
        'model-check.tex': 'Frozen model-selected settings compared with final measurements. Error is predicted/measured minus one. Same setting compares the parameter choice with the earlier measured-tuning choice, not whether the final times are identical.',
    }
    (evidence / 'captions.json').write_text(json.dumps(captions, indent=2) + '\n')
    print(f'Wrote {evidence / "results.json"}, compact TeX tables, and two vector PDF figures.')


if __name__ == '__main__':
    main()
