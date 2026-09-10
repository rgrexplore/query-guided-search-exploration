"""Compare the cost model with complete, isolated query observations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.models import fit_cost_model, predict_search_ms


def load_cases(folder):
    schedule = json.loads((folder/'schedule.json').read_text())
    complete, failures = [], []
    for number, case in enumerate(schedule):
        case_dir = folder/'cases'/f'{number:04d}'
        process_path = case_dir/'process.json'
        if not process_path.exists():
            raise ValueError(f'case {number} is unfinished; analyze after the run completes')
        process = json.loads(process_path.read_text())
        if process['status'] != 'complete':
            failures.append(dict(case_id=number, case=case, process=process))
            continue
        result = json.loads((case_dir/'run/result.json').read_text())
        assert result['case'] == case
        assert result['storage']['documents'] == case['documents']
        assert result['storage']['dimensions'] == case['dimensions']
        queries = [json.loads(line) for line in (case_dir/'run/queries.jsonl').read_text().splitlines()]
        for row in queries:
            row.update(dimensions=case['dimensions'], documents=case['documents'], method=case['method'])
        complete.append(dict(case_id=number, case=case, result=result, queries=queries))
    assert len({c['result']['native_sha256'] for c in complete}) == 1, 'mixed native builds'
    assert len({c['result']['worker_sha256'] for c in complete}) == 1, 'mixed worker implementations'
    return complete, failures


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(folder, output_name='analysis-relative', models_path=None, loss='relative'):
    config = json.loads((folder/'configuration.json').read_text())
    fit_sizes = set(config['timing_model']['fit_sizes'])
    check_sizes = set(config['timing_model']['check_sizes'])
    assert not fit_sizes & check_sizes
    cases, failures = load_cases(folder)
    models = json.loads(models_path.read_text()) if models_path else {}
    if models_path is None:
        for method in ('scan', 'branch', 'keys'):
            training = [row for case in cases if case['case']['documents'] in fit_sizes
                        and case['case']['method']==method for row in case['queries']]
            model = fit_cost_model(training, method, loss=loss)
            # Compose times per query. This is not a sum of stage medians.
            model['composition_gap_ms'] = float(np.median([
                row['query_ms']-row['routing_ms']-row['search_ms'] for row in training]))
            models[method] = model
    summaries = []
    for item in cases:
        case, result, queries = item['case'], item['result'], item['queries']
        model = models[case['method']]
        predicted = [predict_search_ms(model, row)+row['routing_ms']+model['composition_gap_ms']
                     for row in queries]
        observed = [row['query_ms'] for row in queries]
        p50, p95 = np.percentile(observed, [50, 95])
        predicted_p50, predicted_p95 = np.percentile(predicted, [50, 95])
        first = [row for row in queries if row['repetition']==0]
        stable_fields = ('recall', 'documents_scored', 'bitplane_words', 'leaf_words',
                         'nodes', 'key_attempts', 'keys_generated')
        for query in first:
            repeats = [r for r in queries if r['query']==query['query']]
            assert all(all(r[name]==query[name] for name in stable_fields) for r in repeats)
        error = 100*(predicted_p50-p50)/p50
        power_before, power_after = result['power_before'], result['power_after']
        summaries.append(dict(
            case_id=item['case_id'], method=case['method'], documents=case['documents'],
            phase='fit' if case['documents'] in fit_sizes else 'check',
            leaf_size=case.get('leaf_size', 0), node_budget=case.get('node_budget', 0),
            key_bits=case.get('key_bits', 0), candidate_target=case.get('candidate_target', 0),
            recall=float(np.mean([r['recall'] for r in first])),
            p50_ms=float(p50), predicted_p50_ms=float(predicted_p50), p50_error_percent=float(error),
            p95_ms=float(p95), predicted_p95_ms=float(predicted_p95),
            within_twenty_percent=abs(error)<=20,
            logical_index_bytes=result['storage']['logical_bytes'],
            array_capacity_bytes=result['storage']['array_capacity_bytes'],
            rss_before_queries=result['memory']['rss_before_queries'],
            sampled_query_peak=result['memory']['sampled_query_peak'],
            lifetime_peak_bytes=result['memory']['lifetime_peak_bytes'],
            budget_status=result['memory']['budget_status'],
            mean_scored=float(np.mean([r['documents_scored'] for r in first])),
            mean_split_words=float(np.mean([r['bitplane_words'] for r in first])),
            mean_leaf_words=float(np.mean([r['leaf_words'] for r in first])),
            mean_key_attempts=float(np.mean([r['key_attempts'] for r in first])),
            max_mask_bytes=max(r['peak_mask_bytes'] for r in queries),
            max_key_queue_bytes=max(r['peak_key_queue_bytes'] for r in queries),
            power_unchanged=power_before['available'] and power_before==power_after,
        ))
    output = folder/output_name
    output.mkdir(exist_ok=True)
    write_csv(output/'settings.csv', summaries)
    (output/'models.json').write_text(json.dumps(models, indent=2)+'\n')
    (output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    draw_checks(summaries, output/'predicted-versus-measured.png')
    lines = ['# Isolated timing calibration', '',
             f'{len(cases)} complete cases; {len(failures)} failed or limited cases.',
             'The model uses measured work counts and measured routing time. It is not a forecast of recall or work from N alone.',
             ('Frozen model loaded from '+str(models_path)) if models_path else 'Check sizes were excluded from fitting. Query vectors are shared across sizes.', '',
             '| Method | Check settings | Within 20% | Worst absolute p50 error |',
             '|---|---:|---:|---:|']
    for method in models:
        checked = [r for r in summaries if r['method']==method and r['phase']=='check']
        lines.append(f"| {method} | {len(checked)} | {sum(r['within_twenty_percent'] for r in checked)} | {max(abs(r['p50_error_percent']) for r in checked):.1f}% |")
    lines += ['', 'Coefficients combine scoring, memory access, result selection and queue work; they are not hardware instruction timings.',
              'Whole-process lifetime peak includes construction. Query RSS observations can miss brief peaks. Logical payload excludes runtime and allocator overhead.',
              'No tuned winner or billion-document prediction is established by this calibration run.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return summaries


def draw_checks(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, method in zip(axes, ('scan', 'branch', 'keys')):
        selected = [r for r in rows if r['method']==method and r['phase']=='check']
        x = np.array([r['p50_ms'] for r in selected])
        y = np.array([r['predicted_p50_ms'] for r in selected])
        low, high = min(x.min(), y.min())*.8, max(x.max(), y.max())*1.2
        ax.plot([low, high], [low, high], color='#52677a', linewidth=1)
        ax.fill_between([low, high], [.8*low, .8*high], [1.2*low, 1.2*high], color='#dce7ef', alpha=.6)
        ax.scatter(x, y, c=[r['documents'] for r in selected], cmap='viridis', edgecolor='white', linewidth=.5)
        ax.set(xscale='log', yscale='log', xlabel='Measured p50 (ms)', ylabel='Predicted p50 (ms)', title=method)
    fig.suptitle('Excluded check sizes; shaded region is ±20%')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output-name', default='analysis-relative')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--loss', choices=['relative', 'absolute'], default='relative')
    args = parser.parse_args()
    analyze(args.folder, args.output_name, args.models, args.loss)


if __name__ == '__main__':
    main()
