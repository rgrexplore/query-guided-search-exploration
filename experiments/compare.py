"""Select the fastest tested settings at common recall and RAM requirements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis import load_cases, write_csv
from experiments.models import predict_search_ms


def best_settings(rows, targets):
    selected = []
    for documents in sorted({row['documents'] for row in rows}):
        for target in targets:
            for method in ('scan', 'branch', 'keys'):
                eligible = [row for row in rows if row['documents']==documents
                            and row['method']==method and row['recall']>=target
                            and row['budget_status']=='fits_by_lifetime_peak' and row['power_unchanged']]
                if eligible:
                    best = min(eligible, key=lambda row: (row['p50_ms'], row['case_id']))
                    selected.append(dict(target=target, **best))
    return selected


def selected_inputs(selected, cases):
    # Failed cases may have been removed, but their recorded IDs do not change.
    by_id = {item['case_id']: item['case'] for item in cases}
    return [dict(target=row['target'], case_id=row['case_id'], case=by_id[row['case_id']])
            for row in selected]


def compare(folder, models_path=None):
    config = json.loads((folder/'configuration.json').read_text())
    cases, failures = load_cases(folder)
    models = json.loads(models_path.read_text()) if models_path else None
    scan_recall = {}
    for item in cases:
        case = item['case']
        if case['method']=='scan':
            signature = (case['pool'], case['router'], case['probes'])
            scan_recall[signature] = {r['query']: r['recall'] for r in item['queries'] if r['repetition']==0}
    rows = []
    for item in cases:
        case, result, queries = item['case'], item['result'], item['queries']
        first = [r for r in queries if r['repetition']==0]
        signature = (case['pool'], case['router'], case['probes'])
        route_recall = scan_recall[signature]
        # Exact local scan is the quality ceiling for this identical selected pool.
        assert all(r['recall'] <= route_recall[r['query']]+1e-12 for r in first)
        observed = [r['query_ms'] for r in queries]
        p50, p95 = np.percentile(observed, [50, 95])
        power = result['power_before']
        row = dict(case_id=item['case_id'], documents=case['documents'], method=case['method'],
                   router=case['router_kind'], clusters=case['clusters'], probes=case['probes'],
                   node_budget=case.get('node_budget', 0), leaf_size=case.get('leaf_size', 0),
                   key_bits=case.get('key_bits', 0), key_offset=case.get('key_offset', 0),
                   candidate_target=case.get('candidate_target', 0), key_limit=case.get('key_limit', 0),
                   recall=float(np.mean([r['recall'] for r in first])),
                   routing_recall=float(np.mean(list(route_recall.values()))),
                   p50_ms=float(p50), p95_ms=float(p95),
                   mean_routing_ms=float(np.mean([r['routing_ms'] for r in queries])),
                   mean_scored=float(np.mean([r['documents_scored'] for r in first])),
                   logical_bytes=result['storage']['logical_bytes'],
                   routing_bytes=result['memory']['routing_payload_bytes'],
                   rss_before_queries=result['memory']['rss_before_queries'],
                   lifetime_peak_bytes=result['memory']['lifetime_peak_bytes'],
                   budget_status=result['memory']['budget_status'],
                   power_unchanged=power['available'] and power==result['power_after'])
        if models:
            model = models[case['method']]
            predictions = [predict_search_ms(model, q)+q['routing_ms']+model['composition_gap_ms'] for q in queries]
            row['predicted_p50_ms'] = float(np.median(predictions))
            row['prediction_error_percent'] = 100*(row['predicted_p50_ms']/row['p50_ms']-1)
        rows.append(row)
    selected = best_settings(rows, config['search']['recall_targets'])
    output = folder/'comparison'
    output.mkdir(exist_ok=True)
    write_csv(output/'settings.csv', rows)
    write_csv(output/'selected.csv', selected)
    choices = selected_inputs(selected, cases)
    (output/'selected.json').write_text(json.dumps(choices, indent=2)+'\n')
    (output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    draw_frontiers(rows, output/'recall-latency.png')
    lines = ['# Coarse real-data tuning', '',
             f'{len(cases)} complete cases; {len(failures)} failed or limited cases.',
             'Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.',
             'Every local search recall was checked against scan under the same routing choice.', '',
             '| Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |',
             '|---:|---|---|---|---:|---:|---:|']
    for row in selected:
        lines.append(f"| {row['target']:.0%} | {row['method']} | {row['router']} | {row['clusters']} / {row['probes']} | {row['recall']:.2%} | {row['p50_ms']:.4f} | {row['case_id']} |")
    lines += ['', 'The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.',
              'This coarse grid needs refinement where useful settings lie at its boundaries. No universal best method or billion-document result is claimed.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return rows, selected


def draw_frontiers(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for method, color in [('scan','#235e91'), ('branch','#d17624'), ('keys','#7a52a2')]:
        own = sorted([r for r in rows if r['method']==method and r['power_unchanged']
                      and r['budget_status']=='fits_by_lifetime_peak'], key=lambda r:r['p50_ms'])
        frontier = []
        best_recall = -1
        for row in own:
            if row['recall'] > best_recall:
                frontier.append(row)
                best_recall = row['recall']
        ax.scatter([r['p50_ms'] for r in own], [r['recall'] for r in own], color=color, alpha=.12, s=16)
        ax.plot([r['p50_ms'] for r in frontier], [r['recall'] for r in frontier], 'o-', color=color, label=method, markersize=4)
    counts = sorted({r['documents'] for r in rows})
    ax.set(xscale='log', xlabel='Measured median query latency (ms; embedding cached)',
           ylabel='Binary-score Recall@100', title='Tuning observations: '+', '.join(f'{n:,} documents' for n in counts))
    ax.legend()
    ax.grid(alpha=.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--models', type=Path)
    args = parser.parse_args()
    compare(args.folder, args.models)


if __name__ == '__main__':
    main()
