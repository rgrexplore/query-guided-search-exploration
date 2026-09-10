"""Report the frozen evaluation without choosing new parameters from test results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis import load_cases, write_csv


def summarize_setting(items, seeds, query_draws, block_draws):
    quality, timings = {}, {}
    for item in items:
        seed = item['case'].get('seed',0)
        block = item['case'].get('block',0)
        for row in item['queries']:
            key = (row['query'],seed)
            if key in quality:
                assert quality[key]==row['recall']
            quality[key] = row['recall']
            timings.setdefault((block,row['query'],seed),[]).append(row['query_ms'])
    queries = sorted({q for _,q,_ in timings})
    blocks = sorted({b for b,_,_ in timings})
    assert len(quality)==len(queries)*len(seeds), 'evaluation seed coverage is incomplete'
    recall = np.array([np.mean([quality[(q,s)] for s in seeds]) for q in queries])
    times = np.array([[[t for s in seeds for t in timings[(b,q,s)]] for q in queries] for b in blocks])
    sampled_recall = recall[query_draws].mean(axis=1)
    # Resample whole process blocks and the same queries for every method.
    medians = np.array([np.median(times[bs][:,qs,:])
                        for bs,qs in zip(block_draws,query_draws,strict=True)])
    summary = dict(recall=float(recall.mean()), recall_low=float(np.quantile(sampled_recall,.025)),
                   recall_high=float(np.quantile(sampled_recall,.975)),
                   p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)),
                   p50_low=float(np.quantile(medians,.025)), p50_high=float(np.quantile(medians,.975)),
                   queries=len(queries), blocks=len(blocks), timing_observations=int(times.size))
    return summary, medians


def report(folder, bootstrap_samples=2000, seed=63):
    cases, failures = load_cases(folder)
    shortlist = json.loads((folder/'frozen-shortlist.json').read_text())
    source = json.loads((folder/'evaluation-source.json').read_text())
    query_count = len(source['query_ids'])
    config = json.loads((folder/'configuration.json').read_text())
    blocks = config['measurement'].get('blocks',1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,query_count,size=(bootstrap_samples,query_count))
    block_draws = rng.integers(0,blocks,size=(bootstrap_samples,blocks))
    by_setting = {}
    for item in cases:
        by_setting.setdefault(item['case']['setting_id'],[]).append(item)
    settings, target_rows, timing_arrays = {}, [], {}
    for choice in shortlist:
        identifier = choice['setting_id']
        items = by_setting.get(identifier,[])
        if len(items)!=len(choice['evaluation_seeds'])*blocks:
            failures.append(dict(setting_id=identifier,reason='missing evaluation seed case'))
            continue
        summary, median_samples = summarize_setting(items,choice['evaluation_seeds'],draws,block_draws)
        case = choice['case']
        summary.update(setting_id=identifier,documents=case['documents'],top_k=case['top_k'],method=case['method'],
                       router=case['router_kind'],clusters=case['clusters'],probes=case['probes'],
                       node_budget=case.get('node_budget',0),leaf_size=case.get('leaf_size',0),
                       key_bits=case.get('key_bits',0),candidate_target=case.get('candidate_target',0),
                       exploration=case.get('exploration',0),
                       lifetime_peak_bytes=max(i['result']['memory']['lifetime_peak_bytes'] for i in items),
                       qualified_environment=all(i['result']['memory']['budget_status']=='fits_by_lifetime_peak'
                           and i['result']['power_before']['available']
                           and i['result']['power_before']==i['result']['power_after'] for i in items))
        settings[identifier] = summary
        timing_arrays[identifier] = median_samples
        for use in choice['uses']:
            target_rows.append(dict(target=use['target'],selection=use['selection'],
                                    met_target=summary['recall']>=use['target'], **summary))
    comparisons = []
    for documents in sorted({r['documents'] for r in target_rows}):
        for selection in ('mean','conservative'):
            for target in sorted({r['target'] for r in target_rows}):
                selected = {r['method']:r for r in target_rows if r['documents']==documents
                            and r['selection']==selection and r['target']==target}
                if 'scan' not in selected:
                    continue
                baseline = selected['scan']
                for method in ('branch','keys'):
                    if method not in selected:
                        continue
                    other = selected[method]
                    a = timing_arrays[baseline['setting_id']]
                    b = timing_arrays[other['setting_id']]
                    ratios = a/b
                    low, high = np.quantile(ratios,[.025,.975])
                    qualified = baseline['met_target'] and other['met_target'] and baseline['qualified_environment'] and other['qualified_environment']
                    comparisons.append(dict(documents=documents,target=target,selection=selection,method=method,
                                            scan_setting=baseline['setting_id'],other_setting=other['setting_id'],
                                            both_qualified=qualified,speedup=baseline['p50_ms']/other['p50_ms'],
                                            speedup_low=float(low),speedup_high=float(high),
                                            supported_speedup=bool(qualified and low>1)))
    output = folder/'report'
    output.mkdir(exist_ok=True)
    write_csv(output/'settings.csv',list(settings.values()))
    write_csv(output/'targets.csv',target_rows)
    write_csv(output/'comparisons.csv',comparisons)
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    (output/'uncertainty.json').write_text(json.dumps(dict(bootstrap_samples=bootstrap_samples,seed=seed,
        unit='query and independent process block',paired=True,
        scope='Query/process-block bootstrap, conditional on the tested seed set and observed machine state. Repeated timings are not independent quality samples. Intervals are not simultaneous guarantees over all comparisons.'),indent=2)+'\n')
    draw_settings(list(settings.values()),output)
    lines=['# Independent evaluation of frozen settings','',
           f'{query_count} disjoint query IDs. Parameters were frozen before this evaluation.',
           'Mean and conservative training selections are reported separately. Target misses remain visible.', '',
           '| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |',
           '|---:|---:|---|---|---:|---:|---|']
    for row in target_rows:
        lines.append(f"| {row['documents']:,} | {row['target']:.0%} | {row['selection']} | {row['method']} | {row['recall']:.2%} | {row['p50_ms']:.4f} | {row['met_target']} |")
    lines += ['', 'A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.',
              'See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Reported {len(settings)} frozen settings and {len(comparisons)} paired comparisons.')
    return settings,comparisons


def draw_settings(rows, output):
    for documents in sorted({r['documents'] for r in rows}):
        fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
        for method,color in [('scan','#235e91'),('branch','#d17624'),('keys','#7a52a2')]:
            own=[r for r in rows if r['documents']==documents and r['method']==method]
            x=np.array([r['recall'] for r in own]); y=np.array([r['p50_ms'] for r in own])
            ax.errorbar(x,y,xerr=np.array([[r['recall']-r['recall_low'] for r in own],
                                           [r['recall_high']-r['recall'] for r in own]]),
                        yerr=np.array([[r['p50_ms']-r['p50_low'] for r in own],
                                       [r['p50_high']-r['p50_ms'] for r in own]]),
                        fmt='o',color=color,label=method,capsize=3)
        ax.set(yscale='log',xlabel=f"Measured binary-score Recall@{next(r['top_k'] for r in rows if r['documents']==documents)}",ylabel='Median query latency (ms)',
               title=f'{documents:,} documents · frozen settings · independent queries')
        ax.legend(); ax.grid(alpha=.15)
        fig.savefig(output/f'recall-latency-n{documents}.png',dpi=150)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    report(args.folder)


if __name__=='__main__':
    main()
