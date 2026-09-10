"""Check random exploration near the useful deterministic branch settings."""
import json
from pathlib import Path

import numpy as np

from experiments.analysis import load_cases
from experiments.choices import effective_probes, setting_key
from experiments.isolated import ROOT, execute_cases
from experiments.models import mean_recall


def select_bases(rows, targets):
    selected = set()
    for documents in sorted({r['documents'] for r in rows}):
        own = [r for r in rows if r['documents']==documents]
        for target in targets:
            eligible = [r for r in own if r['recall']>=target]
            if not eligible:
                continue
            best = min(eligible, key=lambda r:r['p50_ms'])
            selected.add(best['key'])
            # Include two fast, genuinely pruning settings close to the target.
            # Otherwise exploration would only be tested on complete scans.
            pruning = [r for r in own if r['recall']>=target-.05 and r['p50_ms']<=1.5*best['p50_ms']
                       and r['splits']>0 and r['scored']<.9*r['routed']]
            selected.update(r['key'] for r in sorted(pruning,key=lambda r:r['p50_ms'])[:2])
    return sorted(selected)


def run_exploration(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    groups, scan_cases, scan_work = {}, {}, {}
    for source in config['experiment']['sources']:
        cases, failures = load_cases(ROOT/source)
        if failures:
            raise ValueError('inspect failed source cases before deriving the exploration follow-up')
        for item in cases:
            case, result = item['case'], item['result']
            signature = (case['pool'],case['router'],effective_probes(case,result))
            if case['method']=='scan':
                scan_cases[signature] = dict(case,probes=signature[2])
                scan_work[signature] = np.mean([r['documents_scored'] for r in item['queries']])
        for item in cases:
            case, result = item['case'], item['result']
            if case['method']!='branch' or case.get('exploration',0)!=0:
                continue
            if not result['power_before']['available'] or result['power_before']!=result['power_after'] or result['memory']['budget_status']!='fits_by_lifetime_peak':
                continue
            probes = effective_probes(case,result)
            key = setting_key(case,probes)
            if key not in groups:
                groups[key] = dict(case=dict(case,probes=probes),times=[],quality={},work={})
            group = groups[key]
            for row in item['queries']:
                group['times'].append(row['query_ms'])
                group['quality'][row['query']] = row['recall']
                group['work'][row['query']] = (row['documents_scored'],row['bitplane_words'])
    rows = []
    for key, group in groups.items():
        case = group['case']
        signature = (case['pool'],case['router'],case['probes'])
        rows.append(dict(key=key,documents=case['documents'],recall=mean_recall(list(group['quality'].values()),case['top_k']),
                         p50_ms=np.median(group['times']),scored=np.mean([v[0] for v in group['work'].values()]),
                         splits=np.mean([v[1] for v in group['work'].values()]),routed=scan_work[signature]))
    chosen = select_bases(rows,config['search']['recall_targets'])
    cases = []
    used_scans = set()
    for key in chosen:
        base = groups[key]['case']
        signature = (base['pool'],base['router'],base['probes'])
        if signature not in used_scans:
            cases.append(dict(scan_cases[signature],repetitions=config['measurement']['repetitions']))
            used_scans.add(signature)
        for probability in config['sweep']['exploration']:
            seeds = [0] if probability==0 else config['sweep']['seeds']
            for seed in seeds:
                cases.append(dict(base,exploration=probability,seed=seed,
                                  repetitions=config['measurement']['repetitions']))
    order = np.random.default_rng(config['measurement']['schedule_seed']).permutation(len(cases))
    cases = [cases[int(i)] for i in order]
    (output/'selected-bases.json').write_text(json.dumps([groups[key]['case'] for key in chosen],indent=2)+'\n')
    print(f'Checking {len(chosen)} branch settings in {len(cases)} cases, including shared scan controls.',flush=True)
    execute_cases(config,output,cases,'Exploration follow-up on tuning queries; seeds are averaged, not selected for the best outcome.')
