"""Combine equivalent tuning settings and freeze an evaluation shortlist."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from experiments.analysis import load_cases, write_csv


def effective_probes(case, result):
    if case['router_kind']=='none':
        return 1
    dimensions = int(math.log2(case['clusters'])) if case['router_kind']=='sign' else case['dimensions']
    # A route stores one float32 centroid and one int64 label. This recovers the
    # available label count from saved metadata without needing the vector cache.
    route_count = result['memory']['routing_payload_bytes'] // (4*dimensions + 8)
    return min(case['probes'], route_count)


def setting_key(case, probes):
    shared = {name: case[name] for name in
              ('method','documents','dimensions','top_k','pool','router','router_kind','clusters')}
    shared['probes'] = probes
    if case['method']=='branch':
        shared.update(node_budget=case['node_budget'], leaf_size=case['leaf_size'],
                      exploration=float(case.get('exploration',0)))
    elif case['method']=='keys':
        shared.update({name:case[name] for name in ('key_bits','key_offset','candidate_target','key_limit')})
    return json.dumps(shared, sort_keys=True)


def choose_shortlist(rows, targets):
    choices = []
    for documents in sorted({r['documents'] for r in rows}):
        for target in targets:
            for method in ('scan','branch','keys'):
                for selection, metric in [('mean','recall'), ('conservative','recall_low')]:
                    eligible = [r for r in rows if r['documents']==documents and r['method']==method
                                and r[metric] is not None and r[metric]>=target]
                    if eligible:
                        chosen = min(eligible, key=lambda r:(r['p50_ms'],r['setting_id']))
                        choices.append(dict(selection=selection, target=target, **chosen))
    return choices


def read_cohorts(folder, cases):
    target = folder/'cohorts.json'
    if target.exists():
        return json.loads(target.read_text())
    cohorts = {pool:json.loads((Path(pool)/'pool.json').read_text())
               for pool in sorted({c['case']['pool'] for c in cases})}
    target.write_text(json.dumps(cohorts, indent=2)+'\n')
    return cohorts


def combine(folders, output, targets, bootstrap_samples=1000, seed=51):
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    identities = defaultdict(set)
    failures = []
    for folder in folders:
        cases, failed = load_cases(folder)
        failures.extend(dict(source=str(folder), **row) for row in failed)
        cohorts = read_cohorts(folder, cases)
        for item in cases:
            case, result = item['case'], item['result']
            cohort = cohorts[case['pool']]
            ids = tuple(cohort['query_ids'][i] for i in case['query_rows'])
            identities[case['documents']].add((ids, cohort['hashes']['codes.npy'],
                                               cohort['hashes']['queries.npy'], cohort['hashes']['reference.npy']))
            power = result['power_before']
            if not power['available'] or power!=result['power_after'] or result['memory']['budget_status']!='fits_by_lifetime_peak':
                failures.append(dict(source=str(folder), case_id=item['case_id'], reason='power_or_memory_not_qualified'))
                continue
            probes = effective_probes(case, result)
            key = setting_key(case, probes)
            if key not in groups:
                canonical = dict(case, probes=probes)
                canonical.pop('seed', None)
                groups[key] = dict(case=canonical, times=[], quality={}, sources=[], seeds=set(),
                                   native=set(), workers=set(), lifetime_peak=0)
            group = groups[key]
            active_seed = case.get('seed',0) if case.get('exploration',0)>0 else 0
            group['seeds'].add(active_seed)
            group['native'].add(result['native_sha256'])
            group['workers'].add(result['worker_sha256'])
            group['lifetime_peak'] = max(group['lifetime_peak'], result['memory']['lifetime_peak_bytes'])
            group['sources'].append(dict(folder=str(folder), case_id=item['case_id'], seed=active_seed))
            for row in item['queries']:
                quality_key = (row['query'], active_seed)
                if quality_key in group['quality']:
                    assert group['quality'][quality_key]==row['recall'], 'quality changed across identical repeats'
                group['quality'][quality_key] = row['recall']
                group['times'].append(row['query_ms'])
        del cases
    if any(len(values)!=1 for values in identities.values()):
        raise ValueError('do not combine different query populations or references at the same corpus size')
    rng = np.random.default_rng(seed)
    rows, details = [], {}
    for number, group in enumerate(groups.values()):
        assert len(group['native'])==len(group['workers'])==1
        case = group['case']
        query_numbers = sorted({key[0] for key in group['quality']})
        seeds = sorted(group['seeds'])
        assert len(group['quality'])==len(query_numbers)*len(seeds), 'unequal seed coverage'
        hits = np.array([sum(round(group['quality'][(q,s)]*case['top_k']) for s in seeds) for q in query_numbers])
        denominator = len(query_numbers)*len(seeds)*case['top_k']
        mean = float(hits.sum()/denominator)
        lower = None
        if mean >= min(targets):
            samples = rng.choice(hits, size=(bootstrap_samples,len(hits)), replace=True).sum(axis=1)/denominator
            lower = float(np.quantile(samples,.025))
        setting_id = f's{number:04d}'
        row = dict(setting_id=setting_id, documents=case['documents'], method=case['method'],
                   router=case['router_kind'], clusters=case['clusters'], probes=case['probes'],
                   node_budget=case.get('node_budget',0), leaf_size=case.get('leaf_size',0),
                   key_bits=case.get('key_bits',0), candidate_target=case.get('candidate_target',0),
                   exploration=case.get('exploration',0), seeds=','.join(map(str,seeds)),
                   recall=mean, recall_low=lower, p50_ms=float(np.median(group['times'])),
                   p95_ms=float(np.percentile(group['times'],95)), sources=len(group['sources']),
                   lifetime_peak_bytes=group['lifetime_peak'])
        rows.append(row)
        details[setting_id] = dict(case=case, evaluation_seeds=seeds, sources=group['sources'])
    selected = choose_shortlist(rows, targets)
    unique_ids = sorted({r['setting_id'] for r in selected})
    shortlist = []
    for identifier in unique_ids:
        uses = [dict(target=r['target'],selection=r['selection']) for r in selected if r['setting_id']==identifier]
        shortlist.append(dict(setting_id=identifier, uses=uses, **details[identifier]))
    write_csv(output/'settings.csv',rows)
    write_csv(output/'selected.csv',selected)
    (output/'shortlist.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    (output/'all-settings.json').write_text(json.dumps(details,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(
        created_at=datetime.now(timezone.utc).isoformat(), sources=list(map(str,folders)),
        targets=targets, bootstrap_samples=bootstrap_samples, seed=seed, failures=failures,
        scope='Frozen from tuning data. Duplicate timings pooled; seeds averaged per query. The conservative choice uses a training bootstrap percentile, not a population guarantee.'),indent=2)+'\n')
    print(f'Combined {len(groups)} unique settings; froze {len(shortlist)} distinct evaluation settings.')
    return rows, shortlist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--targets',nargs='+',type=float,default=[.8,.9,.95,.99])
    args=parser.parse_args()
    combine(args.folders,args.output,args.targets)


if __name__=='__main__':
    main()
