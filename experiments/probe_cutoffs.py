"""Derive probe counts from tuning-neighbor cluster ranks, then measure them."""
import csv
import json
import math
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.choices import setting_key
from experiments.isolated import ROOT, execute_cases


def probe_cutoff(neighbor_ranks, target):
    values=np.sort(np.asarray(neighbor_ranks).ravel())
    return int(values[math.ceil(target*len(values))-1])


def cover_boundary_ties(scores, probes):
    # Include an entire tied score group. Increasing P may reach another query's
    # tie, so repeat until every query has an unambiguous selected prefix.
    while probes < scores.shape[1]:
        extended=int(np.max(np.sum(scores>=scores[:,probes-1,None],axis=1)))
        if extended==probes:
            break
        probes=extended
    return probes


def routing_ranks(case):
    pool=Path(case['pool'])
    reference=np.load(pool/'reference.npy')[case['query_rows']]
    queries=np.load(pool/'queries.npy')[case['query_rows']]
    if case['router'] is None:
        return np.ones_like(reference), None, None, None, None
    folder=Path(case['router'])
    centroids=np.load(folder/'centroids.npy')
    labels=np.load(folder/'labels.npy')
    assignments=np.load(folder/'assignments.npy',mmap_mode='r')
    true_clusters=assignments[reference]
    quantizer=faiss.IndexFlatIP(centroids.shape[1])
    quantizer.add(centroids)
    faiss.omp_set_num_threads(1)
    ranking=np.empty_like(reference)
    all_scores=np.empty((len(queries),len(labels)),dtype=np.float32)
    route_queries=np.ascontiguousarray(queries[:,:quantizer.d])
    for qi,query in enumerate(route_queries):
        scores,positions=quantizer.search(query[None,:],len(labels))
        inverse=np.zeros(int(labels.max())+1,dtype=np.int64)
        inverse[labels[positions[0]]]=np.arange(1,len(labels)+1)
        ranking[qi]=inverse[true_clusters[qi]]
        all_scores[qi]=scores[0]
    assert np.all(ranking>0)
    return ranking,all_scores,quantizer,route_queries,(labels,true_clusters)


def run_probe_cutoffs(config,output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    layouts={}
    for source in config['experiment']['sources']:
        folder=ROOT/source
        schedule=json.loads((folder/'schedule.json').read_text())
        with (folder/'comparison/settings.csv').open() as file:
            rows=list(csv.DictReader(file))
        for row in rows:
            if row['power_unchanged']!='True' or row['budget_status']!='fits_by_lifetime_peak':
                continue
            case=schedule[int(row['case_id'])]
            key=(case['pool'],case['router'])
            layouts.setdefault(key,dict(base=case,observations=[]))['observations'].append((case,row))
    unique_cases={}
    oracle=[]
    for layout in layouts.values():
        base=layout['base']
        with threadpool_limits(limits=1):
            ranks,scores,quantizer,queries,lookup=routing_ranks(base)
        for target in config['search']['recall_targets']:
            proposed=probe_cutoff(ranks,target)
            probes=cover_boundary_ties(scores,proposed) if scores is not None else 1
            hits=int(np.sum(ranks<=probes))
            if quantizer is not None:
                labels,true_clusters=lookup
                measured_hits=0
                for qi,query in enumerate(queries):
                    _,positions=quantizer.search(query[None,:],probes)
                    measured_hits+=int(np.isin(true_clusters[qi],labels[positions[0]]).sum())
            else:
                measured_hits=ranks.size
            record=dict(documents=base['documents'],router=base['router_kind'],clusters=base['clusters'],
                        target=target,rank_cutoff=proposed,tie_safe_probes=probes,
                        predicted_recall=hits/ranks.size,verified_router_recall=measured_hits/ranks.size)
            oracle.append(record)
            (output/'probe-oracle.json').write_text(json.dumps(oracle,indent=2)+'\n')
            if measured_hits < math.ceil(target*ranks.size):
                raise ValueError('rank cutoff disagrees with the actual router; inspect probe-oracle.json')
            common={name:base[name] for name in ('pool','documents','dimensions','router','router_kind',
                                                'clusters','top_k','query_rows','ram_budget_bytes')}
            common.update(probes=probes,repetitions=config['measurement']['repetitions'])
            candidates=[dict(common,method='scan'),
                        dict(common,method='branch',node_budget=0,leaf_size=1024),
                        dict(common,method='keys',key_bits=4,
                             key_offset=int(math.log2(base['clusters'])) if base['router_kind']=='sign' else 0,
                             candidate_target=0,key_limit=0)]
            # Also replay the best previously observed local parameters for this
            # layout. All methods get the derived probe count, not just scan.
            for method in ('branch','keys'):
                eligible=[(case,row) for case,row in layout['observations']
                          if case['method']==method and float(row['recall'])>=target]
                if eligible:
                    old,_=min(eligible,key=lambda pair:float(pair[1]['p50_ms']))
                    candidates.append(dict(old,probes=probes,repetitions=config['measurement']['repetitions']))
            for case in candidates:
                unique_cases[setting_key(case,probes)]=case
    cases=list(unique_cases.values())
    order=np.random.default_rng(config['measurement']['schedule_seed']).permutation(len(cases))
    cases=[cases[int(i)] for i in order]
    print(f'Derived {len(oracle)} routing cutoffs; measuring {len(cases)} unique configurations.',flush=True)
    execute_cases(config,output,cases,'Probe counts derived from tuning-neighbor cluster ranks and verified with the actual router; not fresh evaluation.')
