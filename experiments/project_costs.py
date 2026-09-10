"""Conditional larger-corpus scenarios; these are forecasts, not1B benchmarks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import binom

from experiments.models import predict_native_cost as predict

ROOT = Path(__file__).resolve().parents[1]


def prefix_controls():
    folder=ROOT/'results/controlled-fixed-2026-09-10/tuning'
    schedule=json.loads((folder/'schedule.json').read_text())
    times={}
    for i,case in enumerate(schedule):
        if case['documents']!=1000000: continue
        eligible=(case['method']=='keys' and case['router_kind']=='none' and case.get('key_bits')==13)
        eligible |= (case['method']=='scan' and case['router_kind']=='direct' and case['clusters']==8192 and case['probes']==1)
        if eligible:
            rows=[json.loads(line) for line in (folder/'cases'/f'{i:04d}'/'run/queries.jsonl').read_text().splitlines()]
            times[case['method']]={q:statistics.median(r['query_ms'] for r in rows if r['query']==q) for q in range(20)}
    route_extra=statistics.median(times['scan'][q]-times['keys'][q] for q in range(20))
    kernel={}
    for path in (ROOT/'results/prefix-scan-cost-2026-09-11').glob('[0-9][0-9][0-9]/measurements.csv'):
        with path.open() as file: rows=list(csv.DictReader(file))
        if rows[0]['operation'] in ('score','gather'):
            kernel[rows[0]['operation']]=statistics.median(float(r['milliseconds']) for r in rows)
    return route_extra,kernel['gather']/kernel['score']


def key_support_probabilities(dimensions, strong_bits, key_bits):
    denominator=math.comb(dimensions,key_bits)
    return [(s,math.comb(strong_bits,s)*math.comb(dimensions-strong_bits,key_bits-s)/denominator)
            for s in range(min(strong_bits,key_bits)+1)]


def weighted_quantile(values, probability):
    cumulative=0
    for value,weight in sorted(values):
        cumulative+=weight
        if cumulative>=probability: return value
    return max(v for v,_ in values)


def memory_case(n, d, method, depth, strong_bits, ram, reserve):
    code = n*math.ceil(d/64)*8
    ids = n*8
    branch = method.startswith('branch')
    prefix = method in ('scan_prefix','branch_prefix')
    clusters = min(n,2**strong_bits) if prefix else 1
    planes_min = d*math.ceil(n/64)*8 if branch else 0
    keys_upper = min(n,2**strong_bits) if method=='keys' else 0
    # This lower bound does not assume every possible key is occupied.
    payload_lower = code+ids+planes_min
    planes_plan = d*(math.ceil(n/64)+clusters-1)*8 if branch else 0
    index_plan = code+2*ids+planes_plan+80*keys_upper+512*clusters
    if method=='branch':
        scratch=(depth+2)*math.ceil(n/64)*8
    elif method=='branch_prefix':
        scratch=math.ceil(2*n/clusters/64)*8+65536
    else:
        scratch=65536
    total=index_plan+scratch+reserve
    status='payload_exceeds_budget' if payload_lower>ram else 'modeled_fit' if total<=ram else 'allowance_exceeds_budget'
    return dict(payload_lower_bytes=payload_lower,planning_index_bytes=index_plan,scratch_bytes=scratch,
                reserve_bytes=reserve,planning_total_bytes=total,status=status)


def ideal_recall(n, strong_bits, top_k):
    probability = 2.0**-strong_bits
    recall = float(np.mean(binom.sf(np.arange(top_k), n, probability)))
    mean = n*probability
    # Logarithmic Chernoff bound remains meaningful when the exact floating
    # point CDF underflows. It bounds insufficient matches, not runtime.
    log_bound = -(mean-(top_k-1))**2/(2*mean*math.log(10)) if mean>=top_k-1 else 0
    return dict(expected_recall=recall,expected_matches=mean,
                insufficient_probability_upper_log10=log_bound)



def project(output, sizes, budgets, reserve_gb, inputs=None):
    output.mkdir(parents=True,exist_ok=False)
    if inputs is None:
        model_path=ROOT/'results/native-scale-check-2026-09-11/frozen-models.json'
        models=json.loads(model_path.read_text())['models']
        model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
        route_extra,leaf_ratio=prefix_controls()
        work_model=json.loads((ROOT/'results/isolated-calibration-2026-09-10/analysis-relative/models.json').read_text())['keys']['coefficients']
    else:
        saved=json.loads(inputs.read_text())
        models=saved['native_models']; model_hash=saved['model_sha256']
        route_extra=saved['route_adjustment_ms']; leaf_ratio=saved['scan_like_leaf_ratio']
        work_model=saved['key_coefficients']
    by_method={(m['support'],m['method']):m for m in models}
    rows=[]; key_rows=[]
    for n in sizes:
        quality=ideal_recall(n,13,100)
        matching=quality['expected_matches']
        full_scan=predict(by_method[('adaptive','scan')],dict(documents=n))
        fixed_key=predict(by_method[('fixed','keys')],dict(documents_scored=matching))
        # Same13-bit restriction is available to A. B can also choose a large
        # leaf and scan; do not force B to traverse the whole corpus in this case.
        prefix_scan=fixed_key+route_extra
        prefix_branch=fixed_key*leaf_ratio+route_extra
        for branch_scale in (1,2):
            b=by_method[('adaptive','branch')]
            global_branch=b['fixed_ms']+branch_scale*b['per_million_ms']*n/1000000
            for ram_gb in budgets:
                for method,time in [('scan_all',full_scan),('scan_prefix',prefix_scan),
                                    ('keys',fixed_key),('branch',global_branch),('branch_prefix',prefix_branch)]:
                    memory=memory_case(n,256,method,13,13,ram_gb*1e9,reserve_gb*1e9)
                    applicable=matching>=100 or method=='scan_all'
                    rows.append(dict(documents=n,dimensions=256,top_k=100,ram_gb=ram_gb,
                        branch_scale=branch_scale,method=method,modeled_ms=float(time) if applicable else None,
                        expected_matching_rows=matching,expected_prefix_recall=quality['expected_recall'],
                        insufficient_probability_upper_log10=quality['insufficient_probability_upper_log10'],
                        leaf_size=round(160*n/1000000) if method=='branch' else math.ceil(2*n/8192) if method=='branch_prefix' else 0,
                        break_even_scan_fraction=global_branch/full_scan,**memory))
        for h in range(1,25):
            distribution=[]
            for captured,probability in key_support_probabilities(256,13,h):
                found=n/2**captured
                attempts=2**(h-captured)
                cost=(work_model['fixed']+work_model['score_terms']*64*found
                      +work_model['key_attempts']*attempts
                      +work_model['key_queue_work']*attempts*math.log2(attempts+2))
                distribution.append((cost,probability))
            probs=key_support_probabilities(256,13,h)
            key_rows.append(dict(documents=n,key_bits=h,no_strong_coordinate_probability=probs[0][1],
                expected_scored_fraction=sum(p*2**-s for s,p in probs),
                mean_modeled_ms=sum(t*p for t,p in distribution),
                median_modeled_ms=weighted_quantile(distribution,.5),p95_modeled_ms=weighted_quantile(distribution,.95)))
    result=dict(rows=rows,key_work_model=key_rows,route_adjustment_ms=route_extra,
        scan_like_leaf_ratio=leaf_ratio,model_sha256=model_hash,
        native_models=models,key_coefficients=work_model,
        assumptions=dict(dimensions=256,strong_bits=13,strong_weight=1,weak_weight=.0001,top_k=100,
                         runtime_reserve_gb=reserve_gb,row_capacity_allowance=2,directory_bytes_per_key_allowance=80),
        scope='Conditional average-occupancy scenarios. Native cost models were checked to8M;1B is an extrapolation. Doubling B variable cost is sensitivity, not a confidence interval. Fixed-prefix rows assume the indexed13coordinates are the important ones. B may use the same route and a scan-like leaf. For changing supports, key work assumes all weak-key patterns with matching indexed strong signs must be visited; this is not arbitrary-IVF recall or globally optimal routing. RAM fits are planning estimates; only payload-over-budget is a proven rejection.')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,values in [('scenarios',rows),('key-work',key_rows)]:
        with (output/f'{name}.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps([r for r in rows if r['documents']==1000000000 and r['ram_gb']==1000 and r['branch_scale']==1],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--documents',nargs='+',type=int,default=[1000000,100000000,500000000,1000000000])
    parser.add_argument('--ram-gb',nargs='+',type=float,default=[32,64,1000])
    parser.add_argument('--reserve-gb',type=float,default=1)
    parser.add_argument('--inputs',type=Path,help='Saved result bundle; recalculate without benchmark files or native extension.')
    args=parser.parse_args()
    project(args.output,args.documents,args.ram_gb,args.reserve_gb,args.inputs)
