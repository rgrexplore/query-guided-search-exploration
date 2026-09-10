"""Check saved native cost coefficients without changing them."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.analysis import load_cases
from experiments.choices import read_cohorts
from experiments.models import predict_native_cost as predict


def check(folder, model_path):
    models=json.loads(model_path.read_text())['models']
    cases, failures=load_cases(folder)
    cohorts=read_cohorts(folder,cases)
    assert not failures
    rows=[]
    for item in cases:
        case=item['case']
        model=next(m for m in models if m['support']==case['support'] and m['method']==case['method'])
        observations=[r for r in item['queries'] if r['query'] in model['applicable_queries']]
        assert all(r['recall']==1 for r in item['queries'])
        meta=cohorts[case['pool']]
        assert all(meta['strong_match_counts'][q]>=case['top_k'] for q in model['applicable_queries'])
        if case['method']=='branch':
            words=(case['documents']+63)//64
            assert all(r['nodes']==14 and r['bitplane_words']==13*words and r['leaf_words']==words for r in observations)
        actual=float(np.median([r['query_ms'] for r in observations]))
        estimate=float(np.median([predict(model,r) for r in observations]))
        rows.append(dict(support=case['support'],method=case['method'],documents=case['documents'],
                         measured_ms=actual,predicted_ms=estimate,error_percent=100*(estimate/actual-1),
                         within_twenty_percent=abs(estimate/actual-1)<=.2))
    result=dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),rows=rows,
                scope='Frozen model checked at a larger corpus size; no refit. Same prescribed query distribution and existing query IDs; this checks size transfer, not new-query generalization.')
    (folder/'frozen-model-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--models',required=True,type=Path)
    args=parser.parse_args()
    check(args.folder,args.models)
