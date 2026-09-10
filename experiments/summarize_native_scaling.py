"""Fit the prescribed native paths on1M/2M and check their transfer to4M."""
import argparse
import csv
import statistics
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from experiments.analysis import load_cases
from experiments.choices import read_cohorts
from experiments.isolated import ROOT
from experiments.models import predict_native_cost as predict


def fit_median_scaling(rows):
    """The fitted target is one median per size, not a weighted average query."""
    sizes=sorted({r['documents'] for r in rows})
    x=np.array([[1,n/1000000] for n in sizes])
    y=np.array([np.median([r['query_ms'] for r in rows if r['documents']==n]) for n in sizes])
    coefficients,_=nnls(x/y[:,None],np.ones(len(y)))
    return coefficients


def streaming_row_cost():
    """Use the large contiguous scorer run where fixed heap work is amortized."""
    folder=ROOT/'results/buffer-costs-2026-09-11'
    for path in folder.glob('[0-9][0-9][0-9]/measurements.csv'):
        with path.open() as file:
            rows=list(csv.DictReader(file))
        if rows[0]['operation']=='score' and rows[0]['size']=='4000000' and rows[0]['dimensions']=='256':
            return statistics.median(float(r['milliseconds']) for r in rows)/4000000
    raise ValueError('missing contiguous score calibration')


def score_and_selection(rows, row_cost):
    """For small postings, top-K replacement work grows roughly as K log(F/K)."""
    sizes=sorted({r['documents'] for r in rows})
    counts=np.array([np.median([r['documents_scored'] for r in rows if r['documents']==n]) for n in sizes])
    times=np.array([np.median([r['query_ms'] for r in rows if r['documents']==n]) for n in sizes])
    x=np.column_stack([np.ones(len(sizes)),np.log(np.maximum(counts/100,1))])
    coefficients,_=nnls(x/times[:,None],(times-counts*row_cost)/times)
    return dict(kind='score_and_selection',fixed_ms=float(coefficients[0]),
                per_scored_row_ms=row_cost,per_log_ratio_ms=float(coefficients[1]),top_k=100)


def summarize(folder):
    cases, failures = load_cases(folder)
    cohorts=read_cohorts(folder,cases)
    assert not failures
    models=[]; summaries=[]
    for support in ('fixed','adaptive'):
        selected=[item for item in cases if item['case']['support']==support]
        eligible=set(range(8))
        for item in selected:
            case=item['case']
            meta=cohorts[case['pool']]
            eligible &= {q for q,count in enumerate(meta['strong_match_counts']) if count>=100}
            if case['method']=='branch':
                words=(case['documents']+63)//64
                eligible &= {r['query'] for r in item['queries'] if r['repetition']==0
                             and r['bitplane_words']==13*words and r['leaf_words']==words and r['nodes']==14}
        assert eligible, 'No queries satisfy the stated single-path condition at all sizes.'
        for method in ('scan','branch','keys'):
            items=[item for item in selected if item['case']['method']==method]
            observations=[]
            for item in items:
                case=item['case']
                rows=item['queries']
                assert all(r['recall']==1 for r in rows)
                assert item['result']['power_before']['available'] and item['result']['power_before']==item['result']['power_after']
                assert item['result']['memory']['budget_status']=='fits_by_lifetime_peak'
                applicable=[r for r in rows if r['query'] in eligible]
                summaries.append(dict(support=support,method=method,documents=case['documents'],
                    all_query_p50_ms=float(np.median([r['query_ms'] for r in rows])),
                    applicable_p50_ms=float(np.median([r['query_ms'] for r in applicable])),
                    applicable_queries=sorted(eligible),all_queries=8,
                    lifetime_peak_bytes=item['result']['memory']['lifetime_peak_bytes']))
                observations.extend(applicable)
            train=[r for r in observations if r['documents'] in (1000000,2000000)]
            if support=='fixed' and method=='keys':
                model=score_and_selection(train,streaming_row_cost())
            else:
                coefficients=fit_median_scaling(train)
                model=dict(kind='linear_size',fixed_ms=float(coefficients[0]),per_million_ms=float(coefficients[1]))
            check_rows=[r for r in observations if r['documents']==4000000]
            observed=float(np.median([r['query_ms'] for r in check_rows]))
            predicted=float(np.median([predict(model,r) for r in check_rows]))
            models.append(dict(support=support,method=method,**model,check_documents=4000000,
                measured_ms=observed,predicted_ms=predicted,error_percent=100*(predicted/observed-1),
                training_sizes=[1000000,2000000],applicable_queries=sorted(eligible)))
    result=dict(models=models,measurements=summaries,
        revision='Median target and small-posting selection term. Prior fits retained; the revised4M check is diagnostic. Freeze this before the8M check.',
        scope='Conditional fixed-depth paths, fitted on1M/2M only and checked at4M. All queries remain in measurements. Applicability depends on strong-match counts and native work, not latency. This is not proof of linear transfer to1B or of optimal routing at4M.')
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(models,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    summarize(parser.parse_args().folder)
