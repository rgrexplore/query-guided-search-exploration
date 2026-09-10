"""Repeated calls do not become independent quality queries."""
import numpy as np

from experiments.evaluation_report import summarize_setting


def test_summary_keeps_query_quality_and_process_blocks_separate():
    items=[]
    for block,time in [(0,1.0),(1,2.0)]:
        rows=[dict(query=q,repetition=r,recall=recall,query_ms=time)
              for r in range(2) for q,recall in [(0,.8),(1,1.0)]]
        items.append({'case':{'seed':0,'block':block,'top_k':100},'queries':rows})
    qdraw=np.array([[0,1],[0,0],[1,1]])
    bdraw=np.array([[0,1],[0,0],[1,1]])
    summary,samples=summarize_setting(items,[0],qdraw,bdraw)
    assert summary['recall']==.9 and summary['queries']==2
    assert summary['blocks']==2 and summary['timing_observations']==8
    np.testing.assert_array_equal(samples,[1.5,1.0,2.0])
