"""The evaluation changes query inputs, not the frozen search settings."""
import hashlib
import json

import numpy as np
import pytest

from experiments.components import pack_signs
from experiments.evaluate import run_evaluation
from experiments.real_study import prepare_real_pool


@pytest.mark.parametrize('query_count',[2,3])
def test_frozen_setting_is_replayed_on_disjoint_queries(tmp_path,query_count):
    source=tmp_path/'embedding'
    derived=source/'derived'/'5'
    derived.mkdir(parents=True)
    rng=np.random.default_rng(9)
    documents=rng.normal(size=(32,5)).astype(np.float32)
    old_queries=rng.normal(size=(2,5)).astype(np.float32)
    np.save(source/'full-documents.npy',documents)
    np.save(derived/'codes.npy',pack_signs(documents>=0))
    np.save(derived/'queries.npy',old_queries)
    (source/'manifest.json').write_text('{}')
    (derived/'manifest.json').write_text(json.dumps({'query_ids':['old0','old1']}))
    old_config={'data':{'embedding_dir':str(source),'cache_dir':str(tmp_path/'old-pool'),
                        'dimensions':5,'queries':2,'query_start':0},'search':{'top_k':3}}
    old_pool=prepare_real_pool(old_config,32)
    original=dict(method='scan',documents=32,dimensions=5,top_k=3,pool=str(old_pool),router=None,
                  router_kind='none',clusters=1,probes=1,query_rows=[0,1],repetitions=1,
                  ram_budget_bytes=32_000_000_000)
    choices=tmp_path/'shortlist.json'
    choices.write_text(json.dumps([{'setting_id':'s0','uses':[{'target':.95,'selection':'mean'}],
                                    'evaluation_seeds':[0],'case':original}]))
    fresh=tmp_path/'fresh'
    fresh.mkdir()
    np.save(fresh/'queries.npy',rng.normal(size=(3,5)).astype(np.float32))
    (fresh/'manifest.json').write_text(json.dumps({'query_ids':['new0','new1','new2'],
        'queries_sha256':hashlib.sha256((fresh/'queries.npy').read_bytes()).hexdigest()}))
    config={'experiment':{'shortlist':str(choices)},
            'data':dict(old_config['data'],cache_dir=str(tmp_path/'new-pool'),query_dir=str(fresh),queries=query_count),
            'search':{'top_k':3},'measurement':{'repetitions':2,'schedule_seed':1},
            'limits':{'ram_budget_bytes':32_000_000_000,'worker_stop_bytes':34_359_738_368,'case_seconds':20}}
    output=tmp_path/'evaluation'
    run_evaluation(config,output)
    case=json.loads((output/'cases/0000/case.json').read_text())
    assert case['query_rows']==list(range(query_count)) and case['repetitions']==2
    for name in ('method','documents','dimensions','top_k','router','clusters','probes'):
        assert case[name]==original[name]
    result=json.loads((output/'cases/0000/run/result.json').read_text())
    assert result['exact_checks']==2*query_count
    source_record=json.loads((output/'evaluation-source.json').read_text())
    assert source_record['query_ids']==['new0','new1','new2'][:query_count]
