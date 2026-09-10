"""Canonical settings must not depend on nominal duplicate probes or lucky seeds."""
import experiments.choices as choices
from experiments.choices import effective_probes, setting_key, choose_shortlist


def test_sign_probe_count_is_clipped_to_the_stored_route_count():
    case = dict(router_kind='sign', clusters=16, probes=16, dimensions=256)
    result = {'memory': {'routing_payload_bytes': 7*(4*4+8)}}
    assert effective_probes(case, result)==7


def test_setting_key_ignores_seed_and_timing_repeat_but_not_algorithm_options():
    case = dict(method='branch', documents=1000, dimensions=256, top_k=100,
                pool='/pool', router=None, router_kind='none', clusters=1, probes=1,
                node_budget=512, leaf_size=32, exploration=.1, seed=7, repetitions=1)
    other = dict(case, seed=11, repetitions=3)
    assert setting_key(case,1)==setting_key(other,1)
    assert setting_key(case,1)!=setting_key(dict(case, leaf_size=128),1)


def test_shortlist_uses_mean_and_conservative_training_choices():
    rows = [dict(setting_id='s0', documents=1000, method='scan', recall=.96, recall_low=.93, p50_ms=1),
            dict(setting_id='s1', documents=1000, method='scan', recall=.99, recall_low=.97, p50_ms=2)]
    chosen=choose_shortlist(rows,[.95])
    assert [(row['selection'], row['setting_id']) for row in chosen]==[('mean','s0'),('conservative','s1')]


def test_duplicate_runs_pool_timings_instead_of_selecting_the_fastest(tmp_path, monkeypatch):
    case = dict(method='scan', documents=1000, dimensions=256, top_k=100, pool='/pool',
                router=None, router_kind='none', clusters=1, probes=1, query_rows=[0,1], repetitions=1)
    power = {'available':True, 'source':'AC'}
    result = dict(power_before=power,power_after=power,native_sha256='native',worker_sha256='worker',
                  memory={'budget_status':'fits_by_lifetime_peak','lifetime_peak_bytes':1000})
    def load(folder):
        times = [1,3] if folder.name=='first' else [2,4]
        rows = [dict(query=q,recall=.9,query_ms=t) for q,t in enumerate(times)]
        return [dict(case_id=5,case=case,result=result,queries=rows)], []
    monkeypatch.setattr(choices,'load_cases',load)
    monkeypatch.setattr(choices,'read_cohorts',lambda folder,cases:{'/pool':{
        'query_ids':['q0','q1'], 'hashes':{'codes.npy':'c','queries.npy':'q','reference.npy':'r'}}})
    rows, shortlist = choices.combine([tmp_path/'first',tmp_path/'second'],tmp_path/'combined',[.8],100)
    assert len(rows)==1 and rows[0]['p50_ms']==2.5
    assert rows[0]['sources']==2 and shortlist[0]['evaluation_seeds']==[0]
