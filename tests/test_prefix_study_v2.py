"""The study must vary settings without changing benchmark inputs."""
from experiments import prefix_study_v2 as study
import json
from pathlib import Path
from time import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.components import pack_signs, reference_rows
from experiments import real_study
from experiments.prefix_batch_worker_v2 import file_hash


def test_every_method_gets_the_same_routing_choices_and_queries():
    assert hasattr(study, 'build_jobs'), 'the fixed-data schedule is not implemented'
    layouts = [dict(path=None, clusters=1, probes={'1': [1], '100': [1]}),
               dict(path='/ivf-64', clusters=64, probes={'1': [2, 8, 64], '100': [4, 16, 64]})]
    jobs = study.build_jobs('/same-pool', layouts, [1, 100], 'main')
    assert {job['pool'] for job in jobs} == {'/same-pool'}
    assert all(job['query_rows'] == list(range(200)) for job in jobs)
    assert all(job['dimensions'] == 256 and job['ram_budget_bytes'] == 32_000_000_000 for job in jobs)
    for layout in layouts:
        own = [job for job in jobs if job['router'] == layout['path']]
        assert {job['method'] for job in own} == {'scan', 'branch', 'prefix'}
        for k in [1, 100]:
            expected = set(layout['probes'][str(k)])
            for job in own:
                assert {v['probes'] for v in job['variants'] if v['top_k'] == k} == expected


def test_fast_wrong_or_over_budget_results_cannot_win():
    assert hasattr(study, 'choose_settings'), 'selection is not implemented'
    rows = [dict(setting_id='wrong',method='prefix',top_k=100,recall=.2,p50_ms=.01,qualified=True),
            dict(setting_id='fits',method='prefix',top_k=100,recall=.99,p50_ms=1.0,qualified=True),
            dict(setting_id='memory',method='prefix',top_k=100,recall=1.0,p50_ms=.1,qualified=False),
            dict(setting_id='other-k',method='prefix',top_k=1,recall=.99,p50_ms=.001,qualified=True)]
    chosen=study.choose_settings(rows,[.99],[100])
    assert len(chosen)==1 and chosen[0]['setting_id']=='fits'
    assert len(rows)==4


@pytest.fixture
def config(tmp_path):
    signs = np.array([[1,1,0,0], [1,0,1,1], [0,0,0,0], [1,1,1,1]], dtype=np.uint8)
    queries = np.array([[.6,.5,.4,.3], [0,0,0,0]], dtype=np.float32)
    pool = tmp_path/'pool'; pool.mkdir()
    for name, value in dict(codes=pack_signs(signs), queries=queries,
                            reference=reference_rows(signs, queries, 2),
                            documents=(2*signs.astype(np.float32)-1)).items():
        np.save(pool/f'{name}.npy', value)
    metadata = dict(identity=dict(documents=4, dimensions=4, queries=2, top_k=2),
                    hashes={name:file_hash(pool/name) for name in
                            ('codes.npy','queries.npy','reference.npy','documents.npy')},
                    query_ids=['q0','q1'])
    (pool/'pool.json').write_text(json.dumps(metadata))
    return dict(pool=str(pool), cluster_counts=[1], top_ks=[1,2], recall_targets=[1.0],
                probe_targets=[.5,1.0], repetitions=1, ram_budget_bytes=32_000_000_000,
                global_seconds=60, job_seconds=10, schedule_seed=7,
                branch=dict(leaf_sizes=[1],node_budgets=[0]),
                prefix=dict(max_prefix_bits=4,start_depths=[1,4],candidate_targets=[1,0]))


def test_explicit_pool_shape_and_declared_grid_are_preserved(config):
    assert hasattr(study, 'build_jobs')
    layouts = [dict(path=None,clusters=1,probes={'1':[1],'2':[1]})]
    args = dict(shape=dict(documents=4,dimensions=4,queries=2), branch=config['branch'],
                prefix=config['prefix'],schedule_seed=7)
    jobs = study.build_jobs(config['pool'], layouts, [1,2], 'main', **args)
    assert jobs == study.build_jobs(config['pool'], layouts, [1,2], 'main', **args)
    assert all(job['query_rows']==[0,1] and job['dimensions']==4 for job in jobs)
    assert sorted(len(job['variants']) for job in jobs)==[2,2,8]
    assert len({v['setting_id'] for job in jobs for v in job['variants']})==12


def test_prepare_verifies_hashes_and_keeps_every_cached_query(config, tmp_path):
    assert hasattr(study, 'prepare')
    out = tmp_path/'study'
    study.prepare(config, out)
    manifest = json.loads((out/'inputs-manifest.json').read_text())
    assert manifest['identity']['queries']==2
    jobs = json.loads((out/'main/schedule.json').read_text())
    assert all(job['query_rows']==[0,1] for job in jobs)
    source = Path(config['pool'])/'queries.npy'
    values = np.load(source); values[0,0] += 1; np.save(source,values)
    with pytest.raises(ValueError,match='input|hash|changed'):
        study.run(out)
    assert not (out/'main/cases').exists()


def test_probe_cutoffs_use_each_k_and_cover_centroid_ties(config, tmp_path):
    assert hasattr(study, 'prepare')
    pool = Path(config['pool'])
    router = pool/'routers/ivf-2-seed42'; router.mkdir(parents=True)
    np.save(router/'assignments.npy',np.array([0,1,0,1],dtype=np.int64))
    np.save(router/'labels.npy',np.array([0,1],dtype=np.int64))
    np.save(router/'centroids.npy',np.array([[1,1,0,0],[1,0,1,1]],dtype=np.float32))
    (router/'router.json').write_text(json.dumps(dict(kind='ivf',clusters=2,seed=42,
        assignments_sha256=file_hash(router/'assignments.npy'),
        centroids_sha256=file_hash(router/'centroids.npy'))))
    config['cluster_counts']=[2]
    study.prepare(config,tmp_path/'study')
    layout=json.loads((tmp_path/'study/layouts.json').read_text())[0]
    # The zero query ties both centroids, so each proposed prefix must include both.
    assert layout['probes']=={'1':[2],'2':[2]}
    assert all(record['routing_recall']==1 for records in layout['cutoffs'].values()
               for record in records)


def test_serial_run_resume_summary_and_three_repeat_blocks(config,tmp_path):
    assert hasattr(study,'prepare')
    out=tmp_path/'study'
    study.prepare(config,out)
    study.run(out)
    processes=list((out/'main/cases').glob('*/process.json'))
    assert len(processes)==3
    assert all(json.loads(path.read_text())['status']=='complete' for path in processes)
    original={str(path):path.read_bytes() for path in processes}
    study.run(out)
    assert {str(path):path.read_bytes() for path in processes}==original
    # Supply a known stable power observation to test qualification on any host OS.
    for path in (out/'main/cases').glob('*/run/result.json'):
        result=json.loads(path.read_text())
        result['power_before']=result['power_after']={'available':True,'source':'AC'}
        path.write_text(json.dumps(result))
    rows=study.summarize(out)
    assert len(rows)==12 and all(row['complete'] for row in rows)
    assert all(row['p95_ms']>=row['p50_ms']>=0 for row in rows)
    assert all(row['total_query_ms']>=row['p50_ms'] for row in rows)
    selected=json.loads((out/'selected-settings.json').read_text())
    assert {(row['method'],row['top_k']) for row in selected}=={
        (method,k) for method in ('scan','branch','prefix') for k in (1,2)}
    frozen=(out/'selected-settings.json').read_bytes()
    study.repeat(out)
    repeats=json.loads((out/'repeat/schedule.json').read_text())
    assert {job['block'] for job in repeats}=={0,1,2}
    assert len(repeats)==9 and all(job['query_rows']==[0,1] for job in repeats)
    assert (out/'selected-settings.json').read_bytes()==frozen
    repeat_rows=json.loads((out/'repeat/settings.json').read_text())
    assert all(row['complete_processes']==3 for row in repeat_rows)
    assert all(row['process_p50_max_ms']>=row['process_p50_min_ms'] for row in repeat_rows)


def test_failed_jobs_are_saved_and_never_retried(config,tmp_path):
    assert hasattr(study,'prepare')
    config['worker_stop_bytes']=1
    out=tmp_path/'study'; study.prepare(config,out); study.run(out)
    paths=list((out/'main/cases').glob('*/process.json'))
    assert len(paths)==3
    assert all(json.loads(path.read_text())['status']=='memory_stop' for path in paths)
    original=[path.read_bytes() for path in paths]
    study.run(out)
    assert [path.read_bytes() for path in paths]==original
    assert not any(row['qualified'] for row in study.summarize(out))


def test_zero_global_budget_starts_no_process(config,tmp_path):
    assert hasattr(study,'prepare')
    config['global_seconds']=0
    out=tmp_path/'study'; study.prepare(config,out); study.run(out)
    assert not list((out/'main/cases').glob('*/process.json'))
    assert json.loads((out/'main/run-state.json').read_text())['status']=='budget_exhausted'


def test_resume_counts_a_dead_workers_elapsed_time_before_starting_another_job(config,tmp_path):
    config['global_seconds']=1
    out=tmp_path/'study'; study.prepare(config,out)
    job=json.loads((out/'main/schedule.json').read_text())[0]
    folder=out/'main/cases'/job['job_id']; folder.mkdir(parents=True)
    (folder/'process.json').write_text(json.dumps(dict(status='running',pid=999999999,
                                                     started_at=time()-100,seconds=0)))
    study.run(out)
    assert json.loads((folder/'process.json').read_text())['status']=='interrupted'
    assert len(list((out/'main/cases').glob('*/process.json')))==1
    assert json.loads((out/'time-budget.json').read_text())['spent_seconds']>=100


def test_summary_keeps_complete_query_rows_when_a_worker_was_killed_mid_write(config,tmp_path):
    out=tmp_path/'study'; study.prepare(config,out)
    job=json.loads((out/'main/schedule.json').read_text())[0]
    folder=out/'main/cases'/job['job_id']; (folder/'run').mkdir(parents=True)
    (folder/'process.json').write_text(json.dumps(dict(status='timeout',seconds=1)))
    (folder/'run/result.json').write_text('{"status":')
    identifier=job['variants'][0]['setting_id']
    (folder/'run/queries.jsonl').write_text(json.dumps(dict(setting_id=identifier,
        query=0,repetition=0,query_ms=1.2,recall=1.0))+'\n{"setting_id":')
    rows=study.summarize(out)
    row=next(row for row in rows if row['setting_id']==identifier)
    assert row['measurements']==1 and row['p50_ms']==1.2 and row['recall']==1
    assert row['qualified'] is False and row['complete'] is False
    assert (folder/'run/result.json').read_text()=='{"status":'


def test_explicit_policies_replace_grids_and_keep_declared_probes():
    layouts = [dict(path='/ivf-8', clusters=8, probes={'1': [2, 3]})]
    branch = dict(leaf_sizes=[100, 200], node_budgets=[1, 2], exploration=.1, seed=17,
                  settings=[dict(leaf_size=4, node_budget=8),
                            dict(leaf_size=32, node_budget=0, exploration=.5, seed=99,
                                 prefer_deeper_ties=True)])
    prefix = dict(max_prefix_bits=5, start_depths=[99], candidate_targets=[10, 20],
                  settings=[dict(start_depth=2, candidate_target=3),
                            dict(start_depth=4, candidate_target=0)])
    original = json.dumps((branch, prefix), sort_keys=True)
    jobs = study.build_jobs('/same-pool', layouts, [1], 'followup', branch=branch, prefix=prefix)
    assert {job['method']: len(job['variants']) for job in jobs} == {'scan': 2, 'branch': 4, 'prefix': 4}
    assert all({variant['probes'] for variant in job['variants']} == {2, 3} for job in jobs)
    variants = next(job['variants'] for job in jobs if job['method'] == 'branch')
    for variant in variants:
        if variant['leaf_size'] == 4:
            assert (variant['node_budget'], variant['exploration'], variant['seed']) == (8, .1, 17)
        else:
            assert (variant['leaf_size'], variant['node_budget'], variant['exploration'], variant['seed']) == (32, 0, .5, 99)
            assert variant['prefer_deeper_ties'] is True
    variants = next(job['variants'] for job in jobs if job['method'] == 'prefix')
    assert {(variant['start_depth'], variant['candidate_target']) for variant in variants} == {(2, 3), (4, 0)}
    assert json.dumps((branch, prefix), sort_keys=True) == original
    for depth in (-1, 6):
        invalid = dict(prefix, settings=[dict(start_depth=depth, candidate_target=3)])
        with pytest.raises(ValueError, match='depth'):
            study.build_jobs('/same-pool', layouts, [1], 'followup', prefix=invalid)


def test_optional_all_cluster_choice_preserves_default_and_global_reference(config, tmp_path, monkeypatch):
    pool = Path(config['pool'])
    router = pool/'routers/ivf-2-seed42'; router.mkdir(parents=True)
    np.save(router/'assignments.npy', np.array([0, 1, 0, 1], dtype=np.int64))
    np.save(router/'labels.npy', np.array([0, 1], dtype=np.int64))
    np.save(router/'centroids.npy', np.ones((2, 4), dtype=np.float32))
    (router/'router.json').write_text(json.dumps(dict(kind='ivf', clusters=2, seed=42,
        assignments_sha256=file_hash(router/'assignments.npy'),
        centroids_sha256=file_hash(router/'centroids.npy'))))
    # Hold the derived cutoff at one probe to isolate the optional all-cluster addition.
    monkeypatch.setattr(study, 'routing_ranks', lambda case:
        (np.ones((2, 2), dtype=np.int64), np.array([[2, 1], [2, 1]], dtype=np.float32)))
    for option, expected in ((None, [1, 2]), (False, [1]), (True, [1, 2])):
        settings = dict(config, cluster_counts=[1, 2])
        if option is not None:
            settings['include_all_clusters'] = option
        output = tmp_path/f'study-{option}'
        jobs = study.prepare(settings, output)
        layouts = json.loads((output/'layouts.json').read_text())
        assert layouts[0]['probes'] == {'1': [1], '2': [1]}
        assert layouts[1]['probes'] == {'1': expected, '2': expected}
        for job in jobs:
            assert {variant['probes'] for variant in job['variants']} == set(expected if job['clusters'] == 2 else [1])


def test_binary_router_uses_packed_signs_without_padding_or_changing_float_inputs(config, tmp_path, monkeypatch):
    pool = Path(config['pool'])
    signs = np.array([[1, 0] * 32 + [0], [0, 1] * 32 + [1], [1] * 65, [0] * 65], dtype=np.uint8)
    codes = pack_signs(signs)
    codes[:, 1] |= np.uint64(0xffffffffffffffff ^ 1)  # Padding beyond coordinate 64 must be ignored.
    documents = (2 * signs.astype(np.float32) - 1) * np.linspace(.1, 1, 65, dtype=np.float32)
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    queries = np.array([[1] * 65, [0] * 65], dtype=np.float32)
    for name, values in dict(codes=codes, documents=documents, queries=queries,
                              reference=reference_rows(signs, queries, 2)).items():
        np.save(pool/f'{name}.npy', values)
    hashes = {name: file_hash(pool/name) for name in
              ('codes.npy', 'documents.npy', 'queries.npy', 'reference.npy')}
    (pool/'pool.json').write_text(json.dumps(dict(
        identity=dict(documents=4, dimensions=65, queries=2, top_k=2), hashes=hashes)))
    training = []

    def capture_training(values, **settings):
        training.append((values.copy(), settings))
        return SimpleNamespace(assignments=np.array([0, 1, 0, 1], dtype=np.int64), build_ms=0.0,
            quantizer=SimpleNamespace(reconstruct_n=lambda start, count: values[:count].copy()))

    monkeypatch.setattr(real_study, 'build_router', capture_training)
    monkeypatch.setattr(study, 'build_router', capture_training, raising=False)
    monkeypatch.setattr(study, 'routing_ranks', lambda case:
        (np.ones((2, 2), dtype=np.int64), np.array([[2, 1], [2, 1]], dtype=np.float32)))
    config.update(cluster_counts=[1, 2])
    study.prepare(config, tmp_path/'float-study')
    float_cache = pool/'routers/ivf-2-seed42'
    float_hashes = {name: file_hash(float_cache/name) for name in study.ROUTER_FILES}
    binary_config = dict(config, routing_representation='binary')
    jobs = study.prepare(binary_config, tmp_path/'binary-study')
    assert len(training) == 2
    np.testing.assert_array_equal(training[0][0], documents)
    expected = (2 * signs.astype(np.float32) - 1) * np.float32(1 / np.sqrt(65))
    np.testing.assert_array_equal(training[1][0], expected)
    assert training[1][0].shape == (4, 65) and training[1][0].dtype == np.float32
    assert training[1][1] == dict(method='ivf', clusters=2, seed=42, threads=1, retain_float_index=False)
    binary_cache = pool/'routers/ivf-binary-2-seed42'
    metadata = json.loads((binary_cache/'router.json').read_text())
    assert metadata['kind'] == 'ivf' and metadata['training_representation'] == 'binary'
    assert metadata['source_codes_sha256'] == hashes['codes.npy'] and metadata['dimensions'] == 65
    assert metadata['training_scale'] == float(np.float32(1 / np.sqrt(65)))
    assert {job['router'] for job in jobs if job['clusters'] == 2} == {str(binary_cache)}
    assert {job['method'] for job in jobs if job['clusters'] == 2} == {'scan', 'branch', 'prefix'}
    assert all(job['router'] is None for job in jobs if job['clusters'] == 1)
    study.prepare(binary_config, tmp_path/'binary-reuse')
    assert len(training) == 2
    assert all(file_hash(pool/name) == digest for name, digest in hashes.items())
    assert all(file_hash(float_cache/name) == digest for name, digest in float_hashes.items())
