"""Tune and test the declared query-weight examples using the existing worker."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.choices import combine
from experiments.controlled import prepare_controlled_pool
from experiments.direct_routing import DirectPrefixRouter
from experiments.evaluation_report import report
from experiments.isolated import execute_cases
from experiments.probe_cutoffs import routing_ranks, probe_cutoff, cover_boundary_ties
from experiments.real_study import prepare_router


def prepare_layout(pool, kind, clusters, seed):
    if kind != 'direct':
        return prepare_router(pool, kind, clusters, seed)
    # Use the same assignments as sign-prefix routing. Only the query-time
    # routing algorithm changes; it has its own measured memory and time.
    sign_folder = prepare_router(pool, 'sign', clusters, seed)
    folder = sign_folder.parent / f'direct-{clusters}-seed{seed}'
    if not folder.exists():
        folder.mkdir()
        for name in ('assignments.npy', 'labels.npy'):
            (folder / name).symlink_to((sign_folder / name).resolve())
        metadata = json.loads((sign_folder / 'router.json').read_text())
        metadata['kind'] = 'direct'
        (folder / 'router.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return folder


def layout_probes(base, targets):
    """Choose probes from actual tuning-neighbor cluster ranks, not guessed recall."""
    if base['router_kind'] != 'direct':
        ranks, scores, *_ = routing_ranks(base)
        counts = {probe_cutoff(ranks, target) for target in targets}
        if scores is not None:
            counts = {cover_boundary_ties(scores, p) for p in counts}
        return sorted(counts)

    folder = Path(base['router'])
    pool = Path(base['pool'])
    labels = np.load(folder / 'labels.npy')
    metadata = json.loads((folder / 'router.json').read_text())
    router = DirectPrefixRouter(labels, metadata['routing_bits'])
    queries = np.load(pool / 'queries.npy')[base['query_rows']]
    truth = np.load(pool / 'reference.npy')[base['query_rows']]
    assignments = np.load(folder / 'assignments.npy', mmap_mode='r')
    ranks = np.empty_like(truth)
    for qi, query in enumerate(queries):
        ordered = router.select(query[None, :], len(labels))[0]
        inverse = np.zeros(int(labels.max()) + 1, dtype=np.int64)
        inverse[ordered] = np.arange(1, len(labels) + 1)
        ranks[qi] = inverse[assignments[truth[qi]]]
    counts = sorted({probe_cutoff(ranks, target) for target in targets})
    # Partial searches must preserve the ordering used to derive the cutoff.
    # This also checks the native tie rule when many prefix scores are equal.
    for probes in counts:
        for qi, query in enumerate(queries):
            selected = router.select(query[None, :], probes)[0]
            np.testing.assert_array_equal(
                np.isin(assignments[truth[qi]], selected), ranks[qi] <= probes)
    return counts


def controlled_cases(config, pool, documents, layouts):
    cases = []
    for layout in layouts:
        for probes in layout['probes']:
            common = dict(
                pool=str(pool), documents=documents, dimensions=config['data']['dimensions'],
                query_rows=list(range(config['data']['tuning_queries'])),
                router=layout['path'], router_kind=layout['kind'], clusters=layout['clusters'],
                probes=probes, top_k=config['search']['top_k'],
                repetitions=config['measurement']['repetitions'],
                ram_budget_bytes=config['limits']['ram_budget_bytes'])
            cases.append(dict(common, method='scan'))
            for leaf in config['sweep']['leaf_sizes']:
                cases.append(dict(common, method='branch', node_budget=0, leaf_size=leaf))
            for bits in config['sweep']['key_bits']:
                cases.append(dict(common, method='keys', key_bits=bits,
                                  key_offset=layout['routing_bits'], candidate_target=0,
                                  key_limit=config['sweep']['key_limit']))
    return cases


def evaluation_cases(config, shortlist):
    cases = []
    held_out = list(range(config['data']['tuning_queries'], config['data']['queries']))
    for choice in shortlist:
        for block in range(config['measurement']['blocks']):
            for seed in choice['evaluation_seeds']:
                cases.append(dict(
                    choice['case'], query_rows=held_out, block=block, seed=seed,
                    repetitions=config['measurement']['evaluation_repetitions'],
                    setting_id=choice['setting_id'], selection_uses=choice['uses']))
    return cases


def run_controlled(config, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    pools, cases, layouts_by_size = {}, [], {}
    for documents in config['data']['sizes']:
        with threadpool_limits(limits=1):
            pool = prepare_controlled_pool(config, documents)
            pools[documents] = pool
            layouts = [dict(path=None, kind='none', clusters=1, routing_bits=0, probes=[1])]
            for kind, clusters in itertools.product(config['sweep']['routers'], config['sweep']['clusters']):
                folder = prepare_layout(pool, kind, clusters, config['data']['seed'])
                metadata = json.loads((folder / 'router.json').read_text())
                base = dict(pool=str(pool), router=str(folder), router_kind=kind,
                            query_rows=list(range(config['data']['tuning_queries'])))
                # Include an exact-routing target as a control for each layout.
                probes = layout_probes(base, config['search']['recall_targets'] + [1.0])
                layouts.append(dict(path=str(folder), kind=kind, clusters=clusters,
                                    routing_bits=metadata['routing_bits'], probes=probes))
                print(f"N={documents} {kind} C={clusters}: probes={probes}", flush=True)
        layouts_by_size[documents] = layouts
        cases.extend(controlled_cases(config, pool, documents, layouts))
    (output / 'layouts.json').write_text(json.dumps(layouts_by_size, indent=2) + '\n')
    tuning = output / 'tuning'
    tuning.mkdir()
    (tuning / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    rng = np.random.default_rng(config['measurement']['schedule_seed'])
    cases = [cases[int(i)] for i in rng.permutation(len(cases))]
    execute_cases(config, tuning, cases, 'Constructed query weights; first query subset only, used for tuning.')

    # Freeze before running any query in the independent subset.
    _, shortlist = combine([tuning], output / 'frozen', config['search']['recall_targets'])
    evaluation = output / 'evaluation'
    evaluation.mkdir()
    (evaluation / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    shortlist_path = output / 'frozen' / 'shortlist.json'
    (evaluation / 'frozen-shortlist.json').write_bytes(shortlist_path.read_bytes())
    first_pool = next(iter(pools.values()))
    metadata = json.loads((first_pool / 'pool.json').read_text())
    source = dict(
        shortlist_sha256=hashlib.sha256(shortlist_path.read_bytes()).hexdigest(),
        query_ids=metadata['query_ids'][config['data']['tuning_queries']:],
        scope='Disjoint generated queries; same documents. Parameters frozen using the first subset only.')
    (evaluation / 'evaluation-source.json').write_text(json.dumps(source, indent=2) + '\n')
    cases = evaluation_cases(config, shortlist)
    cases = [cases[int(i)] for i in rng.permutation(len(cases))]
    execute_cases(config, evaluation, cases, 'Frozen controlled-data choices on the independent query subset.')
    report(evaluation)
