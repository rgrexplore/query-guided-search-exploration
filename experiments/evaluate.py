"""Replay frozen settings on disjoint query IDs without retuning parameters."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isolated import ROOT, execute_cases
from experiments.real_study import prepare_real_pool


def run_evaluation(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    choices_path = ROOT/config['experiment']['shortlist']
    choices = json.loads(choices_path.read_text())
    embedding = ROOT/config['data']['embedding_dir']
    original_ids = set(json.loads((embedding/'derived'/str(config['data']['dimensions'])/'manifest.json').read_text())['query_ids'])
    fresh_manifest = json.loads((ROOT/config['data']['query_dir']/'manifest.json').read_text())
    if original_ids.intersection(fresh_manifest['query_ids']):
        raise ValueError('evaluation queries overlap the original tuning-query collection')
    pools = {n:prepare_real_pool(config,n) for n in sorted({c['case']['documents'] for c in choices})}
    cases = []
    for choice in choices:
        original = choice['case']
        pool = pools[original['documents']]
        old_meta = json.loads((Path(original['pool'])/'pool.json').read_text())
        new_meta = json.loads((pool/'pool.json').read_text())
        if old_meta['hashes']['codes.npy'] != new_meta['hashes']['codes.npy']:
            raise ValueError('the frozen setting and evaluation use different document codes')
        for seed in choice['evaluation_seeds']:
            cases.append(dict(original, pool=str(pool), query_rows=list(range(config['data']['queries'])),
                              repetitions=config['measurement']['repetitions'], seed=seed,
                              setting_id=choice['setting_id'], selection_uses=choice['uses']))
    order = np.random.default_rng(config['measurement']['schedule_seed']).permutation(len(cases))
    cases = [cases[int(i)] for i in order]
    (output/'frozen-shortlist.json').write_text(json.dumps(choices,indent=2)+'\n')
    (output/'evaluation-source.json').write_text(json.dumps(dict(
        shortlist_sha256=hashlib.sha256(choices_path.read_bytes()).hexdigest(),
        query_ids=fresh_manifest['query_ids'], queries_sha256=fresh_manifest['queries_sha256'],
        excluded_original_query_count=len(original_ids),
        scope='Parameters copied unchanged from the frozen shortlist; only query inputs, repeats and predeclared seeds change.'),indent=2)+'\n')
    execute_cases(config,output,cases,'Independent evaluation of a frozen shortlist; no parameter selection in this stage.')
