"""Prepare small pools and run a declared list of configurations sequentially."""
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments.components import pack_signs, reference_rows
from experiments.worker import power_state

ROOT = Path(__file__).resolve().parents[1]


def prepare_pool(config, documents):
    data = config['data']
    pool = (ROOT/data['cache_dir']/f'n{documents}').resolve()
    identity = dict(kind=data['kind'], documents=documents, dimensions=data['dimensions'],
                    queries=data['queries'], seed=data['seed'], top_k=config['search']['top_k'])
    if pool.exists():
        saved = json.loads((pool/'pool.json').read_text())
        if saved['identity'] != identity:
            raise ValueError(f'{pool} belongs to a different setup; choose a new cache_dir')
        return pool
    pool.mkdir(parents=True)
    if data['kind'] != 'random_signs':
        raise ValueError('this calibration stage uses random_signs')
    rng = np.random.default_rng(data['seed'])
    signs = rng.integers(0, 2, (documents, data['dimensions']), dtype=np.uint8)
    # Reuse the same queries and nested document prefixes at every size.
    queries = np.random.default_rng(data['seed']+1).normal(size=(data['queries'], data['dimensions'])).astype(np.float32)
    np.save(pool/'codes.npy', pack_signs(signs))
    np.save(pool/'queries.npy', queries)
    with threadpool_limits(limits=1):
        reference = reference_rows(signs, queries, config['search']['top_k'])
    np.save(pool/'reference.npy', reference)
    hashes = {name: hashlib.sha256((pool/name).read_bytes()).hexdigest()
              for name in ['codes.npy', 'queries.npy', 'reference.npy']}
    (pool/'pool.json').write_text(json.dumps(dict(identity=identity, hashes=hashes), indent=2)+'\n')
    return pool


def calibration_cases(config, pools):
    cases = []
    for documents, pool in pools.items():
        common = dict(pool=str(pool), documents=documents, dimensions=config['data']['dimensions'],
                      query_rows=list(range(config['data']['queries'])), router=None, probes=1,
                      top_k=config['search']['top_k'], repetitions=config['measurement']['repetitions'],
                      ram_budget_bytes=config['limits']['ram_budget_bytes'])
        cases.append(dict(common, method='scan'))
        for budget, leaf in itertools.product(config['sweep']['node_budgets'], config['sweep']['leaf_sizes']):
            cases.append(dict(common, method='branch', node_budget=budget, leaf_size=leaf))
        for bits, target in itertools.product(config['sweep']['key_bits'], config['sweep']['candidate_targets']):
            cases.append(dict(common, method='keys', key_bits=bits, key_offset=0,
                              candidate_target=target, key_limit=config['sweep']['key_limit']))
    order = np.random.default_rng(config['measurement']['schedule_seed']).permutation(len(cases))
    return [cases[int(i)] for i in order]


def run_worker(case_path, output, limits):
    """One bounded subprocess, with failures preserved rather than silently retried."""
    output.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    observed_peak = 0
    failure = None
    with (output.parent/'worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'experiments.worker', str(case_path), str(output)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        child = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                observed_peak = max(observed_peak, child.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if observed_peak > limits['worker_stop_bytes']:
                failure = 'memory_stop'
            elif monotonic()-started > limits['case_seconds']:
                failure = 'timeout'
            if failure:
                process.kill()
                break
            sleep(.02)
        code = process.wait()
    status = failure or ('complete' if code==0 else 'failed')
    if status == 'complete' and not (output/'result.json').exists():
        status = 'missing_result'
    observation = dict(status=status, returncode=code, parent_sampled_peak=observed_peak,
                       seconds=monotonic()-started)
    (output.parent/'process.json').write_text(json.dumps(observation, indent=2)+'\n')
    return observation


def run_isolated(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    pools = {n: prepare_pool(config, n) for n in config['data']['sizes']}
    cases = calibration_cases(config, pools)
    execute_cases(config, output, cases, 'Isolated single-thread calibration on random signs; check sizes excluded from fitting.')


def execute_cases(config, output, cases, scope):
    (output/'schedule.json').write_text(json.dumps(cases, indent=2)+'\n')
    hardware = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'],
                              capture_output=True, text=True) if sys.platform=='darwin' else None
    environment = dict(platform=platform.platform(), python=sys.version, power_before=power_state(),
                       hardware=hardware.stdout.strip() if hardware else platform.machine(),
                       scope=scope)
    (output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    for number, case in enumerate(cases):
        folder = output/'cases'/f'{number:04d}'
        folder.mkdir(parents=True)
        case_path = folder/'case.json'
        case_path.write_text(json.dumps(case, indent=2)+'\n')
        observed = run_worker(case_path.resolve(), (folder/'run').resolve(), config['limits'])
        print(f"{number+1}/{len(cases)} {case['method']} N={case['documents']} {observed['status']}", flush=True)
    environment['power_after'] = power_state()
    (output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
