"""Check native scaling on packed random signs without expanding them to float matrices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.components import pack_signs
from experiments.isolated import ROOT, execute_cases

BYTE_COUNTS = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def make_queries(dimensions, count, strong_bits, support, seed):
    random = np.random.default_rng(seed)
    signs = random.integers(0,2,(count,dimensions),dtype=np.uint8)
    weights = np.full(signs.shape,.0001,dtype=np.float32)
    supports = []
    for row in range(count):
        positions = np.arange(strong_bits) if support=='fixed' else np.sort(random.choice(dimensions,strong_bits,replace=False))
        weights[row,positions] = 1
        supports.append(positions)
    queries = weights*(2*signs.astype(np.float32)-1)
    queries /= np.linalg.norm(queries,axis=1,keepdims=True)
    return np.ascontiguousarray(queries), supports


def binary_reference(codes, query, support, dimensions, top_k, chunk_size=65536):
    ideal = pack_signs((query>=0)[None,:])[0]
    strong_mask = np.zeros(codes.shape[1],dtype=np.uint64)
    for position in support:
        strong_mask[position//64] |= np.uint64(1) << np.uint64(position%64)
    weak_position = next(i for i in range(dimensions) if i not in support)
    strong_weight = float(abs(query[support[0]]))
    weak_weight = float(abs(query[weak_position]))
    scores = np.empty(len(codes),dtype=np.float64)
    all_strong_matches = 0
    for start in range(0,len(codes),chunk_size):
        difference = np.bitwise_xor(codes[start:start+chunk_size],ideal)
        errors = BYTE_COUNTS[difference.view(np.uint8)].sum(axis=1,dtype=np.int16)
        strong_errors = BYTE_COUNTS[(difference & strong_mask).view(np.uint8)].sum(axis=1,dtype=np.int16)
        weak_errors = errors-strong_errors
        scores[start:start+len(difference)] = ((len(support)-2*strong_errors)*strong_weight
            +(dimensions-len(support)-2*weak_errors)*weak_weight)
        all_strong_matches += int(np.sum(strong_errors==0))
    threshold = np.partition(scores,len(scores)-top_k)[len(scores)-top_k]
    candidates = np.flatnonzero(scores>=threshold)
    chosen = candidates[np.lexsort((candidates,-scores[candidates]))[:top_k]]
    return chosen, all_strong_matches


def prepare_pool(folder, documents, dimensions, queries, supports, support):
    if folder.exists():
        return folder
    folder.mkdir(parents=True)
    codes = np.lib.format.open_memmap(folder/'codes.npy',mode='w+',dtype=np.uint64,
                                     shape=(documents,(dimensions+63)//64))
    random = np.random.default_rng(73)
    for start in range(0,documents,65536):
        count = min(65536,documents-start)
        signs = random.integers(0,2,(count,dimensions),dtype=np.uint8)
        codes[start:start+count] = pack_signs(signs)
    codes.flush()
    # Check the shared prefix against the previously measured document pool.
    original = np.load(ROOT/'data/controlled-fixed-2026-09-10/n1000000/codes.npy',mmap_mode='r')
    np.testing.assert_array_equal(codes[:len(original)],original)
    reference, counts = [], []
    for query, positions in zip(queries,supports,strict=True):
        rows, count = binary_reference(codes,query,positions,dimensions,100)
        reference.append(rows); counts.append(count)
    np.save(folder/'queries.npy',queries)
    np.save(folder/'reference.npy',np.array(reference))
    hashes = {}
    for name in ('codes.npy','queries.npy','reference.npy'):
        with (folder/name).open('rb') as file: hashes[name]=hashlib.file_digest(file,'sha256').hexdigest()
    (folder/'pool.json').write_text(json.dumps(dict(documents=documents,dimensions=dimensions,
        query_ids=[f'{support}-scale-q{i}' for i in range(len(queries))],strong_match_counts=counts,
        support=support,hashes=hashes,scope='Packed random signs; document prefix matches the earlier controlled corpus. New query seed197. Independent count-based full-score reference.'),indent=2)+'\n')
    return folder


def run(output, sizes):
    output=output.resolve(); output.mkdir(parents=True,exist_ok=False)
    config=dict(sizes=sizes,dimensions=256,queries=8,query_seed=197,top_k=100,strong_bits=13,
                repetitions=2,limits=dict(worker_stop_bytes=34359738368,case_seconds=180))
    (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    cases=[]
    for support in ('fixed','adaptive'):
        queries, positions = make_queries(256,8,13,support,197)
        for n in sizes:
            pool = prepare_pool(ROOT/f'data/native-scale-check-2026-09-11/{support}/n{n}',n,256,queries,positions,support)
            common=dict(pool=str(pool),documents=n,dimensions=256,query_rows=list(range(8)),
                        router=None,router_kind='none',clusters=1,probes=1,top_k=100,repetitions=2,
                        ram_budget_bytes=32000000000,support=support)
            cases.append(dict(common,method='scan'))
            cases.append(dict(common,method='branch',node_budget=0,leaf_size=math_leaf(n)))
            cases.append(dict(common,method='keys',key_bits=13,key_offset=0,candidate_target=0,key_limit=0))
    order=np.random.default_rng(97).permutation(len(cases))
    execute_cases(config,output,[cases[int(i)] for i in order],
                  'Prescribed one-cluster native scaling check. All queries retained; this does not retune routing or establish globally optimal4M settings.')


def math_leaf(documents):
    # At1M, mean occupancy122 plus3 standard deviations is about156. Round to160.
    # Scaling L with N keeps the intended13-bit stopping depth as N grows.
    return round(160*documents/1000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000000,2000000,4000000])
    args=parser.parse_args()
    run(args.output,args.sizes)
