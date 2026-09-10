"""Construct the declared fixed/adaptive query-weight cases, without filtering queries."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.components import pack_signs, reference_rows
from experiments.isolated import ROOT


def prepare_controlled_pool(config, documents):
    data=config['data']
    if data['support'] not in ('fixed','adaptive'):
        raise ValueError('support must be fixed or adaptive')
    identity=dict(documents=documents,dimensions=data['dimensions'],queries=data['queries'],
                  seed=data['seed'],support=data['support'],strong_bits=data['strong_bits'],
                  weak_weight=data['weak_weight'],top_k=config['search']['top_k'])
    pool=(ROOT/data['cache_dir']/f'n{documents}').resolve()
    if pool.exists():
        if json.loads((pool/'pool.json').read_text())['identity']!=identity:
            raise ValueError('controlled pool has different assumptions; use a new cache_dir')
        return pool
    pool.mkdir(parents=True)
    dimensions=data['dimensions']
    signs=np.random.default_rng(data['seed']).integers(0,2,(documents,dimensions),dtype=np.uint8)
    query_signs=np.random.default_rng(data['seed']+1).integers(0,2,(data['queries'],dimensions),dtype=np.uint8)
    support_rng=np.random.default_rng(data['seed']+2)
    magnitudes=np.full(query_signs.shape,data['weak_weight'],dtype=np.float32)
    supports=[]
    preferred_counts=[]
    for qi in range(data['queries']):
        support=(np.arange(data['strong_bits']) if data['support']=='fixed'
                 else np.sort(support_rng.choice(dimensions,data['strong_bits'],replace=False)))
        magnitudes[qi,support]=1
        supports.append(support.tolist())
        active=np.ones(documents,dtype=bool)
        counts=[documents]
        for bit in support:
            active &= signs[:,bit]==query_signs[qi,bit]
            counts.append(int(active.sum()))
        preferred_counts.append(counts)
    queries=magnitudes*(2*query_signs.astype(np.float32)-1)
    queries=np.ascontiguousarray(queries/np.linalg.norm(queries,axis=1,keepdims=True))
    vectors=np.ascontiguousarray((2*signs.astype(np.float32)-1)/np.float32(np.sqrt(dimensions)),dtype=np.float32)
    with threadpool_limits(limits=1):
        reference=reference_rows(signs,queries,config['search']['top_k'])
    for name,array in [('codes',pack_signs(signs)),('queries',queries),('documents',vectors),
                       ('reference',reference),('preferred_counts',np.array(preferred_counts,dtype=np.int64))]:
        np.save(pool/f'{name}.npy',array)
    hashes={}
    for name in ['codes.npy','queries.npy','documents.npy','reference.npy','preferred_counts.npy']:
        with (pool/name).open('rb') as file:
            hashes[name]=hashlib.file_digest(file,'sha256').hexdigest()
    metadata=dict(identity=identity,hashes=hashes,supports=supports,
                  query_ids=[f"{data['support']}-q{i}" for i in range(data['queries'])],
                  strong_match_counts=[row[-1] for row in preferred_counts],
                  expected_strong_matches=documents/2**data['strong_bits'],
                  scope='Constructed query-weight distribution, not Nomic embeddings. No insufficient-match query was removed.')
    (pool/'pool.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return pool


def predict_one_path(counts, leaf_size, top_k, dimensions, word_bits=64):
    """Exact work prediction when the declared dominance and enough-match conditions hold.

    The caller must establish sufficient node budget and that strong weights dominate the total weak tail.
    This function rejects cases that need additional strong-error branches.
    """
    if counts[-1]<top_k:
        return None
    depth=next((j for j,n in enumerate(counts) if n<=leaf_size),None)
    if depth is None:
        return None
    words=(counts[0]+word_bits-1)//word_bits
    return dict(nodes=depth+1,bitplane_words=depth*words,leaf_words=words,
                documents_scored=counts[depth],score_terms=counts[depth]*((dimensions+3)//4))
