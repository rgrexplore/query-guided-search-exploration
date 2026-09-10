"""Prepare a disjoint query set for evaluation after tuning choices are frozen.

This command encodes queries only. It does not search the document corpus or
inspect which index settings work well on the new queries.
"""
import argparse
import hashlib
import json
import tarfile
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np

from embedding_model import NomicEncoder, matryoshka_vectors, normalized_prefix, package_versions
from experiments.isolated import ROOT
from msmarco import _read_queries, QUERY_MEMBER, EXPECTED_QUERY_ROWS, EXPECTED_ARCHIVE_SHA256


def select_fresh(ids, texts, excluded, count):
    chosen = [(query_id, text) for query_id, text in zip(ids, texts, strict=True) if query_id not in excluded]
    if len(chosen) < count:
        raise ValueError('not enough unused query IDs for the requested count')
    chosen = chosen[:count]
    return [query_id for query_id, _ in chosen], [text for _, text in chosen]


def prepare_queries(embedding_dir, archive_path, output, count, seed, dimensions, device, extra_manifests=()):
    if output.exists():
        raise FileExistsError('choose a new output directory; evaluation inputs are preserved')
    source = json.loads((embedding_dir/'manifest.json').read_text())
    old_queries = json.loads((embedding_dir/'derived'/str(dimensions)/'manifest.json').read_text())
    excluded = set(old_queries['query_ids'])
    for manifest in extra_manifests:
        excluded.update(json.loads(Path(manifest).read_text())['query_ids'])
    with archive_path.open('rb') as file:
        archive_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('the query archive does not match the recorded official source')
    with tarfile.open(archive_path, 'r:gz') as archive:
        # Reuse the original parser and its row-count/duplicate-ID checks.
        ids, texts, _, _, query_source_hash = _read_queries(
            archive, archive.getmember(QUERY_MEMBER), EXPECTED_QUERY_ROWS, seed)
    ids, texts = select_fresh(ids, texts, excluded, count)
    assert not excluded.intersection(ids)
    output.mkdir(parents=True)
    identity = source['identity']
    started = perf_counter()
    encoder = NomicEncoder(model_name=identity['model'], revision=identity['revision'],
                           batch_size=identity['batch_size'], max_length=identity['max_length'],
                           device=device, cache_dir=ROOT/'data/model')
    load_seconds = perf_counter()-started
    started = perf_counter()
    raw = encoder.encode_prefixed(['search_query: '+text for text in texts])
    # Match the existing cache recipe: full normalization first, then normalize
    # the requested prefix. Search timings remain separate from this batch encode.
    _, full = matryoshka_vectors(raw, 768)
    queries = normalized_prefix(full, dimensions)
    encode_seconds = perf_counter()-started
    np.save(output/'full-queries.npy', full)
    np.save(output/'queries.npy', queries)
    with (output/'queries.jsonl').open('w') as file:
        for query_id, text in zip(ids, texts, strict=True):
            file.write(json.dumps({'query_id':query_id, 'text':text})+'\n')
    manifest = dict(query_ids=ids, count=count, dimensions=dimensions, seed=seed,
                    excluded_query_count=len(excluded), excluded_query_ids=sorted(excluded),
                    archive_sha256=archive_hash, query_source_sha256=query_source_hash,
                    model=identity['model'], revision=identity['revision'], recipe=identity['recipe'],
                    device=device, package_versions=package_versions(),
                    model_load_seconds=load_seconds, batch_encode_seconds=encode_seconds,
                    queries_sha256=hashlib.sha256((output/'queries.npy').read_bytes()).hexdigest(),
                    scope='Disjoint from the original cached query IDs. Prepared without searching or evaluating index configurations. Batch encoding time is not single-query latency.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Encoded {count} queries, with no overlap with the {len(excluded)} previously used query IDs.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ROOT/'data/scaling/collectionandqueries.tar.gz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--exclude-manifest', action='append', type=Path, default=[])
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    embedding_dir = (ROOT/config['data']['embedding_dir']).resolve()
    prepare_queries(embedding_dir, args.archive, args.output, args.count,
                    args.seed, config['data']['dimensions'], args.device, args.exclude_manifest)


if __name__ == '__main__':
    main()
