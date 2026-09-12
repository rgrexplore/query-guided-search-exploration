"""Prepare one fixed query set and new pools without changing old caches.

The three commands follow the data flow: select queries, encode them, then make
normalized prefixes and exact binary-score reference rows for each dimension.
"""

import argparse
import gc
import hashlib
import json
import tarfile
from pathlib import Path
from time import perf_counter

import bitplane_index
import numpy as np
import torch

from embedding_model import (
    NomicEncoder, RECIPE, matryoshka_vectors, normalized_prefix, package_versions,
)
from experiments.components import pack_signs


NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
QUERY_MEMBER = "queries.dev.small.tsv"
QUERY_MEMBER_SHA256 = "f4f71fa87b6be4fb968cca2b8f8554c4ebd003e6c3488af3ac1cc5629c3de189"
POOL_ARRAYS = ("documents.npy", "queries.npy", "codes.npy", "reference.npy")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, value):
    # Replace only after the whole checkpoint has been written.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_selection(archive_path, output, count=1000, seed=42,
                      expected_hash=QUERY_MEMBER_SHA256):
    """Choose IDs before any embedding or search results are available."""
    archive_path, output = Path(archive_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(archive=str(archive_path.resolve()), member=QUERY_MEMBER,
                   member_sha256=expected_hash, count=count, seed=seed)
    manifest_path = output / "selection.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["request"] != request:
            raise ValueError("This folder already has a different query selection.")
        if file_hash(output / "queries.jsonl") != metadata["queries_sha256"]:
            raise ValueError("The saved query selection has changed.")
        return metadata

    # Read only the requested archive member; do not extract the passage corpus.
    content = None
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name.removeprefix("./") == QUERY_MEMBER:
                content = archive.extractfile(member).read()
                break
    if content is None or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("The query member is missing or its SHA-256 differs.")
    rows = []
    for line in content.decode().splitlines():
        identifier, text = line.split("\t", 1)
        rows.append({"_id": identifier, "text": text})
    rows.sort(key=lambda row: int(row["_id"]))
    if not 1 <= count <= len(rows):
        raise ValueError("Query count must fit the source query file.")
    positions = np.random.default_rng(seed).permutation(len(rows))[:count]
    chosen = [rows[int(position)] for position in positions]
    query_path = output / "queries.jsonl"
    query_path.write_text("".join(json.dumps(row) + "\n" for row in chosen))
    metadata = dict(request=request, source_queries=len(rows),
                    query_ids=[row["_id"] for row in chosen],
                    queries_sha256=file_hash(query_path))
    save_json(manifest_path, metadata)
    return metadata


def encode_nomic_queries(folder, *, cache_dir, batch_size=32, chunk_size=128,
                         max_length=512, device="mps"):
    """Encode query chunks with the same recipe as the cached Nomic documents."""
    folder = Path(folder)
    selection = json.loads((folder / "selection.json").read_text())
    rows = [json.loads(line) for line in (folder / "queries.jsonl").read_text().splitlines()]
    if file_hash(folder / "queries.jsonl") != selection["queries_sha256"]:
        raise ValueError("The selected query text has changed.")
    identity = dict(model=NOMIC_MODEL, revision=NOMIC_REVISION, recipe=RECIPE,
                    query_ids=selection["query_ids"], text_sha256=selection["queries_sha256"],
                    device=device, batch_size=batch_size, max_length=max_length,
                    package_versions=package_versions())
    manifest_path = folder / "embedding.json"
    values_path = folder / "full-queries.npy"
    progress_path = folder / "encoding-progress.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["identity"] != identity or file_hash(values_path) != metadata["sha256"]:
            raise ValueError("The query embedding cache belongs to different inputs.")
        return values_path
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("The encoding checkpoint belongs to different inputs.")
        values = np.load(values_path, mmap_mode="r+")
    else:
        values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32,
                                          shape=(len(rows), 768))
        progress = dict(identity=identity, completed_queries=0, chunks=[])
        save_json(progress_path, progress)

    encoder = NomicEncoder(model_name=NOMIC_MODEL, revision=NOMIC_REVISION,
                           batch_size=batch_size, max_length=max_length,
                           device=device, cache_dir=Path(cache_dir))
    for start in range(progress["completed_queries"], len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        texts = ["search_query: " + row["text"] for row in rows[start:stop]]
        if device == "mps":
            torch.mps.synchronize()
        started = perf_counter()
        raw = encoder.encode_prefixed(texts)
        # These are raw model outputs. Apply Nomic's layer normalization once.
        _, full = matryoshka_vectors(raw, 768)
        if device == "mps":
            torch.mps.synchronize()
        seconds = perf_counter() - started
        values[start:stop] = full
        values.flush()
        progress["completed_queries"] = stop
        progress["chunks"].append(dict(start=start, stop=stop, seconds=seconds))
        save_json(progress_path, progress)
        print(f"Encoded {stop}/{len(rows)} queries in {seconds:.2f}s for this chunk", flush=True)
    del values, encoder
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    metadata = dict(identity=identity, shape=[len(rows), 768], dtype="float32",
                    sha256=file_hash(values_path), chunks=progress["chunks"])
    save_json(manifest_path, metadata)
    return values_path


def make_pool_arrays(documents_path, queries_path, pool, dimensions, chunk_size):
    """Cached full vectors are already normalized: crop and normalize, nothing else."""
    full_documents = np.load(documents_path, mmap_mode="r")
    full_queries = np.load(queries_path, mmap_mode="r")
    documents = np.lib.format.open_memmap(pool / "documents.npy", mode="w+", dtype=np.float32,
                                         shape=(len(full_documents), dimensions))
    codes = np.lib.format.open_memmap(pool / "codes.npy", mode="w+", dtype=np.uint64,
                                     shape=(len(full_documents), (dimensions + 63) // 64))
    for start in range(0, len(full_documents), chunk_size):
        stop = min(start + chunk_size, len(full_documents))
        vectors = normalized_prefix(full_documents[start:stop], dimensions)
        documents[start:stop] = vectors
        codes[start:stop] = pack_signs(vectors >= 0)
    documents.flush()
    codes.flush()
    np.save(pool / "queries.npy", normalized_prefix(full_queries, dimensions))
    del documents, codes
    save_json(pool / "arrays.json", {name: file_hash(pool / name)
                                      for name in POOL_ARRAYS if name != "reference.npy"})


def prepare_reference(pool, dimensions, top_k, chunk_size, limit=None):
    """Scan every document. Save each finished query chunk so work can resume."""
    progress_path = pool / "reference-progress.json"
    queries = np.load(pool / "queries.npy")
    query_count = len(queries)
    stop_at = query_count if limit is None else min(limit, query_count)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        reference = np.load(pool / "reference.npy", mmap_mode="r+")
    else:
        progress = dict(completed_queries=0, chunks=[], index_build_seconds=[])
        reference = np.lib.format.open_memmap(pool / "reference.npy", mode="w+", dtype=np.int64,
                                             shape=(query_count, top_k))
        reference[:] = -1
        reference.flush()
        save_json(progress_path, progress)
    if progress["completed_queries"] >= stop_at:
        return progress

    codes = np.load(pool / "codes.npy", mmap_mode="r")
    started = perf_counter()
    index = bitplane_index.Index(codes, np.zeros(len(codes), dtype=np.int64), dimensions,
                                 build_bitplanes=False)
    progress["index_build_seconds"].append(perf_counter() - started)
    # One cluster means this reference includes every document, with no routing loss.
    for start in range(progress["completed_queries"], stop_at, chunk_size):
        stop = min(start + chunk_size, stop_at)
        selected = np.zeros((stop - start, 1), dtype=np.int64)
        started = perf_counter()
        result = index.scan(queries[start:stop], selected, candidate_limit=top_k)
        seconds = perf_counter() - started
        reference[start:stop] = result["rows"]
        reference.flush()
        progress["completed_queries"] = stop
        progress["chunks"].append(dict(start=start, stop=stop, seconds=seconds))
        save_json(progress_path, progress)
        print(f"Reference {stop}/{query_count} queries: {seconds:.2f}s for this chunk", flush=True)
    return progress


def prepare_pool(documents_path, queries_path, query_ids, pool, *, dimensions,
                 top_k=100, provenance, chunk_size=4096, reference_chunk_size=25,
                 reference_limit=None):
    """Make a new dataset/dimension pool. Completed pools are checked, never rewritten."""
    documents_path, queries_path, pool = Path(documents_path), Path(queries_path), Path(pool)
    documents = np.load(documents_path, mmap_mode="r")
    queries = np.load(queries_path, mmap_mode="r")
    if not 1 <= dimensions <= min(documents.shape[1], queries.shape[1]):
        raise ValueError("Dimensions must fit both full embedding arrays.")
    if len(query_ids) != len(queries) or not 1 <= top_k <= len(documents):
        raise ValueError("Query IDs and top_k must fit the supplied arrays.")
    identity = dict(documents=len(documents), queries=len(queries), dimensions=dimensions,
                    top_k=top_k, full_documents_sha256=file_hash(documents_path),
                    full_queries_sha256=file_hash(queries_path), query_ids=list(query_ids),
                    provenance=provenance)
    del documents, queries
    pool.mkdir(parents=True, exist_ok=True)
    request_path = pool / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != identity:
        raise ValueError("This pool belongs to different inputs; choose a new folder.")
    save_json(request_path, identity)
    manifest_path = pool / "pool.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["identity"] != identity or any(
            file_hash(pool / name) != digest for name, digest in metadata["hashes"].items()
        ):
            raise ValueError("The completed pool's inputs or arrays have changed.")
        return metadata
    arrays_path = pool / "arrays.json"
    if not arrays_path.exists():
        make_pool_arrays(documents_path, queries_path, pool, dimensions, chunk_size)
    else:
        for name, digest in json.loads(arrays_path.read_text()).items():
            if file_hash(pool / name) != digest:
                raise ValueError("A prepared input array has changed.")
    progress = prepare_reference(pool, dimensions, top_k, reference_chunk_size, reference_limit)
    if progress["completed_queries"] != identity["queries"]:
        return progress
    metadata = dict(identity=identity, query_ids=list(query_ids),
                    hashes={name: file_hash(pool / name) for name in POOL_ARRAYS},
                    reference_method="native exhaustive binary-score scan; ties by row ID",
                    native_sha256=file_hash(bitplane_index.__file__))
    save_json(manifest_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select")
    selection.add_argument("archive", type=Path)
    selection.add_argument("output", type=Path)
    selection.add_argument("--queries", type=int, default=1000)
    selection.add_argument("--seed", type=int, default=42)
    encoding = commands.add_parser("encode")
    encoding.add_argument("folder", type=Path)
    encoding.add_argument("--cache-dir", type=Path, required=True)
    encoding.add_argument("--batch-size", type=int, default=32)
    encoding.add_argument("--chunk-size", type=int, default=128)
    encoding.add_argument("--device", default="mps")
    pool = commands.add_parser("pool")
    pool.add_argument("documents", type=Path)
    pool.add_argument("queries", type=Path)
    pool.add_argument("query_ids", type=Path, help="JSON list, or JSON object containing query_ids")
    pool.add_argument("output", type=Path)
    pool.add_argument("--dimensions", type=int, required=True)
    pool.add_argument("--top-k", type=int, default=100)
    pool.add_argument("--provenance", type=Path, required=True)
    pool.add_argument("--reference-limit", type=int)
    pool.add_argument("--reference-chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.command == "select":
        prepare_selection(args.archive, args.output, args.queries, args.seed)
    elif args.command == "encode":
        encode_nomic_queries(args.folder, cache_dir=args.cache_dir, batch_size=args.batch_size,
                             chunk_size=args.chunk_size, device=args.device)
    else:
        query_ids = json.loads(args.query_ids.read_text())
        if isinstance(query_ids, dict):
            query_ids = query_ids["query_ids"]
        prepare_pool(args.documents, args.queries, query_ids, args.output,
                     dimensions=args.dimensions, top_k=args.top_k,
                     provenance=json.loads(args.provenance.read_text()),
                     reference_limit=args.reference_limit,
                     reference_chunk_size=args.reference_chunk_size)


if __name__ == "__main__":
    main()
