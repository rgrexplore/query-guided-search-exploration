"""Encode the unchanged BEIR Quora corpus and fixed test queries with pinned Qwen."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
from time import perf_counter

import numpy as np


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/prefix-study-2026-09-13"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_sources(source, selected_queries):
    corpus_ids, document_texts = [], []
    with (source / "corpus.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("title"):
                raise ValueError("Expected Quora questions without titles")
            corpus_ids.append(str(row["_id"]))
            document_texts.append(row["text"])
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("Duplicate corpus IDs")
    official = [json.loads(line) for line in (source / "queries.jsonl").read_text().splitlines()]
    official_by_id = {str(row["_id"]): row["text"] for row in official}
    if len(official_by_id) != len(official):
        raise ValueError("Duplicate official query IDs")
    chosen = [json.loads(line) for line in selected_queries.read_text().splitlines()]
    query_ids = [row["id"] for row in chosen]
    query_texts = [row["text"] for row in chosen]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate selected query IDs")
    if any(official_by_id.get(row["id"]) != row["text"] for row in chosen):
        raise ValueError("Selected queries differ from the official records")
    with (source / "qrels/test.tsv").open() as stream:
        qrels = list(csv.DictReader(stream, delimiter="\t"))
    test_ids = {row["query-id"] for row in qrels}
    corpus_id_set = set(corpus_ids)
    if not set(query_ids) <= test_ids or any(row["corpus-id"] not in corpus_id_set for row in qrels):
        raise ValueError("Query selection or qrels have missing official IDs")
    # Match exact text only; do not remove or rewrite any official row.
    text_counts = {}
    for text in document_texts:
        text_counts[text] = text_counts.get(text, 0) + 1
    overlap = {
        "same_id_query_corpus_count": len(set(query_ids) & corpus_id_set),
        "exact_text_query_corpus_pairs": sum(text_counts.get(text, 0) for text in query_texts),
        "selected_queries_with_exact_corpus_text": sum(text in text_counts for text in query_texts),
        "corpus_duplicate_text_extra_rows": len(document_texts) - len(text_counts),
        "selected_duplicate_query_text_extra_rows": len(query_texts) - len(set(query_texts)),
        "policy": "Preserve every official corpus row and fixed selected query; no matching-text or ID filter.",
    }
    return dict(corpus_ids=corpus_ids, document_texts=document_texts, query_ids=query_ids,
                query_texts=query_texts, qrels=qrels, overlap=overlap)


def encode_chunks(texts, output, kind, identity, encode, *, chunk_size=8192):
    """Save complete normalized chunks atomically, then assemble the final array."""
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    final = output / f"full-{kind}.npy"
    progress_path = output / f"{kind}-progress.json"
    content_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    requested = dict(recipe=identity, text_sha256=content_hash, rows=len(texts), chunk_size=chunk_size)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != requested:
            raise ValueError("Existing chunks belong to different inputs or recipe")
    else:
        if final.exists() or list(chunks.glob(f"{kind}-*.npy")):
            raise ValueError("Existing arrays have no matching input identity")
        progress = dict(identity=requested, state="encoding", completed_rows=0, chunks=[])
        write_json(progress_path, progress)
    if progress["state"] == "complete":
        if file_hash(final) != progress["array_sha256"]:
            raise ValueError("Completed array hash differs from its checkpoint")
        return final, progress
    dimensions = identity["dimensions"]
    paths = []
    previous = {record["start"]: record for record in progress["chunks"]}
    records = []
    for start in range(0, len(texts), chunk_size):
        stop = min(len(texts), start + chunk_size)
        path = chunks / f"{kind}-{start:09d}-{stop:09d}.npy"
        paths.append(path)
        if path.exists():
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (stop - start, dimensions) or values.dtype != np.float32:
                raise ValueError("Saved chunk shape or dtype differs from its checkpoint")
            observed_hash = file_hash(path)
            if start in previous and previous[start]["sha256"] != observed_hash:
                raise ValueError("Saved chunk hash differs from its checkpoint")
            record = previous.get(start, dict(start=start, stop=stop, seconds=None, sha256=observed_hash))
        else:
            before = perf_counter()
            values = np.asarray(encode(texts[start:stop]), dtype=np.float32)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if (values.shape != (stop - start, dimensions) or not np.isfinite(values).all()
                    or not np.isfinite(norms).all() or np.any(norms == 0)):
                raise ValueError("Encoder must return nonzero finite vectors of the expected shape")
            values = np.ascontiguousarray(values / norms)
            temporary = path.with_suffix(".npy.part")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(path)
            record = dict(start=start, stop=stop, seconds=perf_counter() - before, sha256=file_hash(path))
        records.append(record)
        progress.update(completed_rows=stop, chunks=records,
                        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(progress_path, progress)
        print(json.dumps(dict(kind=kind, completed_rows=stop, total_rows=len(texts),
                              chunk_seconds=record["seconds"])), flush=True)
    temporary = final.with_suffix(".npy.part")
    destination = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                           shape=(len(texts), dimensions))
    for path, record in zip(paths, records, strict=True):
        destination[record["start"]:record["stop"]] = np.load(path, mmap_mode="r", allow_pickle=False)
    destination.flush()
    del destination
    temporary.replace(final)
    progress.update(state="complete", array_sha256=file_hash(final), array_path=str(final.resolve()))
    write_json(progress_path, progress)
    return final, progress


class QwenEncoder:
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if not torch.backends.mps.is_available():
            raise RuntimeError("This fixed recipe requires the measured MPS device")
        torch.set_num_threads(4)
        snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        self.torch = torch
        self.model = SentenceTransformer(
            snapshot, device="mps", trust_remote_code=False, local_files_only=True,
            model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"},
            processor_kwargs={"padding_side": "left"})
        self.model.max_seq_length = 8192
        self.model.eval()
        if self.model.prompts.get("query") != QUERY_PROMPT or self.model.get_embedding_dimension() != 1024:
            raise ValueError("Downloaded model differs from the fixed pilot recipe")

    def encode(self, texts, kind):
        self.torch.mps.synchronize()
        with self.torch.inference_mode():
            values = self.model.encode(texts, batch_size=64,
                                       prompt_name="query" if kind == "queries" else None,
                                       show_progress_bar=False, convert_to_numpy=True)
        self.torch.mps.synchronize()
        print(json.dumps(dict(kind=kind,
                              mps_driver_allocated_bytes=self.torch.mps.driver_allocated_memory(),
                              mps_tensor_allocated_bytes=self.torch.mps.current_allocated_memory())), flush=True)
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "quora")
    parser.add_argument("--queries", type=Path, default=BASE / "quora-test-queries-1000.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "quora-qwen-full")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    pilot = json.loads((BASE / "qwen-quora-pilot.json").read_text())
    for name, expected in pilot["metadata"]["file_sha256"].items():
        if file_hash(args.source / name) != expected:
            raise ValueError(f"Official Quora source hash changed: {name}")
    if file_hash(args.queries) != pilot["metadata"]["selected_query_files"]["1000"]["sha256"]:
        raise ValueError("Fixed 1,000-query selection changed")
    source = prepare_sources(args.source, args.queries)
    if len(source["corpus_ids"]) != 522931 or len(source["query_ids"]) != 1000:
        raise ValueError("Expected the complete corpus and exactly 1,000 fixed test queries")
    identity = dict(model=MODEL, revision=REVISION, dimensions=1024, batch_size=64, device="mps",
                    dtype="float16", max_length=8192, attention="sdpa", padding_side="left",
                    pooling="last token", query_prompt=QUERY_PROMPT, document_prompt="",
                    recipe="qwen3-st-lasttoken-normalize-fp16-to-fp32-l2-v1", chunk_size=args.chunk_size,
                    source_sha256=pilot["metadata"]["file_sha256"],
                    selected_queries_sha256=file_hash(args.queries),
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "sentence-transformers", "numpy")})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Output belongs to a different encoding; use a new directory")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Output directory is not an identified Quora encoding cache")
        manifest = dict(state="encoding", identity=identity)
    manifest.update(document_count=len(source["corpus_ids"]), query_count=len(source["query_ids"]),
                    query_ids=source["query_ids"], overlap=source["overlap"])
    write_json(manifest_path, manifest)
    for name, value in (("corpus-ids.json", source["corpus_ids"]), ("query-ids.json", source["query_ids"])):
        write_json(args.output / name, value)
    shutil.copyfile(args.queries, args.output / "queries.jsonl")
    shutil.copyfile(args.source / "qrels/test.tsv", args.output / "qrels-test.tsv")
    encoder = None

    def encode(texts, kind):
        nonlocal encoder
        if encoder is None:
            encoder = QwenEncoder()
        return encoder.encode(texts, kind)

    started = perf_counter()
    arrays = {}
    for kind, texts in (("documents", source["document_texts"]), ("queries", source["query_texts"])):
        path, progress = encode_chunks(texts, args.output, kind, identity,
                                       lambda values: encode(values, kind), chunk_size=args.chunk_size)
        arrays[kind] = dict(path=str(path.resolve()), sha256=progress["array_sha256"],
                            shape=[len(texts), 1024], dtype="float32")
    manifest.update(state="complete", arrays=arrays, current_run_seconds=perf_counter() - started,
                    files={name: file_hash(args.output / name) for name in
                           ("corpus-ids.json", "query-ids.json", "queries.jsonl", "qrels-test.tsv")})
    write_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
