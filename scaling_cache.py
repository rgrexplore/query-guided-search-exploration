"""Prepare resumable full embeddings and dimension-specific search arrays."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from embedding_model import (
    FULL_DIMENSIONS,
    RECIPE,
    NomicEncoder,
    choose_device,
    matryoshka_vectors,
    normalized_prefix,
    package_versions,
    resolve_revision,
)
from embeddings import encode_signs
from scaling_config import ScalingConfigError, StudyConfig, identity_hash


EmbeddingManifest = dict[str, Any]
DerivedManifest = dict[str, Any]


def _invalid(code: str, field: str, message: str) -> None:
    raise ScalingConfigError(code, field, message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            np.save(output, values, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _check_file_hash(record: dict, name: str, code: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        _invalid(code, name, f"file is missing: {path}")
    observed = _sha256(path)
    if observed != record["sha256"]:
        _invalid(code, name, f"file hash does not match: {path}")
    return path


def _selection_inputs(selection: dict) -> tuple[Path, Path, Path]:
    if len(selection["query_ids"]) != selection["query_count"]:
        _invalid("INPUT_MISMATCH", "query_ids", "must match query_count")
    documents_path = _check_file_hash(
        selection["files"]["documents"], "documents", "INPUT_MISMATCH"
    )
    queries_path = _check_file_hash(
        selection["files"]["queries"], "queries", "INPUT_MISMATCH"
    )
    if documents_path.parent != queries_path.parent:
        _invalid("INPUT_MISMATCH", "selection", "document and query files must share a folder")
    manifest_path = documents_path.parent / "selection.json"
    if not manifest_path.is_file():
        _invalid("INPUT_MISMATCH", "selection", f"manifest is missing: {manifest_path}")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved != selection:
        _invalid(
            "INPUT_MISMATCH",
            "selection",
            "saved manifest does not match the supplied selection",
        )
    return documents_path, queries_path, manifest_path.resolve()


def _read_jsonl(path: Path, id_name: str, expected_rows: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != expected_rows:
        _invalid(
            "INPUT_MISMATCH",
            str(path),
            f"has {len(rows)} rows, expected {expected_rows}",
        )
    return rows


def _document_text_chunks(
    path: Path, chunk_size: int, expected_rows: int
) -> Iterator[tuple[int, int, list[str]]]:
    texts: list[str] = []
    start = 0
    row_count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            texts.append(f"search_document: {row['text']}")
            row_count += 1
            if len(texts) == chunk_size:
                yield start, row_count, texts
                start = row_count
                texts = []
    if texts:
        yield start, row_count, texts
    if row_count != expected_rows:
        _invalid(
            "INPUT_MISMATCH",
            str(path),
            f"has {row_count} rows, expected {expected_rows}",
        )


def _chunk_path(chunks_dir: Path, start: int, stop: int) -> Path:
    return chunks_dir / f"documents-{start:012d}-{stop:012d}.npy"


def _load_array(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != dtype:
        _invalid(
            "CACHE_INVALID",
            str(path),
            f"has shape {values.shape} and dtype {values.dtype}; expected {shape} and {dtype}",
        )
    if not np.isfinite(values).all():
        _invalid("CACHE_INVALID", str(path), "contains non-finite values")
    return values


def _array_record(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "shape": list(shape),
        "dtype": np.dtype(dtype).name,
    }


def _validate_array_record(record: dict, name: str) -> np.ndarray:
    path = _check_file_hash(record, name, "CACHE_INVALID")
    return _load_array(path, tuple(record["shape"]), np.dtype(record["dtype"]))


def _validate_complete_manifest(
    manifest: dict,
    *,
    expected_identity: dict | None = None,
    expected_selection_path: Path | None = None,
) -> EmbeddingManifest:
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        _invalid("CACHE_INVALID", "manifest", "full embedding cache is not complete")
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        _invalid("CACHE_INVALID", "identity", "does not match the requested encoding")
    if expected_selection_path is not None and manifest.get("selection_manifest_path") != str(
        expected_selection_path
    ):
        _invalid("CACHE_INVALID", "selection_manifest_path", "does not match the selection")
    if manifest.get("embedding_hash") != identity_hash(manifest.get("identity")):
        _invalid("CACHE_INVALID", "embedding_hash", "does not match the encoding identity")
    document_count = manifest["document_count"]
    query_count = manifest["query_count"]
    if len(manifest["query_ids"]) != query_count:
        _invalid("CACHE_INVALID", "query_ids", "must match query_count")
    documents = _validate_array_record(manifest["arrays"]["documents"], "documents")
    queries = _validate_array_record(manifest["arrays"]["queries"], "queries")
    if documents.shape != (document_count, FULL_DIMENSIONS):
        _invalid("CACHE_INVALID", "documents", "row count does not match manifest")
    if queries.shape != (query_count, FULL_DIMENSIONS):
        _invalid("CACHE_INVALID", "queries", "row count does not match manifest")
    return manifest


def _concatenate_document_chunks(
    cache_dir: Path, chunk_paths: list[Path], rows: int
) -> Path:
    final_path = cache_dir / "full-documents.npy"
    with tempfile.NamedTemporaryFile(
        dir=cache_dir, prefix=".full-documents.npy.", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        output = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(rows, FULL_DIMENSIONS),
        )
        offset = 0
        for chunk_path in chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
            output[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        output.flush()
        del output
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return final_path


def prepare_cache(
    selection: dict,
    config: StudyConfig,
    check_only: bool = False,
) -> EmbeddingManifest:
    """Encode missing atomic chunks, then publish complete full arrays."""
    documents_path, queries_path, selection_manifest_path = _selection_inputs(selection)
    embedding = config["embedding"]
    try:
        actual_device = choose_device(embedding["device"])
        revision = resolve_revision(embedding["model"], embedding["revision"])
        versions = package_versions()
    except Exception as error:
        _invalid("MODEL_FAILED", "embedding", str(error))
    identity = {
        "schema_version": 1,
        "selection_hash": selection["selection_hash"],
        "model": embedding["model"],
        "revision": revision,
        "recipe": RECIPE,
        "full_dimensions": FULL_DIMENSIONS,
        "device": actual_device,
        "batch_size": embedding["batch_size"],
        "max_length": embedding["max_length"],
        "chunk_size": embedding["chunk_size"],
        "package_versions": versions,
    }
    embedding_hash = identity_hash(identity)
    selection_root = documents_path.parent
    cache_dir = selection_root / "embeddings" / embedding_hash
    chunks_dir = cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_complete_manifest(
            manifest,
            expected_identity=identity,
            expected_selection_path=selection_manifest_path,
        )

    encoder: NomicEncoder | None = None

    def get_encoder() -> NomicEncoder:
        nonlocal encoder
        if encoder is None:
            model_cache = selection_root.parent.parent / "model"
            try:
                encoder = NomicEncoder(
                    model_name=embedding["model"],
                    revision=revision,
                    batch_size=embedding["batch_size"],
                    max_length=embedding["max_length"],
                    device=actual_device,
                    cache_dir=model_cache,
                )
            except Exception as error:
                _invalid("MODEL_FAILED", "embedding.model", str(error))
        return encoder

    chunk_paths: list[Path] = []
    completed_document_rows = 0
    for start, stop, texts in _document_text_chunks(
        documents_path, embedding["chunk_size"], selection["document_count"]
    ):
        chunk_path = _chunk_path(chunks_dir, start, stop)
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            _load_array(chunk_path, (stop - start, FULL_DIMENSIONS), np.dtype("float32"))
        elif not check_only or start == 0:
            try:
                raw = get_encoder().encode_prefixed(texts)
                _, full = matryoshka_vectors(raw, FULL_DIMENSIONS)
            except ScalingConfigError:
                raise
            except Exception as error:
                _invalid("MODEL_FAILED", "documents", f"rows {start}:{stop}: {error}")
            _save_array_atomic(chunk_path, full)
        if chunk_path.exists():
            completed_document_rows = stop

    query_chunk_path = chunks_dir / "queries.npy"
    query_rows = _read_jsonl(queries_path, "query_id", selection["query_count"])
    observed_query_ids = [row["query_id"] for row in query_rows]
    if observed_query_ids != selection["query_ids"]:
        _invalid("INPUT_MISMATCH", "query_ids", "query file order does not match selection")
    if query_chunk_path.exists():
        _load_array(
            query_chunk_path,
            (selection["query_count"], FULL_DIMENSIONS),
            np.dtype("float32"),
        )
    else:
        query_texts = [f"search_query: {row['text']}" for row in query_rows]
        try:
            raw_queries = get_encoder().encode_prefixed(query_texts)
            _, full_queries = matryoshka_vectors(raw_queries, FULL_DIMENSIONS)
        except ScalingConfigError:
            raise
        except Exception as error:
            _invalid("MODEL_FAILED", "queries", str(error))
        _save_array_atomic(query_chunk_path, full_queries)

    if check_only and completed_document_rows < selection["document_count"]:
        return {
            "schema_version": 1,
            "state": "partial",
            "embedding_hash": embedding_hash,
            "identity": identity,
            "selection_hash": selection["selection_hash"],
            "selection_manifest_path": str(selection_manifest_path),
            "query_ids": list(selection["query_ids"]),
            "document_count": selection["document_count"],
            "query_count": selection["query_count"],
            "cache_dir": str(cache_dir.resolve()),
            "completed_document_rows": completed_document_rows,
        }

    missing = [path for path in chunk_paths if not path.exists()]
    if missing:
        _invalid("CACHE_INVALID", str(missing[0]), "required document chunk is missing")
    full_documents_path = _concatenate_document_chunks(
        cache_dir, chunk_paths, selection["document_count"]
    )
    full_queries_path = cache_dir / "full-queries.npy"
    query_values = _load_array(
        query_chunk_path,
        (selection["query_count"], FULL_DIMENSIONS),
        np.dtype("float32"),
    )
    _save_array_atomic(full_queries_path, query_values)
    manifest: EmbeddingManifest = {
        "schema_version": 1,
        "state": "complete",
        "embedding_hash": embedding_hash,
        "identity": identity,
        "selection_hash": selection["selection_hash"],
        "selection_manifest_path": str(selection_manifest_path),
        "query_ids": list(selection["query_ids"]),
        "document_count": selection["document_count"],
        "query_count": selection["query_count"],
        "arrays": {
            "documents": _array_record(
                full_documents_path,
                (selection["document_count"], FULL_DIMENSIONS),
                np.dtype("float32"),
            ),
            "queries": _array_record(
                full_queries_path,
                (selection["query_count"], FULL_DIMENSIONS),
                np.dtype("float32"),
            ),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def derive_cache(full: EmbeddingManifest, dimensions: int) -> DerivedManifest:
    """Create packed document signs and normalized query prefixes without the model."""
    if dimensions not in {64, 128, 256, 512, 768}:
        _invalid("INPUT_MISMATCH", "dimensions", "must be 64, 128, 256, 512, or 768")
    full = _validate_complete_manifest(full)
    input_identity = {
        "embedding_hash": full["embedding_hash"],
        "dimensions": dimensions,
        "document_sha256": full["arrays"]["documents"]["sha256"],
        "query_sha256": full["arrays"]["queries"]["sha256"],
        "query_ids": full["query_ids"],
    }
    input_hash = identity_hash(input_identity)
    full_documents_path = Path(full["arrays"]["documents"]["path"])
    cache_dir = full_documents_path.parent / "derived" / str(dimensions)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("input_hash") != input_hash
            or manifest.get("embedding_hash") != full["embedding_hash"]
            or manifest.get("dimensions") != dimensions
            or manifest.get("query_ids") != full["query_ids"]
        ):
            _invalid("CACHE_INVALID", str(manifest_path), "derived identity does not match")
        _validate_array_record(manifest.get("arrays", {}).get("codes", {}), "codes")
        _validate_array_record(manifest.get("arrays", {}).get("queries", {}), "queries")
        return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    document_count = full["document_count"]
    word_count = (dimensions + 63) // 64
    codes_path = cache_dir / "codes.npy"
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".codes.npy.", delete=False) as output:
        codes_temporary = Path(output.name)
    try:
        codes = np.lib.format.open_memmap(
            codes_temporary,
            mode="w+",
            dtype=np.uint64,
            shape=(document_count, word_count),
        )
        documents = np.load(full_documents_path, mmap_mode="r", allow_pickle=False)
        chunk_size = full["identity"]["chunk_size"]
        for start in range(0, document_count, chunk_size):
            stop = min(start + chunk_size, document_count)
            codes[start:stop] = encode_signs(documents[start:stop, :dimensions])
        codes.flush()
        del codes
        codes_temporary.replace(codes_path)
    finally:
        codes_temporary.unlink(missing_ok=True)

    full_queries = np.load(full["arrays"]["queries"]["path"], mmap_mode="r", allow_pickle=False)
    derived_queries = normalized_prefix(full_queries, dimensions)
    queries_path = cache_dir / "queries.npy"
    _save_array_atomic(queries_path, derived_queries)
    manifest: DerivedManifest = {
        "schema_version": 1,
        "input_hash": input_hash,
        "embedding_hash": full["embedding_hash"],
        "dimensions": dimensions,
        "query_ids": list(full["query_ids"]),
        "arrays": {
            "codes": _array_record(
                codes_path,
                (document_count, word_count),
                np.dtype("uint64"),
            ),
            "queries": _array_record(
                queries_path,
                (full["query_count"], dimensions),
                np.dtype("float32"),
            ),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
