"""Cache one Nomic encoding, then use its shorter prefix for binary search."""

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from data import Dataset


@dataclass
class EmbeddingArrays:
    """Short vectors feed the index; full vectors are kept for the shared float reranker."""

    documents: np.ndarray
    queries: np.ndarray
    full_documents: np.ndarray
    full_queries: np.ndarray
    codes: np.ndarray
    metadata: dict


def encode_signs(vectors: np.ndarray) -> np.ndarray:
    """Coordinate 0 goes in the lowest bit. Zero counts as positive; padding stays zero."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Expected a matrix with at least one coordinate.")
    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must be finite.")
    word_count = (vectors.shape[1] + 63) // 64

    # One uint64 holds 64 signs. Pad the final word so C++ can read whole words safely.
    padded = np.zeros((len(vectors), word_count * 64), dtype=np.uint8)
    padded[:, : vectors.shape[1]] = vectors >= 0
    packed = np.packbits(padded, axis=1, bitorder="little")
    words = packed.view("<u8")
    return np.ascontiguousarray(words, dtype=np.uint64)


def _resolve_revision(model_name: str, revision: str | None) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    resolved = model_info(model_name, revision=revision).sha
    if not resolved:
        raise ValueError(f"Could not resolve a commit for {model_name}.")
    return resolved


def _choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_texts(
    document_texts: list[str],
    query_texts: list[str],
    *,
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    device: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    model = SentenceTransformer(
        model_name,
        revision=revision,
        device=device,
        trust_remote_code=False,
        cache_folder=str(cache_dir / "model"),
    )
    model.max_seq_length = max_length
    # Keep normalization below, where the full and short vectors can share the same recipe.
    options = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }
    documents = model.encode(document_texts, **options)
    queries = model.encode(query_texts, **options)
    return documents, queries


def _matryoshka_vectors(raw: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 768 or not np.isfinite(raw).all():
        raise ValueError("Expected finite 768-dimensional Nomic embeddings.")
    # The model recipe is: layer norm the full vector, take a prefix, then L2-normalize.
    centered = raw - raw.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    scale = np.sqrt(variance + 1e-5)
    normalized = centered / scale
    full_norm = np.linalg.norm(normalized, axis=1, keepdims=True)
    short = normalized[:, :dimensions].copy()
    short_norm = np.linalg.norm(short, axis=1, keepdims=True)
    if np.any(full_norm == 0) or np.any(short_norm == 0):
        raise ValueError("The encoder produced an empty embedding.")
    short_vectors = np.ascontiguousarray(short / short_norm)
    full_vectors = np.ascontiguousarray(normalized / full_norm)
    return short_vectors, full_vectors


def _validate_arrays(arrays: EmbeddingArrays, rows: int, queries: int, dimensions: int) -> None:
    expected_shapes = {
        "documents": (rows, dimensions),
        "queries": (queries, dimensions),
        "full_documents": (rows, 768),
        "full_queries": (queries, 768),
    }
    for name, shape in expected_shapes.items():
        values = getattr(arrays, name)
        wrong_shape = values.shape != shape
        wrong_dtype = values.dtype != np.float32
        has_invalid_values = not np.isfinite(values).all()
        if wrong_shape or wrong_dtype or has_invalid_values:
            raise ValueError(
                f"Invalid embedding cache: {name} has the wrong shape, type, or values."
            )
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5):
            raise ValueError(f"Invalid embedding cache: {name} must contain unit vectors.")
    if arrays.codes.dtype != np.uint64 or not np.array_equal(
        arrays.codes, encode_signs(arrays.documents)
    ):
        raise ValueError("Invalid embedding cache: binary codes do not match document signs.")


def _normalized_prefix(full_vectors: np.ndarray, dimensions: int) -> np.ndarray:
    prefix = full_vectors[:, :dimensions].copy()
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The cached full vectors have an empty prefix.")
    return np.ascontiguousarray(prefix / norms)


def prepare_embeddings(
    dataset: Dataset,
    cache_dir: Path,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    model_revision: str | None = None,
    dimensions: int = 256,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "auto",
) -> EmbeddingArrays:
    """Load a matching cache or encode once. A changed corpus/model/recipe gets a new cache."""
    if model_name != "nomic-ai/nomic-embed-text-v1.5":
        raise ValueError("This preprocessing recipe supports nomic-ai/nomic-embed-text-v1.5.")
    if dimensions not in {64, 128, 256, 512, 768}:
        raise ValueError("dimensions must be 64, 128, 256, 512, or 768.")
    if batch_size < 1 or not 1 <= max_length <= 2048:
        raise ValueError("batch_size must be positive; max_length must be in [1, 2048].")
    cache_dir = Path(cache_dir)
    revision = _resolve_revision(model_name, model_revision)
    actual_device = _choose_device(device)

    # Nomic uses different task prefixes for documents and queries.
    document_texts = []
    for title, text in zip(dataset.titles, dataset.texts, strict=True):
        if title:
            document_text = f"search_document: {title}\n{text}"
        else:
            document_text = f"search_document: {text}"
        document_texts.append(document_text)
    query_texts = [f"search_query: {text}" for text in dataset.queries]

    # Include row IDs and the exact text so an old cache can't silently shift the ID mapping.
    content = [dataset.corpus_ids, document_texts, dataset.query_ids, query_texts]
    content_bytes = json.dumps(content, ensure_ascii=False).encode()
    content_hash = hashlib.sha256(content_bytes).hexdigest()
    metadata = {
        "dataset_fingerprint": dataset.fingerprint,
        "content_sha256": content_hash,
        "model_name": model_name,
        "model_revision": revision,
        "recipe": "nomic-v1.5-st-layernorm-eps1e-5-prefix-l2-v1",
        "dimensions": dimensions,
        "full_dimensions": 768,
        "max_length": max_length,
        "device": actual_device,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    key = hashlib.sha256(metadata_bytes).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.npz"
    cached_arrays = None
    # Every cache includes the full vectors, so changing prefix length doesn't need the model.
    cache_dimensions = [dimensions] + sorted({64, 128, 256, 512, 768} - {dimensions}, reverse=True)
    for cached_dimensions in cache_dimensions:
        expected_metadata = {**metadata, "dimensions": cached_dimensions}
        expected_bytes = json.dumps(expected_metadata, sort_keys=True).encode()
        saved_key = hashlib.sha256(expected_bytes).hexdigest()[:24]
        saved_path = cache_dir / f"{saved_key}.npz"
        if not saved_path.exists():
            continue
        try:
            with np.load(saved_path, allow_pickle=False) as saved:
                saved_metadata = json.loads(str(saved["metadata"]))
                if saved_metadata != expected_metadata:
                    raise ValueError("Embedding cache metadata does not match the requested run.")
                cached_arrays = EmbeddingArrays(
                    documents=np.ascontiguousarray(saved["documents"]),
                    queries=np.ascontiguousarray(saved["queries"]),
                    full_documents=np.ascontiguousarray(saved["full_documents"]),
                    full_queries=np.ascontiguousarray(saved["full_queries"]),
                    codes=np.ascontiguousarray(saved["codes"]),
                    metadata=saved_metadata,
                )
            _validate_arrays(
                cached_arrays,
                len(dataset.corpus_ids),
                len(dataset.query_ids),
                cached_dimensions,
            )
        except (ValueError, KeyError, OSError, BadZipFile) as error:
            raise ValueError(
                f"Invalid embedding cache at {saved_path}; remove it and retry."
            ) from error
        print(
            f"Loaded cached embeddings: {len(dataset.corpus_ids):,} documents, "
            f"{len(dataset.query_ids):,} queries.",
            flush=True,
        )
        if cached_dimensions == dimensions:
            return cached_arrays
        break

    if cached_arrays is None:
        raw_documents, raw_queries = _encode_texts(
            document_texts,
            query_texts,
            model_name=model_name,
            revision=revision,
            batch_size=batch_size,
            max_length=max_length,
            device=actual_device,
            cache_dir=cache_dir,
        )
        documents, full_documents = _matryoshka_vectors(raw_documents, dimensions)
        queries, full_queries = _matryoshka_vectors(raw_queries, dimensions)
    else:
        full_documents = cached_arrays.full_documents
        full_queries = cached_arrays.full_queries
        documents = _normalized_prefix(full_documents, dimensions)
        queries = _normalized_prefix(full_queries, dimensions)
        print(
            f"Derived {dimensions}-dimensional vectors from the cached full vectors.",
            flush=True,
        )
    arrays = EmbeddingArrays(
        documents=documents,
        queries=queries,
        full_documents=full_documents,
        full_queries=full_queries,
        codes=encode_signs(documents),
        metadata=metadata,
    )
    _validate_arrays(arrays, len(dataset.corpus_ids), len(dataset.query_ids), dimensions)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.savez(
                output,
                documents=arrays.documents,
                queries=arrays.queries,
                full_documents=arrays.full_documents,
                full_queries=arrays.full_queries,
                codes=arrays.codes,
                metadata=json.dumps(metadata, sort_keys=True),
            )
            output.flush()
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return arrays
