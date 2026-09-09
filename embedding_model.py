"""Shared Nomic encoding and Matryoshka normalization recipe."""

import importlib.metadata
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer


FULL_DIMENSIONS = 768
RECIPE = "nomic-v1.5-st-layernorm-eps1e-5-prefix-l2-v1"


def resolve_revision(model_name: str, revision: str | None) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    resolved = model_info(model_name, revision=revision).sha
    if not resolved:
        raise ValueError(f"Could not resolve a commit for {model_name}.")
    return resolved


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def package_versions() -> dict[str, str]:
    """Return versions that can change model output or array preparation."""
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "sentence-transformers", "torch")
    }


class NomicEncoder:
    """Keep one loaded model available for several encoding chunks."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        batch_size: int,
        max_length: int,
        device: str,
        cache_dir: Path,
    ):
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=False,
            cache_folder=str(cache_dir),
        )
        self.model.max_seq_length = max_length

    def encode_prefixed(self, texts: list[str]) -> np.ndarray:
        """Encode text that already has its document or query task prefix."""
        values = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(texts), FULL_DIMENSIONS) or not np.isfinite(values).all():
            raise ValueError("Expected finite 768-dimensional Nomic embeddings.")
        return values


def encode_texts(
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
    """Compatibility wrapper used by the existing all-at-once cache."""
    encoder = NomicEncoder(
        model_name=model_name,
        revision=revision,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        cache_dir=Path(cache_dir) / "model",
    )
    return encoder.encode_prefixed(document_texts), encoder.encode_prefixed(query_texts)


def matryoshka_vectors(raw: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != FULL_DIMENSIONS or not np.isfinite(raw).all():
        raise ValueError("Expected finite 768-dimensional Nomic embeddings.")
    centered = raw - raw.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    layer_normalized = centered / np.sqrt(variance + 1e-5)
    full_norm = np.linalg.norm(layer_normalized, axis=1, keepdims=True)
    short = layer_normalized[:, :dimensions].copy()
    short_norm = np.linalg.norm(short, axis=1, keepdims=True)
    if np.any(full_norm == 0) or np.any(short_norm == 0):
        raise ValueError("The encoder produced an empty embedding.")
    short_vectors = np.ascontiguousarray(short / short_norm)
    full_vectors = np.ascontiguousarray(layer_normalized / full_norm)
    return short_vectors, full_vectors


def normalized_prefix(full_vectors: np.ndarray, dimensions: int) -> np.ndarray:
    prefix = np.asarray(full_vectors, dtype=np.float32)[:, :dimensions].copy()
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The cached full vectors have an empty prefix.")
    return np.ascontiguousarray(prefix / norms)
