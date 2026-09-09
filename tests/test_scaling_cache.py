import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scaling_cache
from embedding_model import matryoshka_vectors, normalized_prefix
from scaling_cache import derive_cache, prepare_cache
from scaling_config import ScalingConfigError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _selection(tmp_path: Path, name: str = "selection", *, reverse: bool = False) -> dict:
    selection_root = tmp_path / name
    selection_root.mkdir(parents=True)
    documents = [
        {"document_id": "d0", "text": "zero"},
        {"document_id": "d1", "text": "one"},
        {"document_id": "d2", "text": "two"},
        {"document_id": "d3", "text": "three"},
        {"document_id": "d4", "text": "four"},
    ]
    if reverse:
        documents.reverse()
    queries = [
        {"query_id": "q9", "text": "nine"},
        {"query_id": "q2", "text": "two?"},
    ]
    documents_path = selection_root / "documents.jsonl"
    queries_path = selection_root / "queries.jsonl"
    _write_jsonl(documents_path, documents)
    _write_jsonl(queries_path, queries)
    content_identity = {
        "documents_sha256": _sha256(documents_path),
        "queries_sha256": _sha256(queries_path),
        "query_ids": [row["query_id"] for row in queries],
    }
    selection_hash = hashlib.sha256(
        json.dumps(content_identity, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "selection_hash": selection_hash,
        "identity": content_identity,
        "document_count": len(documents),
        "query_count": len(queries),
        "query_ids": content_identity["query_ids"],
        "files": {
            "documents": {
                "path": str(documents_path.resolve()),
                "sha256": content_identity["documents_sha256"],
            },
            "queries": {
                "path": str(queries_path.resolve()),
                "sha256": content_identity["queries_sha256"],
            },
        },
    }
    (selection_root / "selection.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _config(*, revision: str = "a" * 40, device: str = "auto", model: str = "nomic") -> dict:
    return {
        "embedding": {
            "model": model,
            "revision": revision,
            "batch_size": 2,
            "max_length": 128,
            "device": device,
            "chunk_size": 2,
        },
        "search": {"dimensions": 256},
    }


def _raw_vector(text: str) -> np.ndarray:
    coordinates = np.arange(768, dtype=np.float32)
    text_number = sum((index + 1) * ord(character) for index, character in enumerate(text))
    frequency = np.float32((text_number % 17 + 3) / 97)
    curve = np.sin(coordinates * frequency) + np.cos(coordinates * frequency * 0.37)
    return (curve + (coordinates % (text_number % 11 + 2)) * 0.013).astype(np.float32)


class FakeEncoder:
    instances: list["FakeEncoder"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[list[str]] = []
        self.__class__.instances.append(self)

    def encode_prefixed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([_raw_vector(text) for text in texts])


@pytest.fixture
def fake_encoder(monkeypatch):
    FakeEncoder.instances = []
    monkeypatch.setattr(scaling_cache, "NomicEncoder", FakeEncoder)
    monkeypatch.setattr(scaling_cache, "choose_device", lambda requested: "cpu")
    monkeypatch.setattr(
        scaling_cache,
        "package_versions",
        lambda: {"numpy": "test", "sentence-transformers": "test", "torch": "test"},
    )
    return FakeEncoder


def _independent_recipe(raw: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    values = raw.astype(np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    layer_normalized = centered / np.sqrt(
        np.mean(centered * centered, axis=1, keepdims=True) + 1e-5
    )
    full = layer_normalized / np.linalg.norm(layer_normalized, axis=1, keepdims=True)
    short = layer_normalized[:, :dimensions]
    short = short / np.linalg.norm(short, axis=1, keepdims=True)
    return short.astype(np.float32), full.astype(np.float32)


def test_shared_recipe_matches_independent_layernorm_prefix_and_l2():
    first = np.concatenate(
        [np.full(256, 5, dtype=np.float32), np.full(512, -1, dtype=np.float32)]
    )
    second = np.tile(np.array([-3, -1, 1, 3], dtype=np.float32), 192)
    raw = np.stack([first, second])
    expected_full = np.stack(
        [
            np.concatenate([np.full(256, 4), np.full(512, -2)]) / np.sqrt(6144),
            second / np.sqrt(3840),
        ]
    ).astype(np.float32)
    expected_short = np.stack(
        [np.full(256, 1 / 16), second[:256] / np.sqrt(1280)]
    ).astype(np.float32)

    short, full = matryoshka_vectors(raw, 256)

    np.testing.assert_allclose(short, expected_short, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(full, expected_full, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(normalized_prefix(full, 256), expected_short, rtol=1e-6, atol=1e-7)


def test_prepare_cache_preserves_prefixed_document_and_query_row_order(
    tmp_path, fake_encoder
):
    selection = _selection(tmp_path)

    manifest = prepare_cache(selection, _config())

    assert len(fake_encoder.instances) == 1
    assert fake_encoder.instances[0].calls == [
        ["search_document: zero", "search_document: one"],
        ["search_document: two", "search_document: three"],
        ["search_document: four"],
        ["search_query: nine", "search_query: two?"],
    ]
    documents = np.load(manifest["arrays"]["documents"]["path"], allow_pickle=False)
    queries = np.load(manifest["arrays"]["queries"]["path"], allow_pickle=False)
    expected_documents = np.stack(
        [
            _raw_vector(f"search_document: {text}")
            for text in ["zero", "one", "two", "three", "four"]
        ]
    )
    expected_queries = np.stack(
        [_raw_vector(f"search_query: {text}") for text in ["nine", "two?"]]
    )
    np.testing.assert_allclose(documents, _independent_recipe(expected_documents, 768)[1])
    np.testing.assert_allclose(queries, _independent_recipe(expected_queries, 768)[1])
    assert manifest["query_ids"] == ["q9", "q2"]


def test_check_only_then_prepare_encodes_only_remaining_document_chunks(
    tmp_path, fake_encoder
):
    selection = _selection(tmp_path)

    partial = prepare_cache(selection, _config(), check_only=True)
    complete = prepare_cache(selection, _config())
    reused = prepare_cache(selection, _config())

    assert partial["state"] == "partial"
    assert partial["completed_document_rows"] == 2
    assert complete["state"] == reused["state"] == "complete"
    assert len(fake_encoder.instances) == 2
    assert fake_encoder.instances[0].calls == [
        ["search_document: zero", "search_document: one"],
        ["search_query: nine", "search_query: two?"],
    ]
    assert fake_encoder.instances[1].calls == [
        ["search_document: two", "search_document: three"],
        ["search_document: four"],
    ]


def test_interrupted_chunk_publication_repeats_only_that_chunk(
    tmp_path, fake_encoder, monkeypatch
):
    selection = _selection(tmp_path)
    real_save = scaling_cache._save_array_atomic
    failed = False

    def interrupt_second_document_chunk(path, values):
        nonlocal failed
        if path.name == "documents-000000000002-000000000004.npy" and not failed:
            failed = True
            temporary = path.with_name(f".{path.name}.interrupted")
            np.save(temporary, values)
            raise RuntimeError("interrupted chunk write")
        return real_save(path, values)

    monkeypatch.setattr(scaling_cache, "_save_array_atomic", interrupt_second_document_chunk)
    with pytest.raises(RuntimeError, match="interrupted chunk"):
        prepare_cache(selection, _config())
    monkeypatch.setattr(scaling_cache, "_save_array_atomic", real_save)

    manifest = prepare_cache(selection, _config())

    assert manifest["state"] == "complete"
    assert len(fake_encoder.instances) == 2
    assert fake_encoder.instances[1].calls == [
        ["search_document: two", "search_document: three"],
        ["search_document: four"],
        ["search_query: nine", "search_query: two?"],
    ]


def test_interrupted_concatenation_recopies_chunks_without_encoding(
    tmp_path, fake_encoder, monkeypatch
):
    selection = _selection(tmp_path)
    real_concatenate = scaling_cache._concatenate_document_chunks
    failed = False

    def interrupt(cache_dir, chunk_paths, rows):
        nonlocal failed
        if not failed:
            failed = True
            temporary = cache_dir / ".full-documents.npy.interrupted"
            np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(rows, 768))
            raise RuntimeError("interrupted concatenation")
        return real_concatenate(cache_dir, chunk_paths, rows)

    monkeypatch.setattr(scaling_cache, "_concatenate_document_chunks", interrupt)
    with pytest.raises(RuntimeError, match="interrupted concatenation"):
        prepare_cache(selection, _config())
    encoded_call_count = sum(len(instance.calls) for instance in fake_encoder.instances)

    manifest = prepare_cache(selection, _config())

    assert manifest["state"] == "complete"
    assert sum(len(instance.calls) for instance in fake_encoder.instances) == encoded_call_count
    assert len(fake_encoder.instances) == 1


def test_complete_cache_reuse_checks_saved_input_hashes(tmp_path, fake_encoder):
    selection = _selection(tmp_path)
    first = prepare_cache(selection, _config())
    second = prepare_cache(selection, _config())

    assert first == second
    assert len(fake_encoder.instances) == 1

    documents_path = Path(selection["files"]["documents"]["path"])
    documents_path.write_text(documents_path.read_text() + '{"document_id":"extra","text":"bad"}\n')
    with pytest.raises(ScalingConfigError) as error:
        prepare_cache(selection, _config())
    assert error.value.code == "INPUT_MISMATCH"
    assert "documents" in str(error.value)


def test_revision_device_model_and_selection_change_cache_identity(
    tmp_path, fake_encoder, monkeypatch
):
    selection = _selection(tmp_path, "first")
    first = prepare_cache(selection, _config())
    revision = prepare_cache(selection, _config(revision="b" * 40))
    model = prepare_cache(selection, _config(model="other-model"))
    monkeypatch.setattr(scaling_cache, "choose_device", lambda requested: "mps")
    device = prepare_cache(selection, _config())
    other_selection = _selection(tmp_path, "second", reverse=True)
    other = prepare_cache(other_selection, _config())

    assert len({item["embedding_hash"] for item in [first, revision, model, device, other]}) == 5
    assert len(fake_encoder.instances) == 5


def test_derive_cache_reuses_full_cache_for_another_width_without_encoding(
    tmp_path, fake_encoder
):
    full = prepare_cache(_selection(tmp_path), _config())

    derived_256 = derive_cache(full, 256)
    derived_128 = derive_cache(full, 128)
    reused_128 = derive_cache(full, 128)

    assert len(fake_encoder.instances) == 1
    assert derived_128 == reused_128
    assert derived_256["input_hash"] != derived_128["input_hash"]
    full_documents = np.load(full["arrays"]["documents"]["path"], mmap_mode="r")
    full_queries = np.load(full["arrays"]["queries"]["path"], mmap_mode="r")
    codes = np.load(derived_128["arrays"]["codes"]["path"], allow_pickle=False)
    queries = np.load(derived_128["arrays"]["queries"]["path"], allow_pickle=False)
    np.testing.assert_array_equal(codes, scaling_cache.encode_signs(full_documents[:, :128]))
    np.testing.assert_allclose(queries, normalized_prefix(full_queries, 128))
    assert derived_128["query_ids"] == ["q9", "q2"]
    assert derived_128["arrays"]["codes"]["shape"] == [5, 2]
    assert derived_128["arrays"]["queries"]["shape"] == [2, 128]
