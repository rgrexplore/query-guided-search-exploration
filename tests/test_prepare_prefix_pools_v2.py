"""Preparation keeps query selection, embedding recipes, and exact rows reproducible."""

import hashlib
import json
import tarfile
from io import BytesIO

import numpy as np
import pytest

from embedding_model import matryoshka_vectors, normalized_prefix
from experiments import prepare_prefix_pools_v2 as preparation


def query_archive(path, rows):
    content = "".join(f"{identifier}\t{text}\n" for identifier, text in rows).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("queries.dev.small.tsv")
        member.size = len(content)
        archive.addfile(member, BytesIO(content))
    return hashlib.sha256(content).hexdigest()


def test_selection_is_fixed_before_embedding_and_ignores_input_order(tmp_path):
    rows = [(str(i), f"question {i}") for i in range(20)]
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_hash = query_archive(first, rows)
    second_hash = query_archive(second, list(reversed(rows)))
    selection = preparation.prepare_selection(first, tmp_path / "one", 8, 42, first_hash)
    other = preparation.prepare_selection(second, tmp_path / "two", 8, 42, second_hash)
    assert selection["query_ids"] == other["query_ids"]
    assert len(set(selection["query_ids"])) == 8
    assert selection["source_queries"] == 20
    assert selection["query_ids"] != [str(i) for i in range(8)]
    assert preparation.prepare_selection(first, tmp_path / "one", 8, 42, first_hash) == selection
    with pytest.raises(ValueError, match="selection"):
        preparation.prepare_selection(first, tmp_path / "one", 9, 42, first_hash)


def test_official_query_member_hash_is_checked(tmp_path):
    archive = tmp_path / "queries.tar.gz"
    query_archive(archive, [("1", "question")])
    with pytest.raises(ValueError, match="SHA"):
        preparation.prepare_selection(archive, tmp_path / "selection", 1, 42, "wrong")


def test_query_encoding_uses_nomic_prefix_and_normalizes_raw_values_once(tmp_path, monkeypatch):
    archive = tmp_path / "queries.tar.gz"
    digest = query_archive(archive, [(str(i), f"question {i}") for i in range(5)])
    folder = tmp_path / "queries"
    selection = preparation.prepare_selection(archive, folder, 5, 42, digest)
    calls = []
    raw = np.arange(768, dtype=np.float32)[None, :] + np.arange(5, dtype=np.float32)[:, None]

    class Encoder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def encode_prefixed(self, texts):
            calls.append(texts)
            return np.stack([raw[int(text.rsplit(" ", 1)[1])] for text in texts])

    monkeypatch.setattr(preparation, "NomicEncoder", Encoder)
    preparation.encode_nomic_queries(folder, cache_dir=tmp_path, chunk_size=2, device="cpu")
    values = np.load(folder / "full-queries.npy")
    ordered_raw = raw[[int(identifier) for identifier in selection["query_ids"]]]
    expected = matryoshka_vectors(ordered_raw, 768)[1]
    np.testing.assert_array_equal(values, expected)
    assert all(text.startswith("search_query: ") for chunk in calls[1:] for text in chunk)
    assert calls[0]["revision"] == preparation.NOMIC_REVISION
    preparation.encode_nomic_queries(folder, cache_dir=tmp_path, chunk_size=2, device="cpu")
    assert len(calls) == 4  # One model load and three chunks; completed cache is reused.


def vectors_fixture(tmp_path):
    documents = np.array([[1, 2, -1, -3], [1, 2, -1, -3], [-1, 1, 1, -1],
                          [-2, -1, 1, 2], [1, -1, 1, -1]], dtype=np.float32)
    queries = np.array([[1, 1, -1, -1], [0.25, 0, 0, 1], [1, -1, 1, -1]], dtype=np.float32)
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    docs_path, queries_path = tmp_path / "full-documents.npy", tmp_path / "full-queries.npy"
    np.save(docs_path, documents)
    np.save(queries_path, queries)
    return docs_path, queries_path, documents, queries


@pytest.mark.parametrize("dimensions", [2, 4])
def test_pool_uses_normalized_prefix_not_layernorm_and_exact_tie_order(tmp_path, dimensions):
    docs_path, queries_path, documents, queries = vectors_fixture(tmp_path)
    pool = tmp_path / "pool"
    preparation.prepare_pool(docs_path, queries_path, ["a", "b", "c"], pool,
                             dimensions=dimensions, top_k=3, provenance={"model": "fixture"}, chunk_size=2)
    expected_docs = normalized_prefix(documents, dimensions)
    expected_queries = normalized_prefix(queries, dimensions)
    np.testing.assert_array_equal(np.load(pool / "documents.npy"), expected_docs)
    np.testing.assert_array_equal(np.load(pool / "queries.npy"), expected_queries)
    full_scores = expected_queries.astype(np.float64) @ (2 * (expected_docs >= 0).astype(np.float64) - 1).T
    expected = np.argsort(-full_scores, axis=1, kind="stable")[:, :3]
    np.testing.assert_array_equal(np.load(pool / "reference.npy"), expected)
    assert expected[0, :2].tolist() == [0, 1]
    metadata = json.loads((pool / "pool.json").read_text())
    assert metadata["query_ids"] == ["a", "b", "c"]
    assert metadata["identity"]["dimensions"] == dimensions
    assert set(metadata["hashes"]) == {"codes.npy", "documents.npy", "queries.npy", "reference.npy"}


def test_reference_checkpoint_resumes_without_scoring_finished_queries(tmp_path):
    docs_path, queries_path, documents, queries = vectors_fixture(tmp_path)
    pool = tmp_path / "pool"
    preparation.prepare_pool(docs_path, queries_path, ["a", "b", "c"], pool,
                             dimensions=4, top_k=3, provenance={}, chunk_size=1, reference_chunk_size=1,
                             reference_limit=1)
    progress = json.loads((pool / "reference-progress.json").read_text())
    assert progress["completed_queries"] == 1
    assert not (pool / "pool.json").exists()
    preparation.prepare_pool(docs_path, queries_path, ["a", "b", "c"], pool,
                             dimensions=4, top_k=3, provenance={}, chunk_size=1, reference_chunk_size=1)
    progress = json.loads((pool / "reference-progress.json").read_text())
    assert progress["completed_queries"] == 3
    assert [block["start"] for block in progress["chunks"]] == [0, 1, 2]
    assert (pool / "pool.json").exists()


def test_pool_rejects_changed_inputs_in_a_completed_folder(tmp_path):
    docs_path, queries_path, _, _ = vectors_fixture(tmp_path)
    pool = tmp_path / "pool"
    preparation.prepare_pool(docs_path, queries_path, ["a", "b", "c"], pool,
                             dimensions=4, top_k=3, provenance={})
    with pytest.raises(ValueError, match="inputs"):
        preparation.prepare_pool(docs_path, queries_path, ["c", "b", "a"], pool,
                                 dimensions=4, top_k=3, provenance={})
