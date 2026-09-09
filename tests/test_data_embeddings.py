import json
import zipfile

import numpy as np
import pytest

import embeddings
from data import load_dataset
from embeddings import encode_signs, prepare_embeddings


def write_dataset(root):
    folder = root / "fiqa"
    (folder / "qrels").mkdir(parents=True)
    # Keep file order different from ID order to catch accidental row/ID mixups.
    corpus = [
        {
            "_id": "doc-b",
            "title": "B",
            "text": "second document",
        },
        {
            "_id": "doc-a",
            "title": "A",
            "text": "first document",
        },
        {
            "_id": "doc-c",
            "title": "C",
            "text": "third document",
        },
    ]
    queries = [
        {
            "_id": "train-query",
            "text": "not evaluated",
        },
        {
            "_id": "query-9",
            "text": "question nine",
        },
        {
            "_id": "query-2",
            "text": "question two",
        },
    ]
    for name, rows in [("corpus.jsonl", corpus), ("queries.jsonl", queries)]:
        (folder / name).write_text("\n".join(json.dumps(row) for row in rows))
    (folder / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nquery-2\tdoc-c\t2\nquery-9\tdoc-a\t1\n"
    )
    return folder


def test_test_queries_keep_their_ids_and_qrels_after_document_subset(tmp_path):
    write_dataset(tmp_path)
    dataset = load_dataset(tmp_path, max_documents=1)
    assert dataset.corpus_ids == ["doc-b"]
    assert dataset.query_ids == ["query-9", "query-2"]
    assert dataset.qrels == {"query-9": {"doc-a": 1}, "query-2": {"doc-c": 2}}
    assert dataset.full_document_count == 3
    assert dataset.full_query_count == 2
    assert dataset.is_subset


def test_dataset_fingerprint_changes_with_text_or_selected_rows(tmp_path):
    folder = write_dataset(tmp_path)
    full = load_dataset(tmp_path)
    subset = load_dataset(tmp_path, max_queries=1)
    assert full.fingerprint != subset.fingerprint
    corpus_path = folder / "corpus.jsonl"
    corpus_path.write_text(corpus_path.read_text().replace("second document", "changed document"))
    assert full.fingerprint != load_dataset(tmp_path).fingerprint


def test_missing_relevance_document_is_rejected(tmp_path):
    folder = write_dataset(tmp_path)
    (folder / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nquery-2\tmissing\t1\n")
    with pytest.raises(ValueError, match="missing"):
        load_dataset(tmp_path)


def test_archive_cannot_escape_dataset_directory(tmp_path):
    with zipfile.ZipFile(tmp_path / "fiqa.zip", "w") as archive:
        archive.writestr("../outside.txt", "no")
    with pytest.raises(ValueError, match="archive"):
        load_dataset(tmp_path)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_sign_bits_cross_word_boundaries_and_leave_padding_zero():
    vectors = np.full((2, 67), -1.0, dtype=np.float32)
    vectors[0, [0, 63, 64, 66]] = 1
    vectors[1, 65] = 0
    codes = encode_signs(vectors)
    assert codes.dtype == np.uint64
    assert codes.flags.c_contiguous
    assert codes.tolist() == [[(1 << 63) + 1, 5], [0, 2]]
    with pytest.raises(ValueError, match="finite"):
        encode_signs(np.array([[np.nan]], dtype=np.float32))


@pytest.fixture
def fake_model(monkeypatch):
    # Record calls so cache checks can tell whether encoding happened again.
    calls = []
    monkeypatch.setattr(
        embeddings, "_resolve_revision", lambda name, revision: revision or "a" * 40
    )
    monkeypatch.setattr(embeddings, "_choose_device", lambda device: "cpu")

    def encode_texts(document_texts, query_texts, **kwargs):
        calls.append((document_texts, query_texts, kwargs))
        rng = np.random.default_rng(17)
        documents = rng.normal(size=(len(document_texts), 768)).astype(np.float32)
        queries = rng.normal(size=(len(query_texts), 768)).astype(np.float32)
        return documents, queries

    monkeypatch.setattr(embeddings, "_encode_texts", encode_texts)
    return calls


def test_embeddings_use_prefixes_and_reuse_only_matching_cache(tmp_path, fake_model):
    write_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    first = prepare_embeddings(dataset, tmp_path / "vectors")
    second = prepare_embeddings(dataset, tmp_path / "vectors")
    assert len(fake_model) == 1
    assert fake_model[0][0][0] == "search_document: B\nsecond document"
    assert fake_model[0][1][0] == "search_query: question nine"
    assert first.documents.shape == (3, 256)
    assert first.full_documents.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(first.documents, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(first.full_documents, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(first.full_documents.mean(axis=1), 0, atol=1e-7)
    np.testing.assert_array_equal(first.codes, encode_signs(first.documents))
    np.testing.assert_array_equal(first.documents, second.documents)
    expected = first.full_documents[:, :256].copy()
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(first.documents, expected, atol=1e-6)
    prepare_embeddings(dataset, tmp_path / "vectors", model_revision="b" * 40)
    prepare_embeddings(dataset, tmp_path / "vectors", max_length=256)
    assert len(fake_model) == 3


def test_dimension_change_reuses_full_vectors_without_encoding_again(tmp_path, fake_model):
    write_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    cache_dir = tmp_path / "vectors"
    original = prepare_embeddings(dataset, cache_dir, dimensions=256)
    shorter = prepare_embeddings(dataset, cache_dir, dimensions=128)
    assert len(fake_model) == 1
    assert shorter.metadata["dimensions"] == 128
    for name, full_name in [
        ("documents", "full_documents"),
        ("queries", "full_queries"),
    ]:
        full = getattr(original, full_name)
        expected = full[:, :128].copy()
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_array_equal(getattr(shorter, name), expected)
        np.testing.assert_array_equal(getattr(shorter, full_name), full)
    np.testing.assert_array_equal(shorter.codes, encode_signs(shorter.documents))
    assert len(list(cache_dir.glob("*.npz"))) == 2
    cached = prepare_embeddings(dataset, cache_dir, dimensions=128)
    np.testing.assert_array_equal(cached.documents, shorter.documents)
    assert len(fake_model) == 1


@pytest.mark.parametrize("dimensions", [256, 128])
def test_corrupt_cached_embeddings_fail_instead_of_becoming_results(
    tmp_path, fake_model, dimensions
):
    write_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    cache_dir = tmp_path / "vectors"
    prepare_embeddings(dataset, cache_dir)
    cache_file = next(cache_dir.glob("*.npz"))
    with np.load(cache_file, allow_pickle=False) as cached:
        payload = dict(cached)
    payload["documents"][0, 0] = np.nan
    np.savez(cache_file, **payload)
    with pytest.raises(ValueError, match="cache"):
        prepare_embeddings(dataset, cache_dir, dimensions=dimensions)
    assert len(fake_model) == 1
