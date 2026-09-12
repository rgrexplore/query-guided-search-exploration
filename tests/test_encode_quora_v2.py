import json

import numpy as np
import pytest

from experiments.encode_quora_v2 import encode_chunks, prepare_sources


def test_interrupted_encoding_resumes_saved_chunks_in_original_order(tmp_path):
    texts = ["1", "2", "3", "4", "5"]
    identity = {"model": "fixture", "dimensions": 2, "batch_size": 2}
    calls = []

    def interrupted(values):
        calls.append(values)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return np.array([[float(x), 1] for x in values], dtype=np.float32)

    with pytest.raises(RuntimeError, match="interrupted"):
        encode_chunks(texts, tmp_path, "documents", identity, interrupted, chunk_size=2)
    first = (tmp_path / "chunks/documents-000000000-000000002.npy").read_bytes()
    resumed_calls = []

    def resumed(values):
        resumed_calls.append(values)
        return np.array([[float(x), 1] for x in values], dtype=np.float32)

    path, _ = encode_chunks(texts, tmp_path, "documents", identity, resumed, chunk_size=2)
    assert resumed_calls == [["3", "4"], ["5"]]
    assert first == (tmp_path / "chunks/documents-000000000-000000002.npy").read_bytes()
    expected = np.array([[float(x), 1] for x in texts], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_array_equal(np.load(path), expected)
    complete_calls = []
    encode_chunks(texts, tmp_path, "documents", identity,
                  lambda values: complete_calls.append(values), chunk_size=2)
    assert complete_calls == []


@pytest.mark.parametrize("change", ["text", "recipe", "chunk_size"])
def test_changed_inputs_cannot_reuse_completed_chunks(tmp_path, change):
    identity = {"model": "fixture", "dimensions": 2, "batch_size": 2}
    encoder = lambda values: np.ones((len(values), 2), dtype=np.float32)
    encode_chunks(["first", "second"], tmp_path, "documents", identity, encoder, chunk_size=2)
    texts = ["changed", "second"] if change == "text" else ["first", "second"]
    requested = dict(identity, model="other") if change == "recipe" else identity
    chunk_size = 1 if change == "chunk_size" else 2
    with pytest.raises(ValueError, match="different inputs or recipe"):
        encode_chunks(texts, tmp_path, "documents", requested, encoder, chunk_size=chunk_size)


def test_invalid_encoder_output_is_not_saved_as_a_finished_chunk(tmp_path):
    identity = {"model": "fixture", "dimensions": 2}
    with pytest.raises(ValueError, match="nonzero finite"):
        encode_chunks(["text"], tmp_path, "queries", identity,
                      lambda values: np.zeros((1, 2), dtype=np.float32), chunk_size=2)
    assert not list((tmp_path / "chunks").glob("queries-*.npy"))


def test_source_preparation_preserves_order_ids_labels_and_matching_text(tmp_path):
    source = tmp_path / "source"
    (source / "qrels").mkdir(parents=True)
    corpus = [{"_id": "d2", "title": "", "text": "same"},
              {"_id": "d1", "title": "", "text": "other"}]
    (source / "corpus.jsonl").write_text("".join(json.dumps(x) + "\n" for x in corpus))
    (source / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "same"}) + "\n")
    (source / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td2\t1\n")
    chosen = tmp_path / "chosen.jsonl"
    chosen.write_text(json.dumps({"id": "q1", "text": "same"}) + "\n")
    result = prepare_sources(source, chosen)
    assert result["corpus_ids"] == ["d2", "d1"]
    assert result["document_texts"] == ["same", "other"]
    assert result["query_ids"] == ["q1"]
    assert result["query_texts"] == ["same"]
    assert result["overlap"]["exact_text_query_corpus_pairs"] == 1
    assert result["qrels"] == [{"query-id": "q1", "corpus-id": "d2", "score": "1"}]
