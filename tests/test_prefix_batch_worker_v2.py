"""Small real native jobs verify index reuse and the saved measurement contract."""

import hashlib
import importlib
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.components import pack_signs, reference_rows


@pytest.fixture
def job(tmp_path):
    signs = np.array([[1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    queries = np.array([[.6, .5, .4, .3], [0, 0, 0, 0]], dtype=np.float32)
    pool = tmp_path / "pool"
    pool.mkdir()
    np.save(pool / "codes.npy", pack_signs(signs))
    np.save(pool / "queries.npy", queries)
    np.save(pool / "reference.npy", reference_rows(signs, queries, 2))
    return dict(pool=str(pool), router=None, method="prefix", dimensions=4,
                max_prefix_bits=4, query_rows=[0, 1], ram_budget_bytes=32_000_000_000,
                variants=[dict(setting_id="one", top_k=1, probes=1, node_budget=0,
                               leaf_size=1, exploration=0, seed=42, start_depth=4,
                               candidate_target=0, repetitions=1),
                          dict(setting_id="two", top_k=2, probes=1, node_budget=0,
                               leaf_size=4, exploration=0, seed=42, start_depth=4,
                               candidate_target=0, repetitions=2)])


def read_records(output):
    return [json.loads(line) for line in (output / "queries.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("method", ["scan", "branch", "prefix"])
def test_multiple_variants_reuse_one_real_index_and_preserve_queries(job, tmp_path, monkeypatch,
                                                                     method):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    job["method"] = method
    constructor = "PrefixIndexV2" if method == "prefix" else "Index"
    native_type = getattr(worker.bitplane_index, constructor)
    builds, seen_queries = [], []

    class ObservedIndex:
        def __init__(self, *args, **kwargs):
            builds.append((args, kwargs))
            self.index = native_type(*args, **kwargs)

        def info(self):
            return self.index.info()

        def scan(self, query, *args, **kwargs):
            seen_queries.append(query.copy())
            return self.index.scan(query, *args, **kwargs)

        def search(self, query, *args, **kwargs):
            seen_queries.append(query.copy())
            return self.index.search(query, *args, **kwargs)

    monkeypatch.setattr(worker.bitplane_index, constructor, ObservedIndex)
    source = tmp_path / "pool" / "queries.npy"
    original = source.read_bytes()
    output = tmp_path / "out"
    result = worker.run_job(job, output)
    records = read_records(output)
    assert len(builds) == 1
    assert len(seen_queries) == 8  # Six measurements and one warmup per variant.
    queries = np.load(source)
    np.testing.assert_array_equal(np.concatenate(seen_queries),
                                   queries[[0, 0, 1, 0, 0, 1, 0, 1]])
    assert source.read_bytes() == original
    assert result["status"] == "complete"
    assert result["completed_variants"] == ["one", "two"]
    assert result["measurements"] == result["exact_checks"] == len(records) == 6
    assert [(row["setting_id"], row["query"], row["repetition"]) for row in records] == [
        ("one", 0, 0), ("one", 1, 0), ("two", 0, 0), ("two", 1, 0),
        ("two", 0, 1), ("two", 1, 1)]
    assert [row["returned"] for row in records] == [1, 1, 2, 2, 2, 2]
    assert all(row["recall"] == 1 for row in records)
    assert all(row["query_ms"] >= row["search_ms"] >= row["elapsed_ms"] for row in records)
    assert all(row["routing_ms"] >= 0 for row in records)
    assert result["memory"]["rss_before_queries"] > 0
    assert result["memory"]["sampled_query_peak"] >= result["memory"]["rss_before_queries"]
    assert result["memory"]["budget_status"] == "fits_by_lifetime_peak"
    assert result["build_ms"] > 0 and len(result["native_sha256"]) == 64
    assert result["worker_sha256"] == hashlib.sha256(open(worker.__file__, "rb").read()).hexdigest()
    assert all(len(digest) == 64 for digest in result["source_hashes"].values())
    assert "power_before" in result and "power_after" in result
    assert json.loads((output / "result.json").read_text()) == result
    if method == "prefix":
        assert result["storage"]["prefix_keys_bytes"] == 12
        assert all(row["final_depth"] == 0 and row["documents_scored"] == 3 for row in records)


def test_positive_target_records_the_paper_miss_with_exact_control(job, tmp_path):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    job["variants"][0]["candidate_target"] = 1
    result = worker.run_job(job, tmp_path / "out")
    records = read_records(tmp_path / "out")
    assert records[0]["recall"] == 0
    assert records[0]["final_depth"] == 2 and records[0]["documents_scored"] == 1
    assert records[2]["recall"] == 1 and records[2]["documents_scored"] == 3
    assert result["exact_checks"] == 4


def test_centroid_routing_finishes_prefix_depth_across_selected_clusters(job, tmp_path):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    signs = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    queries = np.array([[.1, .2, 2]], dtype=np.float32)
    pool = tmp_path / "pool"
    np.save(pool / "codes.npy", pack_signs(signs))
    np.save(pool / "queries.npy", queries)
    np.save(pool / "reference.npy", reference_rows(signs, queries, 2))
    router = tmp_path / "router"
    router.mkdir()
    (router / "router.json").write_text(json.dumps({"kind": "ivf"}))
    np.save(router / "assignments.npy", np.array([8, 2, 8], dtype=np.int64))
    np.save(router / "labels.npy", np.array([8, 2], dtype=np.int64))
    np.save(router / "centroids.npy", np.array([[1, 1, -1], [-1, -1, 1]], dtype=np.float32))
    job.update(router=str(router), dimensions=3, max_prefix_bits=3, query_rows=[0])
    job["variants"] = [dict(job["variants"][0], top_k=2, probes=2,
                            start_depth=2, candidate_target=1),
                       dict(job["variants"][1], top_k=2, probes=1,
                            start_depth=2, candidate_target=0, repetitions=1)]
    result = worker.run_job(job, tmp_path / "out")
    records = read_records(tmp_path / "out")
    assert records[0]["recall"] == 1 and records[0]["documents_scored"] == 2
    assert records[0]["final_depth"] == 2 and records[0]["prefix_lookups"] == 4
    assert records[1]["recall"] == .5 and records[1]["documents_scored"] == 1
    assert result["exact_checks"] == 0  # Target zero alone does not make routing exact.
    assert result["memory"]["routing_label_count"] == 2
    assert result["memory"]["routing_payload_bytes"] == 40


def test_exact_prefix_root_rejects_wrong_reference_and_retains_failure(job, tmp_path):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    reference_path = tmp_path / "pool" / "reference.npy"
    truth = np.load(reference_path)
    truth[0, 0] = 0  # The actual best row is 1; the healthy test above must stay green.
    np.save(reference_path, truth)
    with pytest.raises(AssertionError):
        worker.run_job(job, tmp_path / "out")
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["status"] == "failed" and result["completed_variants"] == []
    assert result["error"]["type"] == "AssertionError"
    assert result["memory"]["sampled_query_peak"] >= result["memory"]["rss_before_queries"] > 0


def test_rejects_shortened_query_set_and_existing_output(job, tmp_path):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    job["query_rows"] = [0]
    with pytest.raises(ValueError, match="all cached queries"):
        worker.run_job(job, tmp_path / "out")
    with pytest.raises(FileExistsError):
        worker.run_job(job, tmp_path / "out")


def test_cli_runs_declared_variants_in_a_fresh_process(job, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    output = tmp_path / "out"
    run = subprocess.run([sys.executable, "-m", "experiments.prefix_batch_worker_v2",
                          str(path), str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["completed_variants"] == ["one", "two"]
    assert result["measurements"] == 6


def test_worker_forwards_optional_exact_stop_without_changing_its_default(job, tmp_path):
    worker = importlib.import_module("experiments.prefix_batch_worker_v2")
    base = job["variants"][0]
    job["variants"] = [dict(base, setting_id="default"),
                       dict(base, setting_id="bounded", stop_when_exact=True)]
    result = worker.run_job(job, tmp_path / "out")
    records = read_records(tmp_path / "out")
    assert result["completed_variants"] == ["default", "bounded"]
    assert result["exact_checks"] == 4
    assert all(row["recall"] == 1 for row in records)
    assert [row["documents_scored"] for row in records] == [3, 3, 2, 3]
    assert records[2]["stop_reason"] == "bound"
    assert records[3]["stop_reason"] == "exhausted"
