"""Compare sorted-prefix analysis with direct scalar bit comparisons."""

import csv
import importlib
import json

import bitplane_index
import numpy as np

from experiments.components import pack_signs, reference_rows
from experiments.prefix_batch_worker_v2 import file_hash


def test_all_depths_and_reference_sizes_match_scalar_prefix_membership(tmp_path):
    geometry = importlib.import_module("experiments.prefix_geometry_v2")
    signs = np.random.default_rng(42).integers(0, 2, (101, 32), dtype=np.uint8)
    signs[0] = 1
    signs[1] = 0
    queries = np.array([[1] * 32, [-1] * 32, [0] * 32, [.1, -2] * 16], dtype=np.float32)
    truth = reference_rows(signs, queries, 100)
    pool = tmp_path / "pool"; pool.mkdir()
    for name, values in dict(codes=pack_signs(signs), queries=queries, reference=truth,
                              documents=2 * signs.astype(np.float32) - 1).items():
        np.save(pool / f"{name}.npy", values)
    hashes = {name: file_hash(pool / name) for name in
              ("codes.npy", "queries.npy", "reference.npy", "documents.npy")}
    (pool / "pool.json").write_text(json.dumps(dict(
        identity=dict(documents=101, dimensions=32, queries=4, top_k=100), hashes=hashes)))
    output = tmp_path / "out"
    report = geometry.analyze(pool, output)
    assert report["depths"] == [1, 2, 4, 8, 16, 32]
    assert report["top_ks"] == [1, 10, 100]
    assert len(report["queries"]) == 4 * 6 * 3
    for row in report["queries"]:
        depth, qi, k = row["depth"], row["query"], row["top_k"]
        matches = [i for i, document in enumerate(signs)
                   if all(bool(document[bit]) == bool(queries[qi, bit] >= 0)
                          for bit in range(depth))]
        hits = sum(int(reference) in matches for reference in truth[qi, :k])
        assert row["matched_rows"] == len(matches)
        assert row["reference_matches"] == hits
        assert row["neighbor_survival"] == hits / k
        assert row["balanced_rows"] == 101 / 2 ** depth
    for row in report["summary"]:
        own = [value for value in report["queries"]
               if (value["depth"], value["top_k"]) == (row["depth"], row["top_k"])]
        counts = [value["matched_rows"] for value in own]
        assert row["mean_matched_rows"] == np.mean(counts)
        assert row["median_matched_rows"] == np.median(counts)
        assert row["p95_matched_rows"] == np.percentile(counts, 95)
        assert row["neighbor_survival"] == sum(value["reference_matches"] for value in own) / (4 * row["top_k"])
        k = row["top_k"]
        rates = [sum(bool(signs[truth[qi, rank], bit]) == bool(queries[qi, bit] >= 0)
                     for qi in range(4) for rank in range(k)) / (4 * k)
                 for bit in range(row["depth"])]
        assert row["independence_survival"] == np.prod(rates)
        assert row["independence_error"] == row["independence_survival"] - row["neighbor_survival"]
    assert json.loads((output / "geometry.json").read_text()) == report
    with (output / "queries.csv").open() as source:
        assert len(list(csv.DictReader(source))) == len(report["queries"])
    assert all(file_hash(pool / name) == digest for name, digest in hashes.items())


def test_logical_storage_audit_uses_actual_cluster_padding():
    geometry = importlib.import_module("experiments.prefix_geometry_v2")
    codes = np.zeros((65, 1), dtype=np.uint64)
    assignments = np.array([0, 1] + [2] * 63, dtype=np.int64)
    for method in ("scan", "branch", "prefix"):
        index = (bitplane_index.PrefixIndexV2(codes, assignments, 5, max_prefix_bits=5)
                 if method == "prefix" else
                 bitplane_index.Index(codes, assignments, 5, build_bitplanes=method == "branch"))
        audit = geometry.storage_audit(method, index.info(), [1, 1, 63])
        assert audit["matches"] is True
        assert audit["expected"]["logical_bytes"] == 65 * 16 + {"scan": 0, "branch": 120, "prefix": 260}[method]


def test_optimistic_bounds_use_full_scores_strict_ties_and_all_queries(tmp_path):
    geometry = importlib.import_module("experiments.prefix_geometry_v2")
    signs = np.zeros((100, 65), dtype=np.uint8)
    signs[0] = 1
    queries = np.array([[2, 1] + [0] * 62 + [1],
                        [-2, 1] + [0] * 62 + [1], [0] * 65], dtype=np.float32)
    truth = reference_rows(signs, queries, 100)
    pool = tmp_path / "pool"; pool.mkdir()
    for name, values in dict(codes=pack_signs(signs), queries=queries, reference=truth,
                              documents=2 * signs.astype(np.float32) - 1).items():
        np.save(pool / f"{name}.npy", values)
    (pool / "pool.json").write_text(json.dumps(dict(
        identity=dict(documents=100, dimensions=65, queries=3, top_k=100),
        hashes={name: file_hash(pool / name) for name in
                ("codes.npy", "queries.npy", "reference.npy", "documents.npy")})))
    output = tmp_path / "bounds"
    report = geometry.analyze_bounds(pool, output, chunk_size=1)
    assert report["top_ks"] == [1, 10, 100] and len(report["queries"]) == 9
    for row in report["queries"]:
        qi, k = row["query"], row["top_k"]
        query = queries[qi]
        score = sum(float(query[bit]) * (1 if signs[truth[qi, k - 1], bit] else -1)
                    for bit in range(65))
        q = sum(abs(float(value)) for value in query)
        penalty, expected = 0, 66
        for depth, weight in enumerate(sorted(map(lambda value: abs(float(value)), query), reverse=True), 1):
            penalty += weight
            if q - 2 * penalty < score:
                expected = depth
                break
        assert row["query_l1"] == q and row["kth_score"] == score
        assert row["gap"] == q - score and row["optimistic_min_depth"] == expected
    by_pair = {(row["query"], row["top_k"]): row for row in report["queries"]}
    assert by_pair[0, 1]["optimistic_min_depth"] == 1
    assert by_pair[0, 100]["optimistic_min_depth"] == 66  # Equality at full depth cannot prune.
    assert by_pair[1, 1]["optimistic_min_depth"] == 2  # At depth one the bound equals the score.
    assert by_pair[2, 1]["optimistic_min_depth"] == 66  # Every zero-weight bound is tied.
    for summary in report["summary"]:
        own = [row for row in report["queries"] if row["top_k"] == summary["top_k"]]
        for name in ("query_l1", "kth_score", "gap", "optimistic_min_depth"):
            values = [row[name] for row in own]
            assert summary[f"median_{name}"] == np.median(values)
            assert summary[f"p95_{name}"] == np.percentile(values, 95)
    assert json.loads((output / "bounds.json").read_text()) == report
    with (output / "queries.csv").open() as source:
        assert len(list(csv.DictReader(source))) == 9
