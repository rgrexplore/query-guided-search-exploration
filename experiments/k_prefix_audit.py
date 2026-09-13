"""Recheck selected results from their returned IDs, outside the timing experiment."""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import bitplane_index
import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.fixed_data_comparison import save_json
from experiments.k_prefix_study import OUTPUT, POOL
from experiments.prefix_batch_worker_v2 import file_hash, prepare_router, search_index


def audit():
    rows = json.loads((OUTPUT / "comparison.json").read_text())
    choices = {(row["study"], row["setting_id"]): row for row in rows if row["status"] == "complete"}
    expected_hits = {}
    manifest = None
    for study in sorted({row["study"] for row in choices.values()}):
        current = json.loads((Path(study)/"inputs-manifest.json").read_text())
        if manifest is None:
            manifest = current
        assert current == manifest
        for path in Path(study).glob("repeat/cases/*/run/queries.jsonl"):
            with path.open() as source:
                for line in source:
                    record = json.loads(line)
                    key = (study, record["setting_id"])
                    if key not in choices:
                        continue
                    query_key = (*key, record["query"])
                    hits = round(record["recall"] * choices[key]["top_k"])
                    if query_key in expected_hits:
                        assert expected_hits[query_key] == hits
                    expected_hits[query_key] = hits
    assert all(file_hash(POOL/name)==digest for name,digest in manifest["array_hashes"].items())
    groups = defaultdict(list)
    for row in choices.values():
        groups[(row["method"], row["router"])].append(row)
    codes = np.load(POOL / "codes.npy", mmap_mode="r")
    queries = np.load(POOL / "queries.npy")
    reference = np.load(POOL / "reference.npy")
    dimensions = queries.shape[1]
    coordinates = np.arange(dimensions, dtype=np.uint64)
    summaries = []
    output = OUTPUT / "selected-returned-ids.jsonl.gz"
    with gzip.open(output, "wt") as stream:
        for (method, router), settings in groups.items():
            assignments, route, _, _ = prepare_router(router, len(codes))
            index = (bitplane_index.PrefixIndexV2(codes, assignments, dimensions, max_prefix_bits=32)
                     if method == "prefix" else bitplane_index.Index(
                         codes, assignments, dimensions, build_bitplanes=method == "branch"))
            for setting in settings:
                k = setting["top_k"]
                hits = scored = 0
                for qi, query in enumerate(queries):
                    selected = route(query[None,:], setting["probes"])
                    result = search_index(index, method, query[None,:], selected, setting)
                    returned = result["rows"][0]
                    valid = returned >= 0
                    ids = returned[valid]
                    bits = (codes[ids[:,None], coordinates//64] >> (coordinates%64)) & 1
                    signs = np.where(bits, 1.0, -1.0)
                    direct_scores = np.einsum("ij,j->i", signs, query.astype(np.float64))
                    np.testing.assert_allclose(result["scores"][0][valid], direct_scores,
                                               atol=1e-12, rtol=1e-12)
                    query_hits = len(set(ids) & set(reference[qi,:k]))
                    assert query_hits == expected_hits[(setting["study"],setting["setting_id"],qi)]
                    hits += query_hits
                    scored += result["stats"][0]["documents_scored"]
                    stream.write(json.dumps(dict(study=setting["study"],
                        setting_id=setting["setting_id"], query=qi, top_k=k,
                        rows=ids.tolist(), correct_ids=query_hits), allow_nan=False) + "\n")
                actual_recall = hits/(len(queries)*k)
                assert actual_recall == setting["recall"], (setting["setting_id"], actual_recall, setting["recall"])
                summaries.append(dict(study=setting["study"], setting_id=setting["setting_id"],
                    method=method, top_k=k, queries=len(queries), recall=actual_recall,
                    mean_documents_scored=scored/len(queries), scores_checked=True))
            print(f"Audited {method}: {len(settings)} selected settings", flush=True)
    save_json(OUTPUT / "audit.json", dict(selected_settings=len(summaries),
        query_checks=len(summaries)*len(queries), returned_scores_match_direct_calculation=True,
        recall_matches_saved_measurements=True, input_hashes_unchanged=True,
        native_sha256=file_hash(bitplane_index.__file__),
        returned_ids_file=output.name, returned_ids_sha256=file_hash(output), settings=summaries))


if __name__ == "__main__":
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        audit()
