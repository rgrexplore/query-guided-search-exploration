"""Measure original-sign-prefix occupancy and exact-neighbor survival on a finished pool."""

import argparse
from pathlib import Path

import numpy as np

from experiments.analysis import write_csv
from experiments.fixed_data_comparison import save_json
from experiments.prefix_study_v2 import verify_pool

DEPTHS = (1, 2, 4, 8, 16, 32)


def document_keys(codes, width):
    """Coordinate zero is the packed word's low bit but the prefix key's high bit."""
    keys = np.zeros(len(codes), dtype=np.uint32)
    for bit in range(width):
        keys = (keys << np.uint32(1)) | ((codes[:, 0] >> np.uint64(bit)) & 1).astype(np.uint32)
    return keys


def analyze(pool, output):
    pool, output = Path(pool).resolve(), Path(output)
    manifest = verify_pool(pool)
    output.mkdir(parents=True, exist_ok=False)
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    queries = np.load(pool / "queries.npy", mmap_mode="r")
    reference = np.load(pool / "reference.npy", mmap_mode="r")
    width = min(32, manifest["identity"]["dimensions"])
    keys = document_keys(codes, width)
    # Sort once. The search buffer uses uint64 so the excluded endpoint 2**32 fits.
    sorted_keys = np.sort(keys).astype(np.uint64)
    reference_keys = keys[reference].astype(np.uint64)
    query_keys = np.zeros(len(queries), dtype=np.uint32)
    for bit in range(width):
        query_keys = (query_keys << np.uint32(1)) | (queries[:, bit] >= 0).astype(np.uint32)
    depths = [depth for depth in DEPTHS if depth <= width]
    top_ks = [k for k in (1, 10, 100) if k <= reference.shape[1]]
    agreement_rates = {k: [] for k in top_ks}
    for bit in range(width):
        shift = width - bit - 1
        agrees = ((reference_keys >> shift) & 1) == ((query_keys.astype(np.uint64) >> shift) & 1)[:, None]
        for k in top_ks:
            agreement_rates[k].append(float(np.mean(agrees[:, :k])))
    independence_products = {k: np.cumprod(rates) for k, rates in agreement_rates.items()}
    records, summary = [], []
    for depth in depths:
        shift = width - depth
        prefixes = query_keys.astype(np.uint64) >> shift
        low, high = prefixes << shift, (prefixes + np.uint64(1)) << shift
        left = np.searchsorted(sorted_keys, low, side="left")
        right = np.searchsorted(sorted_keys, high, side="left")
        counts = right - left
        matching_references = (reference_keys >> shift) == prefixes[:, None]
        balanced = len(codes) / 2 ** depth
        for k in top_ks:
            hits = matching_references[:, :k].sum(axis=1)
            for query in range(len(queries)):
                records.append(dict(query=query, depth=depth, top_k=k,
                    matched_rows=int(counts[query]), balanced_rows=balanced,
                    reference_matches=int(hits[query]), neighbor_survival=int(hits[query]) / k))
            summary.append(dict(depth=depth, top_k=k, queries=len(queries),
                mean_matched_rows=float(np.mean(counts)), median_matched_rows=float(np.median(counts)),
                p95_matched_rows=float(np.percentile(counts, 95)), balanced_rows=balanced,
                reference_matches=int(hits.sum()), neighbor_survival=int(hits.sum()) / (len(queries) * k),
                independence_survival=float(independence_products[k][depth - 1]),
                independence_error=float(independence_products[k][depth - 1]) - int(hits.sum()) / (len(queries) * k)))
    report = dict(inputs=manifest, depths=depths, top_ks=top_ks, summary=summary, queries=records,
        position_agreement_rates={str(k): rates for k, rates in agreement_rates.items()},
        scope="All cached queries and all documents, in their original coordinate order. "
              "Neighbor survival at depth d is a recall ceiling if every query finishes at depth d "
              "or deeper; routing can reduce recall further. It does not bound walks that widen "
              "below d. Balanced rows is N/2**d under independent equally likely signs, not recall. "
              "Independence survival multiplies per-position agreement rates estimated on these same "
              "reference pairs; it is not a future-query guarantee. Error is product minus measured survival.")
    save_json(output / "geometry.json", report)
    write_csv(output / "queries.csv", records)
    write_csv(output / "summary.csv", summary)
    return report


def storage_audit(method, info, cluster_sizes):
    """Compare native logical fields with exact word padding in each cluster."""
    documents, dimensions = info["documents"], info["dimensions"]
    expected = dict(codes_bytes=8 * documents * ((dimensions + 63) // 64),
                    row_ids_bytes=8 * documents,
                    bitplanes_bytes=8 * dimensions * sum((size + 63) // 64 for size in cluster_sizes)
                    if method == "branch" else 0,
                    prefix_keys_bytes=4 * documents if method == "prefix" else 0)
    expected["logical_bytes"] = sum(expected.values())
    reported = {name: info[name] for name in expected}
    return dict(method=method, expected=expected, reported=reported, matches=expected == reported,
                scope="Native logical fields only; router, unused capacity and process RAM are separate.")


def analyze_bounds(pool, output, chunk_size=32):
    """Find the earliest possible strict score-bound pruning depth for each query."""
    pool, output = Path(pool).resolve(), Path(output)
    manifest = verify_pool(pool)
    output.mkdir(parents=True, exist_ok=False)
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    queries = np.load(pool / "queries.npy", mmap_mode="r")
    reference = np.load(pool / "reference.npy", mmap_mode="r")
    dimensions = queries.shape[1]
    top_ks = [k for k in (1, 10, 100) if k <= reference.shape[1]]
    positions = np.arange(dimensions, dtype=np.uint64)
    records = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start:start + chunk_size].astype(np.float64)
        kth_rows = reference[start:start + len(query)][:, np.array(top_ks) - 1]
        packed = codes[kth_rows]
        signs = ((packed[..., positions // 64] >> (positions % 64)) & 1).astype(bool)
        # Full stored dimensions, including the final partly used packed word.
        kth_scores = np.where(signs, query[:, None, :], -query[:, None, :]).sum(axis=2)
        absolute = np.abs(query)
        query_l1 = absolute.sum(axis=1)
        bounds = query_l1[:, None] - 2 * np.cumsum(np.sort(absolute, axis=1)[:, ::-1], axis=1)
        for column, k in enumerate(top_ks):
            strict = bounds < kth_scores[:, column, None]
            depths = np.where(strict.any(axis=1), strict.argmax(axis=1) + 1, dimensions + 1)
            for local in range(len(query)):
                records.append(dict(query=start + local, top_k=k, query_l1=float(query_l1[local]),
                    kth_score=float(kth_scores[local, column]),
                    gap=float(query_l1[local] - kth_scores[local, column]),
                    optimistic_min_depth=int(depths[local])))
    summary = []
    for k in top_ks:
        own = [row for row in records if row["top_k"] == k]
        row = dict(top_k=k, queries=len(own),
                   no_strict_prune_queries=sum(value["optimistic_min_depth"] == dimensions + 1 for value in own))
        for name in ("query_l1", "kth_score", "gap", "optimistic_min_depth"):
            values = [value[name] for value in own]
            row[f"median_{name}"] = float(np.median(values))
            row[f"p95_{name}"] = float(np.percentile(values, 95))
        summary.append(row)
    report = dict(inputs=manifest, top_ks=top_ks, queries=records, summary=summary,
        no_strict_prune_depth=dimensions + 1,
        scope="Q is sum(abs(query)); gap is Q minus the final exact Kth full score. "
              "The minimum depth is the first j with Q - 2*cumsum(descending weights)[j-1] < Kth score. "
              "It assumes the final score is already known and every tested sign mismatches the query, "
              "giving the largest possible penalty at each depth. This is an optimistic lower bound, "
              "not a prediction of visited nodes. Equality does not prune. The native rounding guard "
              "can postpone pruning further; scores here are direct float64 sums of stored query/sign products.")
    save_json(output / "bounds.json", report)
    write_csv(output / "queries.csv", records)
    write_csv(output / "summary.csv", summary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.pool, args.output)


if __name__ == "__main__":
    main()
