"""Check binary-reference score agreement with a supplied full-width float pool.

Run only after timing jobs finish. This is an exhaustive score diagnostic, not a
latency benchmark. No new embeddings or approximate search algorithms are used.
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.analysis import write_csv
from experiments.fixed_data_comparison import save_json
from experiments.prefix_study_v2 import verify_pool

SCORE_TOLERANCE = 1e-5


def full_float_best(documents, queries, *, threads=8, batch_size=64):
    """Return one best full-float score and row ID per query using exhaustive IP."""
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(np.ascontiguousarray(documents, dtype=np.float32))
    scores = np.empty(len(queries), dtype=np.float32)
    rows = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), batch_size):
        stop = min(len(queries), start + batch_size)
        found_scores, found_rows = index.search(np.ascontiguousarray(queries[start:stop], dtype=np.float32), 1)
        scores[start:stop], rows[start:stop] = found_scores[:, 0], found_rows[:, 0]
    return scores, rows


def candidate_agreement(documents, queries, references, best_scores, best_rows, *, batch_size=64):
    """Rescore the saved candidate rows without changing their binary ranking."""
    if (references.ndim != 2 or len(references) != len(queries) or references.shape[1] == 0
            or references.min() < 0 or references.max() >= len(documents)):
        raise ValueError("Reference rows must identify existing documents for every query")
    records = []
    for start in range(0, len(queries), batch_size):
        stop = min(len(queries), start + batch_size)
        candidates = references[start:stop]
        scores = np.einsum("qkd,qd->qk", documents[candidates], queries[start:stop],
                           dtype=np.float32, optimize=True)
        gaps = best_scores[start:stop].astype(np.float64)[:, None] - scores.astype(np.float64)
        matching = np.abs(gaps) <= SCORE_TOLERANCE
        best_positions = scores.argmax(axis=1)
        for local, qi in enumerate(range(start, stop)):
            position = best_positions[local]
            records.append(dict(query=qi, full_float_best_row=int(best_rows[qi]),
                full_float_best_score=float(best_scores[qi]),
                binary_top1_row=int(candidates[local, 0]), binary_top1_full_score=float(scores[local, 0]),
                top1_full_score_gap=float(gaps[local, 0]), top1_agrees=bool(matching[local, 0]),
                candidate_count=references.shape[1], candidate_best_row=int(candidates[local, position]),
                candidate_best_full_score=float(scores[local, position]),
                candidate_best_full_score_gap=float(gaps[local, position]),
                candidates_agree=bool(matching[local].any())))
    return records


def check_alignment(full, shorter):
    left, right = full["identity"], shorter["identity"]
    required = ("documents", "queries", "full_documents_sha256", "full_queries_sha256")
    if any(name not in left or name not in right or left[name] != right[name] for name in required):
        raise ValueError("Pools must record matching counts and the same original full-array hashes")
    if right["dimensions"] > left["dimensions"]:
        raise ValueError("Comparison dimensions exceed the supplied full-width pool")
    if left.get("provenance") != right.get("provenance"):
        raise ValueError("Pools record different embedding/source provenance")
    left_ids, right_ids = left.get("query_ids"), right.get("query_ids")
    for identity, ids in ((left, left_ids), (right, right_ids)):
        if ids is not None and len(ids) != identity["queries"]:
            raise ValueError("Recorded query IDs do not match the query count")
    if left_ids is not None and right_ids is not None and left_ids != right_ids:
        raise ValueError("Pools record different query IDs or query order")
    return dict(pool=shorter["pool"], dimensions=right["dimensions"],
                matching_full_document_source=left["full_documents_sha256"],
                matching_full_query_source=left["full_queries_sha256"],
                ordered_query_ids_verified=left_ids is not None and right_ids is not None,
                provenance_available=left.get("provenance") is not None,
                document_order_basis="Recorded common full-document-array hash and source provenance; no fabricated corpus IDs.")


def analyze(full_pool, pools, output, *, threads=8, batch_size=64):
    full_pool, output = Path(full_pool).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Use a new representation-analysis output directory")
    full = verify_pool(full_pool)
    paths = list(dict.fromkeys(Path(pool).resolve() for pool in pools))
    inputs = [full if path == full_pool else verify_pool(path) for path in paths]
    alignment = [check_alignment(full, item) for item in inputs]
    if any(item["identity"]["top_k"] < 100 for item in inputs):
        raise ValueError("Every compared pool must contain the complete binary top100 references")
    documents = np.load(full_pool / "documents.npy", mmap_mode="r")
    queries = np.load(full_pool / "queries.npy", mmap_mode="r")
    query_ids = full["identity"].get("query_ids")
    output.mkdir(parents=True)
    summaries, all_rows = [], []
    with threadpool_limits(limits=threads):
        best_scores, best_rows = full_float_best(documents, queries, threads=threads, batch_size=batch_size)
        best = [dict(query=qi, full_float_best_row=int(best_rows[qi]),
                     full_float_best_score=float(best_scores[qi]),
                     **({"query_id": query_ids[qi]} if query_ids is not None else {}))
                for qi in range(len(queries))]
        save_json(output / "full-float-best.json", best)
        write_csv(output / "full-float-best.csv", best)
        for path, item in zip(paths, inputs, strict=True):
            references = np.load(path / "reference.npy", mmap_mode="r")[:, :100]
            rows = candidate_agreement(documents, queries, references, best_scores, best_rows,
                                       batch_size=batch_size)
            for row in rows:
                row.update(pool=str(path), binary_dimensions=item["identity"]["dimensions"])
                if query_ids is not None:
                    row["query_id"] = query_ids[row["query"]]
            gaps = [row["top1_full_score_gap"] for row in rows]
            summaries.append(dict(pool=str(path), binary_dimensions=item["identity"]["dimensions"],
                queries=len(rows), top1_score_agreement_count=sum(row["top1_agrees"] for row in rows),
                top1_score_agreement_rate=float(np.mean([row["top1_agrees"] for row in rows])),
                top100_score_agreement_count=sum(row["candidates_agree"] for row in rows),
                top100_score_agreement_rate=float(np.mean([row["candidates_agree"] for row in rows])),
                mean_top1_full_score_gap=float(np.mean(gaps)), median_top1_full_score_gap=float(np.median(gaps)),
                p95_top1_full_score_gap=float(np.percentile(gaps, 95)),
                min_top1_full_score_gap=float(np.min(gaps)), max_top1_full_score_gap=float(np.max(gaps))))
            all_rows.extend(rows)
    report = dict(full_float_inputs=full, binary_inputs=inputs, alignment_checks=alignment,
        tolerance=SCORE_TOLERANCE, agreement_rule="abs(full_float_best_score - candidate_full_float_score) <= 1e-5",
        threads=threads, query_batch_size=batch_size, summary=summaries, queries=all_rows,
        method="Faiss IndexFlatIP over every supplied full-width float document; all cached queries in batches. Rescore the first100 exact binary-reference rows with the same full float vectors. Retain signed raw score gaps; do not clamp numerical differences.",
        setup_contract="The first pool is explicitly supplied as the normalized full-width float reference. All compared widths come from the identical recorded full caches and preserve row order. Match counts, original full-array hashes, available ordered query IDs and provenance; verify each pool's saved array hashes. Document IDs in these outputs are zero-based rows in that common source order.",
        scope="Score-agreement diagnostics, not relevance judgments and not measured A/B/C end-to-end quality. The top100 result assumes exact binary-reference candidates and full-float rescoring; approximate B/C searches may fail to retrieve those candidates. Float32 arithmetic and ties are handled with one fixed score tolerance. No latency results are reported.")
    save_json(output / "representation.json", report)
    write_csv(output / "queries.csv", all_rows)
    write_csv(output / "summary.csv", summaries)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pool", type=Path, required=True)
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    print(analyze(args.full_pool, args.pools, args.output,
                  threads=args.threads, batch_size=args.batch_size)["summary"])


if __name__ == "__main__":
    main()
