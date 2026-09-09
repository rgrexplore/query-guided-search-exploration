"""Choose coarse buckets before scanning or traversing their documents."""

from numbers import Integral
from time import perf_counter

import faiss
import numpy as np


def _matrix(values, name, dimensions=None, allow_empty=False):
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        raise TypeError(f"{name} must be a float32 NumPy array")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must have shape (rows, dimensions)")
    if not allow_empty and len(values) == 0:
        raise ValueError(f"{name} must contain at least one row")
    if dimensions is not None and values.shape[1] != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(values)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bucket_info(assignments, possible_buckets):
    """Count documents per occupied bucket; possible buckets may include empty ones."""
    _, counts = np.unique(assignments, return_counts=True)
    bucket_sizes = {
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "p95": float(np.percentile(counts, 95)),
        "max": int(counts.max()),
    }
    return {
        "possible_buckets": int(possible_buckets),
        "occupied_buckets": len(counts),
        "bucket_sizes": bucket_sizes,
    }


def _timed_search(index, queries, top_k):
    """One native call per query keeps the latency distribution meaningful."""
    rows = np.full((len(queries), top_k), -1, dtype=np.int64)
    scores = np.full((len(queries), top_k), -np.inf, dtype=np.float32)
    latencies = np.empty(len(queries), dtype=np.float64)
    for query_row in range(len(queries)):
        start = perf_counter()
        query_scores, query_rows = index.search(queries[query_row : query_row + 1], top_k)
        latencies[query_row] = (perf_counter() - start) * 1000
        rows[query_row] = query_rows[0]
        scores[query_row] = query_scores[0]
    return rows, scores, latencies


class Router:
    """A document-to-bucket map and a way to pick buckets for a query."""

    def select(self, queries, probes):
        raise NotImplementedError


class IVFRouter(Router):
    """Faiss owns centroid training, assignments, and the float IVF baseline."""

    def __init__(self, documents, clusters, seed, threads):
        self.dimensions = documents.shape[1]
        self.threads = threads

        # The quantizer stores centroids. IVF uses them to assign documents to lists.
        self.quantizer = faiss.IndexFlatIP(self.dimensions)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dimensions, clusters, faiss.METRIC_INNER_PRODUCT
        )
        self.index.cp.spherical = True
        self.index.cp.seed = seed
        self.index.cp.min_points_per_centroid = 1
        faiss.omp_set_num_threads(threads)
        self.index.train(documents)
        self.index.add(documents)

        # Give the C++ index the same assignments as the float IVF baseline.
        _, assignments = self.quantizer.search(documents, 1)
        self.assignments = np.ascontiguousarray(assignments[:, 0], dtype=np.int64)
        self.info = {
            "method": "ivf",
            "dimensions": self.dimensions,
            "clusters": clusters,
            "seed": seed,
            "threads": threads,
            "spherical_centroids": True,
            "routing": "nearest inner-product centroids, including empty lists",
            **_bucket_info(self.assignments, clusters),
        }

    def select(self, queries, probes):
        """Find the nearest centroids. These are bucket IDs, not document IDs."""
        queries = _matrix(queries, "queries", self.dimensions, allow_empty=True)
        probes = _positive_integer(probes, "probes")
        faiss.omp_set_num_threads(self.threads)
        rows, _, latencies = _timed_search(self.quantizer, queries, probes)
        return rows, latencies

    def ivf_search(self, queries, candidate_limit, probes):
        """Return document rows using Faiss's float scoring inside each list."""
        queries = _matrix(queries, "queries", self.dimensions, allow_empty=True)
        candidate_limit = _positive_integer(candidate_limit, "candidate_limit")
        requested_probes = _positive_integer(probes, "probes")
        self.index.nprobe = min(requested_probes, self.index.nlist)
        faiss.omp_set_num_threads(self.threads)
        return _timed_search(self.index, queries, candidate_limit)


class SignRouter(Router):
    """A prefix of document signs forms its bucket address."""

    def __init__(self, documents, routing_bits):
        self.dimensions = documents.shape[1]
        self.routing_bits = routing_bits

        # The first coordinate is the low bit. Zero has a positive sign.
        bit_positions = np.arange(routing_bits, dtype=np.uint32)
        powers = np.left_shift(np.uint32(1), bit_positions)
        signs = (documents[:, :routing_bits] >= 0).astype(np.uint32)
        self.assignments = np.ascontiguousarray(signs @ powers, dtype=np.int64)
        self.prefixes = np.unique(self.assignments)

        # Decode occupied addresses as -1/+1 vectors so each query can score them.
        bits = (self.prefixes[:, None].astype(np.uint32) & powers) != 0
        self.signed_codes = np.ascontiguousarray(2 * bits.astype(np.float32) - 1)
        self.info = {
            "method": "sign",
            "dimensions": self.dimensions,
            "routing_bits": routing_bits,
            "threads": 1,
            "routing": "scan occupied prefixes with float-query / binary-code scores",
            "zero_sign": "positive",
            **_bucket_info(self.assignments, 1 << routing_bits),
        }

    def select(self, queries, probes):
        """Score occupied sign buckets with the query's original coordinate magnitudes."""
        queries = _matrix(queries, "queries", self.dimensions, allow_empty=True)
        probes = _positive_integer(probes, "probes")
        rows = np.full((len(queries), probes), -1, dtype=np.int64)
        latencies = np.empty(len(queries), dtype=np.float64)
        selected_count = min(probes, len(self.prefixes))
        for query_row, query in enumerate(queries):
            start = perf_counter()
            # This is a single-threaded scan, so BLAS thread settings don't affect it.
            scores = np.einsum(
                "ij,j->i", self.signed_codes, query[: self.routing_bits], optimize=False
            )
            # Same score means same mismatch cost; use the address to break ties.
            order = np.lexsort((self.prefixes, -scores))[:selected_count]
            rows[query_row, :selected_count] = self.prefixes[order]
            latencies[query_row] = (perf_counter() - start) * 1000
        return rows, latencies


def build_router(documents, method="ivf", clusters=64, routing_bits=8, seed=42, threads=1):
    """Build once from float32 vectors. Input vectors are not normalized here."""
    documents = _matrix(documents, "documents")
    threads = _positive_integer(threads, "threads")
    start = perf_counter()
    if method == "ivf":
        clusters = _positive_integer(clusters, "clusters")
        if clusters > len(documents):
            raise ValueError("clusters cannot exceed the document count")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be an integer between 0 and 2^31 - 1")
        router = IVFRouter(
            documents=documents,
            clusters=clusters,
            seed=int(seed),
            threads=threads,
        )
    elif method == "sign":
        routing_bits = _positive_integer(routing_bits, "routing_bits")
        if routing_bits > min(documents.shape[1], 24):
            raise ValueError("routing_bits cannot exceed the vector dimension or 24")
        router = SignRouter(documents=documents, routing_bits=routing_bits)
    else:
        raise ValueError("method must be 'ivf' or 'sign'")
    router.build_ms = (perf_counter() - start) * 1000
    return router


def exact_search(documents, queries, top_k, threads=1):
    """Exhaustive inner-product reference. Index construction is outside query time."""
    documents = _matrix(documents, "documents")
    queries = _matrix(queries, "queries", documents.shape[1], allow_empty=True)
    top_k = _positive_integer(top_k, "top_k")
    faiss.omp_set_num_threads(_positive_integer(threads, "threads"))
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(documents)
    return _timed_search(index, queries, top_k)
