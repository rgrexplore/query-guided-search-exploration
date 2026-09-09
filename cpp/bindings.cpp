#include "index.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

namespace py = pybind11;

namespace {

// ----- Check NumPy inputs -----

template <typename T>
const T* checked_array(const py::array& array, const char* name, int rank) {
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has the wrong dtype");
    }
    if (array.ndim() != rank) {
        throw py::value_error(std::string(name) + " has the wrong number of axes");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T)) {
        throw py::value_error(std::string(name) + " must have aligned storage");
    }

    // Borrow NumPy's buffer. Nothing gets cast or copied here.
    return static_cast<const T*>(array.data());
}

void check_queries(const bitplane::Index& index, const py::array& queries,
                   const py::array& buckets) {
    const auto* query_data = checked_array<float>(queries, "queries", 2);
    const auto* bucket_data = checked_array<std::int64_t>(buckets, "buckets", 2);

    if (queries.shape(1) != static_cast<py::ssize_t>(index.dimensions())) {
        throw py::value_error("queries must have the index's number of dimensions");
    }
    if (buckets.shape(0) != queries.shape(0)) {
        throw py::value_error("buckets must have one row per query");
    }
    for (py::ssize_t value = 0; value < queries.size(); ++value) {
        if (!std::isfinite(query_data[value])) {
            throw py::value_error("queries must contain only finite values");
        }
    }

    for (py::ssize_t row = 0; row < buckets.shape(0); ++row) {
        std::unordered_set<std::int64_t> seen;
        for (py::ssize_t column = 0; column < buckets.shape(1); ++column) {
            const auto label = bucket_data[row * buckets.shape(1) + column];
            if (label < -1) {
                throw py::value_error("bucket labels must be nonnegative, or -1 for padding");
            }
            if (label >= 0 && !seen.insert(label).second) {
                throw py::value_error("a query cannot select the same bucket twice");
            }
        }
    }
}

// ----- Convert results back to Python -----

py::dict stats_to_python(const bitplane::SearchStats& stats) {
    py::dict result;
    result["elapsed_ms"] = stats.elapsed_ms;
    result["nodes"] = stats.nodes;
    result["random_nodes"] = stats.random_nodes;
    result["bitplane_words"] = stats.bitplane_words;
    result["documents_scored"] = stats.documents_scored;
    result["stop_reason"] = stats.stop_reason;

    if (stats.trace_enabled) {
        py::list trace;
        for (const auto& step : stats.trace) {
            py::dict record;
            record["depth"] = step.depth;
            record["dimension"] = step.dimension;
            record["penalty"] = step.penalty;
            record["upper_bound"] = step.upper_bound;
            record["documents"] = step.documents;
            record["random"] = step.random;
            record["event"] = step.event;
            trace.append(record);
        }
        result["trace"] = trace;
    }
    return result;
}

py::dict results_to_python(const std::vector<bitplane::QueryResult>& results,
                           py::ssize_t limit) {
    const auto query_count = static_cast<py::ssize_t>(results.size());
    py::array_t<std::int64_t> rows({query_count, limit});
    py::array_t<double> scores({query_count, limit});
    py::array_t<std::int64_t> counts(query_count);

    // A budget-limited search can return fewer rows than requested.
    std::fill_n(rows.mutable_data(), rows.size(), -1);
    std::fill_n(scores.mutable_data(), scores.size(), -std::numeric_limits<double>::infinity());

    py::list stats;
    for (py::ssize_t query = 0; query < query_count; ++query) {
        const auto& result = results[query];
        counts.mutable_at(query) = static_cast<std::int64_t>(result.rows.size());
        std::copy(result.rows.begin(), result.rows.end(), rows.mutable_data(query, 0));
        std::copy(result.scores.begin(), result.scores.end(), scores.mutable_data(query, 0));
        stats.append(stats_to_python(result.stats));
    }

    py::dict result;
    result["rows"] = rows;
    result["scores"] = scores;
    result["counts"] = counts;
    result["stats"] = stats;
    return result;
}

// ----- Build, scan, and search -----

std::unique_ptr<bitplane::Index> build_index(const py::array& codes,
                                             const py::array& assignments,
                                             int dimensions) {
    const auto* code_data = checked_array<std::uint64_t>(codes, "codes", 2);
    const auto* assignment_data = checked_array<std::int64_t>(assignments, "assignments", 1);
    const auto documents = codes.shape(0);

    if (dimensions <= 0) {
        throw py::value_error("dimensions must be positive");
    }
    if (codes.shape(1) != (static_cast<py::ssize_t>(dimensions) + 63) / 64) {
        throw py::value_error("codes must have ceil(dimensions / 64) words per row");
    }
    if (assignments.shape(0) != documents) {
        throw py::value_error("assignments must have one label per document");
    }
    for (py::ssize_t row = 0; row < assignments.size(); ++row) {
        if (assignment_data[row] < 0) {
            throw py::value_error("assignments must be nonnegative");
        }
    }

    // The index makes its own copy; it doesn't keep these NumPy buffers.
    py::gil_scoped_release release;
    return std::make_unique<bitplane::Index>(code_data, assignment_data, documents, dimensions);
}

py::dict scan_index(const bitplane::Index& index, const py::array& queries,
                    const py::array& buckets, py::ssize_t candidate_limit) {
    check_queries(index, queries, buckets);
    if (candidate_limit <= 0) {
        throw py::value_error("candidate_limit must be positive");
    }

    const auto* query_data = static_cast<const float*>(queries.data());
    const auto* bucket_data = static_cast<const std::int64_t*>(buckets.data());
    const auto query_count = queries.shape(0);
    const auto probes = buckets.shape(1);

    std::vector<bitplane::QueryResult> results;
    {
        // Let other Python threads run while C++ works on the borrowed buffers.
        py::gil_scoped_release release;
        results = index.scan(query_data, query_count, bucket_data, probes, candidate_limit);
    }
    return results_to_python(results, candidate_limit);
}

py::dict search_index(const bitplane::Index& index, const py::array& queries,
                      const py::array& buckets, py::ssize_t candidate_limit,
                      py::ssize_t node_budget, py::ssize_t leaf_size,
                      double explore_probability, std::uint64_t seed, bool trace) {
    check_queries(index, queries, buckets);
    if (candidate_limit <= 0) {
        throw py::value_error("candidate_limit must be positive");
    }
    if (node_budget < 0) {
        throw py::value_error("node_budget must be nonnegative");
    }
    if (leaf_size <= 0) {
        throw py::value_error("leaf_size must be positive");
    }
    if (!std::isfinite(explore_probability) || explore_probability < 0 || explore_probability > 1) {
        throw py::value_error("explore_probability must be between 0 and 1");
    }

    bitplane::SearchOptions options;
    options.candidate_limit = candidate_limit;
    options.node_budget = node_budget;
    options.leaf_size = leaf_size;
    options.explore_probability = explore_probability;
    options.seed = seed;
    options.trace = trace;

    const auto* query_data = static_cast<const float*>(queries.data());
    const auto* bucket_data = static_cast<const std::int64_t*>(buckets.data());
    const auto query_count = queries.shape(0);
    const auto probes = buckets.shape(1);

    std::vector<bitplane::QueryResult> results;
    {
        py::gil_scoped_release release;
        results = index.search(query_data, query_count, bucket_data, probes, options);
    }
    return results_to_python(results, candidate_limit);
}

py::dict index_info(const bitplane::Index& index) {
    const auto info = index.info();
    py::dict result;
    result["documents"] = info.documents;
    result["dimensions"] = info.dimensions;
    result["buckets"] = info.buckets;
    result["codes_bytes"] = info.codes_bytes;
    result["bitplanes_bytes"] = info.bitplanes_bytes;
    result["row_ids_bytes"] = info.row_ids_bytes;
    result["logical_bytes"] = info.codes_bytes + info.bitplanes_bytes + info.row_ids_bytes;
    return result;
}

} // namespace

// ----- Python API -----

PYBIND11_MODULE(bitplane_index, module) {
    // This is what Python loads with `import bitplane_index`.
    module.doc() = "Packed binary scan and bucket-local bitplane search.";
    auto index_type = py::class_<bitplane::Index>(module, "Index");

    // Keep NumPy inputs as-is. A hidden dtype conversion would also hide a copy.
    index_type.def(
        py::init(&build_index),
        py::arg("codes").noconvert(),
        py::arg("assignments").noconvert(),
        py::arg("dimensions")
    );

    index_type.def(
        "scan", &scan_index,
        py::arg("queries").noconvert(),
        py::arg("buckets").noconvert(),
        py::arg("candidate_limit") = 100
    );

    index_type.def(
        "search", &search_index,
        py::arg("queries").noconvert(),
        py::arg("buckets").noconvert(),
        py::arg("candidate_limit") = 100,
        py::arg("node_budget") = 128,
        py::arg("leaf_size") = 32,
        py::arg("explore_probability") = 0.0,
        py::arg("seed") = 0,
        py::arg("trace") = false
    );

    index_type.def("info", &index_info);
}
