#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bitplane {

struct SearchOptions {
    std::size_t candidate_limit = 100;
    std::size_t node_budget = 128; // Zero lets the search finish.
    std::size_t leaf_size = 32; // Score the rows once a group is this small.
    double explore_probability = 0.0; // Chance to try a queued alternative.
    std::uint64_t seed = 0;
    bool trace = false;
};

struct TraceStep {
    std::size_t depth;
    std::int64_t dimension;
    double penalty;
    double upper_bound;
    std::size_t documents;
    bool random;
    std::string event;
};

struct SearchStats {
    double elapsed_ms = 0;
    std::size_t nodes = 0;
    std::size_t random_nodes = 0;
    std::size_t bitplane_words = 0;
    std::size_t documents_scored = 0;
    std::size_t leaf_words = 0; // Separate from the existing split-word counter.
    std::size_t peak_mask_bytes = 0; // Live bitmap allocations, excluding queue metadata.
    std::size_t key_attempts = 0; // Includes lookups with no matching documents.
    std::size_t keys_generated = 0;
    std::size_t peak_key_queue_bytes = 0;
    std::size_t prefix_levels = 0; // Visited depths, including depth zero.
    std::size_t prefix_lookups = 0; // Binary boundary searches; none at depth zero.
    std::size_t final_depth = 0;
    std::string stop_reason = "exhausted";
    bool trace_enabled = false;
    std::vector<TraceStep> trace;
};

struct QueryResult {
    std::vector<std::int64_t> rows;
    std::vector<double> scores;
    SearchStats stats;
};

struct StorageInfo {
    std::size_t documents = 0;
    std::size_t dimensions = 0;
    std::size_t buckets = 0;
    std::size_t codes_bytes = 0;
    std::size_t bitplanes_bytes = 0;
    std::size_t row_ids_bytes = 0;
    std::size_t prefix_keys_bytes = 0;
    std::size_t array_capacity_bytes = 0; // Allocated vector storage; not whole-process RAM.
    std::size_t key_directory_payload_bytes = 0;
    std::size_t occupied_keys = 0;
    std::size_t directory_slots = 0; // Hash bucket counts, not an allocator byte estimate.
};

// Bit j lives at word j / 64, offset j % 64. A 1 scores as +1; a 0 scores as -1.
// Each bucket owns packed rows for scoring and, when requested, bitplanes for splitting.
class Index {
public:
    Index(const std::uint64_t* codes, const std::int64_t* assignments,
          std::size_t documents, std::size_t dimensions, bool build_bitplanes = true);

    std::vector<QueryResult> scan(const float* queries, std::size_t query_count,
                                  const std::int64_t* selected_buckets,
                                  std::size_t probes, std::size_t candidate_limit) const;

    std::vector<QueryResult> search(const float* queries, std::size_t query_count,
                                    const std::int64_t* selected_buckets,
                                    std::size_t probes, const SearchOptions& options) const;

    StorageInfo info() const;
    std::size_t dimensions() const { return dimensions_; }

private:
    struct Bucket {
        std::vector<std::int64_t> rows; // Original document row IDs.
        std::vector<std::uint64_t> codes; // Packed signs, one document after another.
        std::vector<std::uint64_t> planes; // One bitmap per dimension, over local rows.
        std::size_t bitmap_words = 0;
    };

    std::size_t documents_;
    std::size_t dimensions_;
    std::size_t code_words_;
    bool has_bitplanes_;
    std::vector<Bucket> buckets_;
    std::unordered_map<std::int64_t, std::size_t> bucket_lookup_;
};

} // namespace bitplane
