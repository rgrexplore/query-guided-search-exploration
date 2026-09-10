#pragma once

#include "index.hpp"

namespace bitplane {

struct KeySearchOptions {
    std::size_t candidate_limit = 100; // Number of final results to keep.
    std::size_t candidate_target = 1000; // Stop after this many scores; zero disables it.
    std::size_t key_limit = 8192; // Maximum directory lookups; zero disables it.
};

// Each cluster stores rows in key order. A directory maps an occupied key to a
// contiguous range, so IDs also serve as the postings; no second ID array is needed.
class KeyIndex {
public:
    KeyIndex(const std::uint64_t* codes, const std::int64_t* assignments,
             std::size_t documents, std::size_t dimensions,
             std::size_t key_bits, std::size_t key_offset = 0);

    std::vector<QueryResult> search(const float* queries, std::size_t query_count,
                                   const std::int64_t* selected_buckets, std::size_t probes,
                                   const KeySearchOptions& options) const;
    StorageInfo info() const;
    std::size_t dimensions() const { return dimensions_; }

private:
    struct Range {
        std::size_t start;
        std::size_t count;
    };
    struct Bucket {
        std::vector<std::int64_t> rows;
        std::vector<std::uint64_t> codes;
        std::unordered_map<std::uint32_t, Range> directory;
    };

    std::uint32_t key_of(const std::uint64_t* code) const;
    std::size_t documents_;
    std::size_t dimensions_;
    std::size_t code_words_;
    std::size_t key_bits_;
    std::size_t key_offset_;
    std::vector<Bucket> buckets_;
    std::unordered_map<std::int64_t, std::size_t> bucket_lookup_;
};

} // namespace bitplane
