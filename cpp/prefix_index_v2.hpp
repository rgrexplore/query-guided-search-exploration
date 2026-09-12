#pragma once

#include "index.hpp"

namespace bitplane {

struct PrefixSearchOptionsV2 {
    std::size_t candidate_limit = 100;
    std::size_t start_depth = 16;
    std::size_t candidate_target = 1000; // Zero visits every depth through zero.
};

// Static rows sorted by the first embedding signs, then original document ID.
// Relaxing one trailing prefix bit widens a contiguous range in each cluster.
class PrefixIndexV2 {
public:
    PrefixIndexV2(const std::uint64_t* codes, const std::int64_t* assignments,
                  std::size_t documents, std::size_t dimensions,
                  std::size_t max_prefix_bits = 24);

    std::vector<QueryResult> search(const float* queries, std::size_t query_count,
                                    const std::int64_t* selected_buckets, std::size_t probes,
                                    const PrefixSearchOptionsV2& options) const;
    StorageInfo info() const;
    std::size_t dimensions() const { return dimensions_; }
    std::size_t max_prefix_bits() const { return max_prefix_bits_; }

private:
    struct Range {
        std::size_t left;
        std::size_t right; // Excluded.
    };
    struct Bucket {
        std::vector<std::int64_t> rows;
        std::vector<std::uint64_t> codes;
        std::vector<std::uint32_t> prefix_keys;
    };

    std::uint32_t key_of(const std::uint64_t* code) const;
    Range prefix_range(const Bucket& bucket, std::uint32_t query_key,
                        std::size_t depth) const;

    std::size_t documents_;
    std::size_t dimensions_;
    std::size_t code_words_;
    std::size_t max_prefix_bits_;
    std::vector<Bucket> buckets_;
    std::unordered_map<std::int64_t, std::size_t> bucket_lookup_;
};

} // namespace bitplane
