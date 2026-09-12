#include "prefix_index_v2.hpp"
#include "score.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace bitplane {
namespace {

std::size_t score_rows(const std::vector<std::uint64_t>& codes,
                       const std::vector<std::int64_t>& rows, std::size_t code_words,
                       std::size_t left, std::size_t right,
                       const detail::ScoreTable& scorer, detail::TopCandidates& best) {
    for (std::size_t local = left; local < right; ++local) {
        best.offer(rows[local], scorer.score(codes.data() + local * code_words));
    }
    return right - left;
}

} // namespace

std::uint32_t PrefixIndexV2::key_of(const std::uint64_t* code) const {
    std::uint32_t key = 0;
    for (std::size_t bit = 0; bit < max_prefix_bits_; ++bit) {
        // Packed bit zero is the first embedding sign, so it becomes the key's
        // most significant bit. Numeric key order now follows prefix order.
        key = (key << 1) | std::uint32_t((code[bit / 64] >> (bit % 64)) & 1);
    }
    return key;
}

PrefixIndexV2::PrefixIndexV2(const std::uint64_t* codes, const std::int64_t* assignments,
                            std::size_t documents, std::size_t dimensions,
                            std::size_t max_prefix_bits)
    : documents_(documents), dimensions_(dimensions), code_words_((dimensions + 63) / 64),
      max_prefix_bits_(max_prefix_bits) {
    // Extract each key once, instead of reading its bits on every sort comparison.
    std::vector<std::uint32_t> keys(documents);
    for (std::size_t row = 0; row < documents; ++row) {
        keys[row] = key_of(codes + row * code_words_);
        auto [entry, inserted] = bucket_lookup_.try_emplace(assignments[row], buckets_.size());
        if (inserted) {
            buckets_.emplace_back();
        }
        buckets_[entry->second].rows.push_back(static_cast<std::int64_t>(row));
    }
    for (auto& bucket : buckets_) {
        std::sort(bucket.rows.begin(), bucket.rows.end(), [&](auto left, auto right) {
            return keys[left] < keys[right] || (keys[left] == keys[right] && left < right);
        });
        bucket.codes.resize(bucket.rows.size() * code_words_);
        bucket.prefix_keys.resize(bucket.rows.size());
        for (std::size_t local = 0; local < bucket.rows.size(); ++local) {
            const auto row = bucket.rows[local];
            bucket.prefix_keys[local] = keys[row];
            std::copy_n(codes + row * code_words_, code_words_,
                        bucket.codes.data() + local * code_words_);
        }
    }
}

PrefixIndexV2::Range PrefixIndexV2::prefix_range(const Bucket& bucket,
                                                 std::uint32_t query_key,
                                                 std::size_t depth) const {
    if (depth == 0) {
        return {0, bucket.rows.size()};
    }
    const auto shift = max_prefix_bits_ - depth;
    // A 32-bit all-ones prefix has an exclusive high endpoint of 2**32.
    const auto prefix = std::uint64_t(query_key) >> shift;
    const auto low = prefix << shift;
    const auto high = (prefix + 1) << shift;
    const auto begin = bucket.prefix_keys.begin();
    const auto end = bucket.prefix_keys.end();
    const auto left = std::lower_bound(begin, end, low);
    const auto right = std::lower_bound(begin, end, high);
    return {static_cast<std::size_t>(left - begin), static_cast<std::size_t>(right - begin)};
}

std::vector<QueryResult> PrefixIndexV2::search(const float* queries, std::size_t query_count,
                                              const std::int64_t* selected_buckets,
                                              std::size_t probes,
                                              const PrefixSearchOptionsV2& options) const {
    struct SelectedBucket {
        const Bucket* bucket;
        Range previous;
    };

    std::vector<QueryResult> results(query_count);
    for (std::size_t qi = 0; qi < query_count; ++qi) {
        const auto start = std::chrono::steady_clock::now();
        auto& result = results[qi];
        auto& stats = result.stats;
        const auto* query = queries + qi * dimensions_;
        detail::ScoreTable scorer(query, dimensions_);
        detail::TopCandidates best(options.candidate_limit);
        std::uint32_t query_key = 0;
        for (std::size_t bit = 0; bit < max_prefix_bits_; ++bit) {
            query_key = (query_key << 1) | std::uint32_t(query[bit] >= 0);
        }

        double query_l1 = 0;
        std::vector<double> prefix_min_weights;
        if (options.stop_when_exact && options.start_depth > 0) {
            prefix_min_weights.resize(options.start_depth);
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                const auto weight = std::abs(double(query[dimension]));
                query_l1 += weight;
                if (dimension < options.start_depth) {
                    prefix_min_weights[dimension] = dimension == 0 ? weight
                        : std::min(prefix_min_weights[dimension - 1], weight);
                }
            }
        }

        std::vector<SelectedBucket> selected;
        selected.reserve(probes);
        for (std::size_t probe = 0; probe < probes; ++probe) {
            const auto entry = bucket_lookup_.find(selected_buckets[qi * probes + probe]);
            if (entry != bucket_lookup_.end()) {
                selected.push_back({&buckets_[entry->second], {0, 0}});
            }
        }

        for (std::size_t depth = options.start_depth;; --depth) {
            ++stats.prefix_levels;
            stats.final_depth = depth;
            for (auto& opened : selected) {
                const auto& bucket = *opened.bucket;
                const auto range = prefix_range(bucket, query_key, depth);
                stats.prefix_lookups += depth ? 2 : 0;
                if (depth == options.start_depth) {
                    stats.documents_scored += score_rows(bucket.codes, bucket.rows, code_words_,
                                                          range.left, range.right, scorer, best);
                } else {
                    // The old range is inside its parent, including when it is
                    // empty: its insertion position still divides the new slices.
                    stats.documents_scored += score_rows(bucket.codes, bucket.rows, code_words_,
                                                          range.left, opened.previous.left,
                                                          scorer, best);
                    stats.documents_scored += score_rows(bucket.codes, bucket.rows, code_words_,
                                                          opened.previous.right, range.right,
                                                          scorer, best);
                }
                opened.previous = range;
            }
            // All matching rows at this depth have been scored across every opened
            // cluster. Any unseen row mismatches at least one constrained sign.
            if (options.stop_when_exact && depth > 0
                    && detail::cannot_improve(prefix_min_weights[depth - 1], query_l1,
                                               dimensions_, best)) {
                stats.stop_reason = "bound";
                break;
            }
            // A positive target retains its existing approximate stopping rule.
            if (options.candidate_target && stats.documents_scored >= options.candidate_target) {
                stats.stop_reason = "candidate_target";
                break;
            }
            if (depth == 0) {
                break;
            }
        }
        best.write_to(result);
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    return results;
}

StorageInfo PrefixIndexV2::info() const {
    StorageInfo result;
    result.documents = documents_;
    result.dimensions = dimensions_;
    result.buckets = buckets_.size();
    for (const auto& bucket : buckets_) {
        result.codes_bytes += bucket.codes.size() * sizeof(std::uint64_t);
        result.row_ids_bytes += bucket.rows.size() * sizeof(std::int64_t);
        result.prefix_keys_bytes += bucket.prefix_keys.size() * sizeof(std::uint32_t);
        result.array_capacity_bytes += bucket.codes.capacity() * sizeof(std::uint64_t)
                                       + bucket.rows.capacity() * sizeof(std::int64_t)
                                       + bucket.prefix_keys.capacity() * sizeof(std::uint32_t);
    }
    return result;
}

} // namespace bitplane
