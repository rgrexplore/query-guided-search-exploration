#include "key_index.hpp"
#include "score.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace bitplane {
namespace {

// last is the greatest position in the sorted list of query weights. A state
// has at most two successors: append the next bit, or replace last with next.
// Every flip set has one parent, so no visited-set table is needed.
struct FlipSet {
    double penalty;
    std::uint32_t mask;
    int last;
};

bool more_expensive(const FlipSet& left, const FlipSet& right) {
    if (left.penalty != right.penalty) {
        return left.penalty > right.penalty;
    }
    return left.mask > right.mask;
}

} // namespace

std::uint32_t KeyIndex::key_of(const std::uint64_t* code) const {
    std::uint32_t key = 0;
    for (std::size_t bit = 0; bit < key_bits_; ++bit) {
        const auto dimension = key_offset_ + bit;
        key |= std::uint32_t((code[dimension / 64] >> (dimension % 64)) & 1) << bit;
    }
    return key;
}

KeyIndex::KeyIndex(const std::uint64_t* codes, const std::int64_t* assignments,
                   std::size_t documents, std::size_t dimensions,
                   std::size_t key_bits, std::size_t key_offset)
    : documents_(documents), dimensions_(dimensions), code_words_((dimensions + 63) / 64),
      key_bits_(key_bits), key_offset_(key_offset) {
    for (std::size_t row = 0; row < documents; ++row) {
        auto [entry, inserted] = bucket_lookup_.try_emplace(assignments[row], buckets_.size());
        if (inserted) {
            buckets_.emplace_back();
        }
        buckets_[entry->second].rows.push_back(static_cast<std::int64_t>(row));
    }
    for (auto& bucket : buckets_) {
        // Sorting is offline. Copy full codes into this order too, so a posting
        // range can be scored sequentially rather than gathering original rows.
        std::sort(bucket.rows.begin(), bucket.rows.end(), [&](auto left, auto right) {
            const auto left_key = key_of(codes + left * code_words_);
            const auto right_key = key_of(codes + right * code_words_);
            return left_key < right_key || (left_key == right_key && left < right);
        });
        bucket.codes.resize(bucket.rows.size() * code_words_);
        for (std::size_t local = 0; local < bucket.rows.size(); ++local) {
            const auto* source = codes + bucket.rows[local] * code_words_;
            std::copy_n(source, code_words_, bucket.codes.data() + local * code_words_);
            const auto key = key_of(source);
            auto [entry, inserted] = bucket.directory.try_emplace(key, Range{local, 0});
            ++entry->second.count;
        }
    }
}

std::vector<QueryResult> KeyIndex::search(const float* queries, std::size_t query_count,
                                         const std::int64_t* selected_buckets, std::size_t probes,
                                         const KeySearchOptions& options) const {
    std::vector<QueryResult> results(query_count);
    for (std::size_t qi = 0; qi < query_count; ++qi) {
        const auto start = std::chrono::steady_clock::now();
        auto& result = results[qi];
        auto& stats = result.stats;
        const auto* query = queries + qi * dimensions_;
        detail::ScoreTable scorer(query, dimensions_);
        detail::TopCandidates best(options.candidate_limit);
        double query_l1 = 0;
        for (std::size_t bit = 0; bit < dimensions_; ++bit) {
            query_l1 += std::abs(double(query[bit]));
        }

        std::vector<const Bucket*> selected;
        for (std::size_t probe = 0; probe < probes; ++probe) {
            const auto entry = bucket_lookup_.find(selected_buckets[qi * probes + probe]);
            if (entry != bucket_lookup_.end()) {
                selected.push_back(&buckets_[entry->second]);
            }
        }

        std::vector<std::size_t> order(key_bits_);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
            return std::abs(query[key_offset_ + left]) < std::abs(query[key_offset_ + right]);
        });
        std::uint32_t ideal = 0;
        for (std::size_t bit = 0; bit < key_bits_; ++bit) {
            ideal |= std::uint32_t(query[key_offset_ + bit] >= 0) << bit;
        }

        std::vector<FlipSet> queue;
        auto push = [&](FlipSet state) {
            queue.push_back(state);
            std::push_heap(queue.begin(), queue.end(), more_expensive);
            stats.peak_key_queue_bytes = std::max(stats.peak_key_queue_bytes,
                                                  queue.capacity() * sizeof(FlipSet));
        };
        if (!selected.empty()) {
            push({0, 0, -1}); // Start with the query's own key.
        }

        while (!queue.empty()) {
            if (detail::cannot_improve(queue.front().penalty, query_l1, dimensions_, best)) {
                stats.stop_reason = "bound";
                break;
            }
            std::pop_heap(queue.begin(), queue.end(), more_expensive);
            const auto state = queue.back();
            queue.pop_back();
            ++stats.keys_generated;
            const auto key = ideal ^ state.mask;

            for (const auto* bucket : selected) {
                if (options.key_limit && stats.key_attempts >= options.key_limit) {
                    stats.stop_reason = "key_limit";
                    break;
                }
                ++stats.key_attempts;
                const auto entry = bucket->directory.find(key);
                if (entry == bucket->directory.end()) {
                    continue; // Empty keys still consume one lookup.
                }
                const auto range = entry->second;
                for (std::size_t local = range.start; local < range.start + range.count; ++local) {
                    best.offer(bucket->rows[local], scorer.score(bucket->codes.data() + local * code_words_));
                }
                stats.documents_scored += range.count;
            }
            if (stats.stop_reason == "key_limit") {
                break;
            }
            // Finish this key in every selected cluster before applying the quota.
            if (options.candidate_target && stats.documents_scored >= options.candidate_target) {
                stats.stop_reason = "candidate_target";
                break;
            }

            const auto next = state.last + 1;
            if (next < static_cast<int>(key_bits_)) {
                const auto next_mask = std::uint32_t{1} << order[next];
                const auto next_weight = std::abs(double(query[key_offset_ + order[next]]));
                push({state.penalty + next_weight, state.mask | next_mask, next});
                if (state.last >= 0) {
                    const auto last_mask = std::uint32_t{1} << order[state.last];
                    const auto last_weight = std::abs(double(query[key_offset_ + order[state.last]]));
                    push({state.penalty - last_weight + next_weight,
                          (state.mask ^ last_mask) | next_mask, next});
                }
            }
        }
        best.write_to(result);
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    return results;
}

StorageInfo KeyIndex::info() const {
    StorageInfo result;
    result.documents = documents_;
    result.dimensions = dimensions_;
    result.buckets = buckets_.size();
    for (const auto& bucket : buckets_) {
        result.codes_bytes += bucket.codes.size() * sizeof(std::uint64_t);
        result.row_ids_bytes += bucket.rows.size() * sizeof(std::int64_t);
        result.array_capacity_bytes += bucket.codes.capacity() * sizeof(std::uint64_t)
                                       + bucket.rows.capacity() * sizeof(std::int64_t);
        result.occupied_keys += bucket.directory.size();
        result.directory_slots += bucket.directory.bucket_count();
        // Logical payload only. Hash-node padding and allocator overhead belong
        // in the separate process-memory measurement, not this byte count.
        result.key_directory_payload_bytes += bucket.directory.size()
                                              * (sizeof(std::uint32_t) + sizeof(Range));
    }
    return result;
}

} // namespace bitplane
