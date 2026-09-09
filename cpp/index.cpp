#include "index.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <utility>

namespace bitplane {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Score four signs at a time. Queries stay floating point throughout.
class ScoreTable {
public:
    ScoreTable(const float* query, std::size_t dimensions)
        : entries_((dimensions + 3) / 4) {
        for (std::size_t group = 0; group < entries_.size(); ++group) {
            for (std::size_t pattern = 0; pattern < 16; ++pattern) {
                double sum = 0;
                for (std::size_t bit = 0; bit < 4; ++bit) {
                    const auto dimension = group * 4 + bit;
                    if (dimension < dimensions) {
                        const bool positive = (pattern >> bit) & 1;
                        const double query_value = query[dimension];
                        sum += positive ? query_value : -query_value;
                    }
                }
                entries_[group][pattern] = sum;
            }
        }
    }

    double score(const std::uint64_t* code) const {
        double total = 0;
        for (std::size_t group = 0; group < entries_.size(); ++group) {
            // A 64-bit word holds sixteen groups of four signs.
            const auto word = code[group / 16];
            const auto shift = (group % 16) * 4;
            const auto pattern = (word >> shift) & 15;
            total += entries_[group][pattern];
        }
        return total;
    }

private:
    std::vector<std::array<double, 16>> entries_;
};

struct Candidate {
    std::int64_t row;
    double score;
};

struct BetterCandidate {
    bool operator()(const Candidate& left, const Candidate& right) const {
        return left.score > right.score || (left.score == right.score && left.row < right.row);
    }
};

class TopCandidates {
public:
    explicit TopCandidates(std::size_t limit) : limit_(limit) {}

    void offer(std::int64_t row, double score) {
        Candidate candidate{row, score};
        if (heap_.size() < limit_) {
            heap_.push(candidate);
        } else if (BetterCandidate{}(candidate, heap_.top())) {
            heap_.pop();
            heap_.push(candidate);
        }
    }

    bool full() const { return heap_.size() == limit_; }
    double worst_score() const { return heap_.top().score; }

    void write_to(QueryResult& result) {
        std::vector<Candidate> ranked;
        ranked.reserve(heap_.size());
        while (!heap_.empty()) {
            ranked.push_back(heap_.top());
            heap_.pop();
        }
        std::sort(ranked.begin(), ranked.end(), BetterCandidate{});
        for (const auto& candidate : ranked) {
            result.rows.push_back(candidate.row);
            result.scores.push_back(candidate.score);
        }
    }

private:
    std::size_t limit_;
    // The worst kept result stays on top, so replacing it is cheap.
    std::priority_queue<Candidate, std::vector<Candidate>, BetterCandidate> heap_;
};

struct Node {
    std::size_t bucket;
    std::size_t depth;
    std::size_t count;
    double penalty; // Sum of |q[j]| for signs this branch got wrong.
    std::uint64_t order;
    std::vector<std::uint64_t> active; // One bit for each row still in this branch.
};

bool better_node(const Node& left, const Node& right) {
    return left.penalty < right.penalty ||
           (left.penalty == right.penalty && left.order < right.order);
}

// Index 0 is the cheapest branch. Other slots can be picked for exploration.
class Frontier {
public:
    void push(Node node) {
        nodes_.push_back(std::move(node));
        sift_up(nodes_.size() - 1);
    }

    Node take(std::size_t index) {
        Node result = std::move(nodes_[index]);
        if (index == nodes_.size() - 1) {
            nodes_.pop_back();
            return result;
        }
        nodes_[index] = std::move(nodes_.back());
        nodes_.pop_back();
        if (index > 0 && better_node(nodes_[index], nodes_[(index - 1) / 2])) {
            sift_up(index);
        } else {
            sift_down(index);
        }
        return result;
    }

    bool empty() const { return nodes_.empty(); }
    std::size_t size() const { return nodes_.size(); }
    double best_penalty() const { return nodes_.front().penalty; }

private:
    void sift_up(std::size_t index) {
        while (index > 0) {
            const auto parent = (index - 1) / 2;
            if (!better_node(nodes_[index], nodes_[parent])) {
                break;
            }
            std::swap(nodes_[index], nodes_[parent]);
            index = parent;
        }
    }

    void sift_down(std::size_t index) {
        while (index * 2 + 1 < nodes_.size()) {
            auto child = index * 2 + 1;
            if (child + 1 < nodes_.size() && better_node(nodes_[child + 1], nodes_[child])) {
                ++child;
            }
            if (!better_node(nodes_[child], nodes_[index])) {
                break;
            }
            std::swap(nodes_[index], nodes_[child]);
            index = child;
        }
    }

    std::vector<Node> nodes_;
};

bool cannot_improve(double best_penalty, double query_l1, std::size_t dimensions,
                    const TopCandidates& candidates) {
    if (!candidates.full()) {
        return false;
    }
    // A wrong sign changes +|q[j]| to -|q[j]|, so it costs twice the penalty.
    const double upper_bound = query_l1 - 2 * best_penalty;
    // Rounding can differ between the bound and the lookup-table score.
    const double guard = 32 * std::numeric_limits<double>::epsilon() *
                         (dimensions + 1) * (query_l1 + std::abs(candidates.worst_score()) + 1);
    // Keep ties alive: another branch may contain a smaller row ID.
    return upper_bound + guard < candidates.worst_score();
}

} // namespace

// ----- Build the two document layouts -----

Index::Index(const std::uint64_t* codes, const std::int64_t* assignments,
             std::size_t documents, std::size_t dimensions)
    : documents_(documents), dimensions_(dimensions), code_words_((dimensions + 63) / 64) {
    for (std::size_t row = 0; row < documents_; ++row) {
        auto entry = bucket_lookup_.find(assignments[row]);
        if (entry == bucket_lookup_.end()) {
            const auto bucket_id = buckets_.size();
            buckets_.emplace_back();
            entry = bucket_lookup_.emplace(assignments[row], bucket_id).first;
        }
        auto& bucket = buckets_[entry->second];
        bucket.rows.push_back(static_cast<std::int64_t>(row));
        const auto* row_start = codes + row * code_words_;
        bucket.codes.insert(bucket.codes.end(), row_start, row_start + code_words_);
    }

    // Turn the rows sideways: each plane marks the rows with a 1 at one dimension.
    for (auto& bucket : buckets_) {
        bucket.bitmap_words = (bucket.rows.size() + 63) / 64;
        bucket.planes.resize(dimensions_ * bucket.bitmap_words, 0);
        for (std::size_t local_row = 0; local_row < bucket.rows.size(); ++local_row) {
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                const auto code = bucket.codes[local_row * code_words_ + dimension / 64];
                if ((code >> (dimension % 64)) & 1) {
                    bucket.planes[dimension * bucket.bitmap_words + local_row / 64] |=
                        std::uint64_t{1} << (local_row % 64);
                }
            }
        }
    }
}

// ----- Scan every row in the selected buckets -----

std::vector<QueryResult> Index::scan(const float* queries, std::size_t query_count,
                                     const std::int64_t* selected_buckets,
                                     std::size_t probes, std::size_t candidate_limit) const {
    std::vector<QueryResult> results(query_count);
    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const auto start = Clock::now();
        auto& result = results[query_id];
        ScoreTable table(queries + query_id * dimensions_, dimensions_);
        TopCandidates candidates(candidate_limit);

        for (std::size_t probe = 0; probe < probes; ++probe) {
            const auto entry = bucket_lookup_.find(selected_buckets[query_id * probes + probe]);
            if (entry == bucket_lookup_.end()) {
                continue;
            }
            const auto& bucket = buckets_[entry->second];
            for (std::size_t row = 0; row < bucket.rows.size(); ++row) {
                candidates.offer(bucket.rows[row], table.score(bucket.codes.data() + row * code_words_));
            }
            result.stats.documents_scored += bucket.rows.size();
        }
        candidates.write_to(result);
        result.stats.elapsed_ms = milliseconds_since(start);
    }
    return results;
}

// ----- Traverse bitplanes, keeping alternative branches in the frontier -----

std::vector<QueryResult> Index::search(const float* queries, std::size_t query_count,
                                       const std::int64_t* selected_buckets,
                                       std::size_t probes, const SearchOptions& options) const {
    std::vector<QueryResult> results(query_count);
    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const auto start = Clock::now();
        const auto* query = queries + query_id * dimensions_;
        auto& result = results[query_id];
        result.stats.trace_enabled = options.trace;
        ScoreTable table(query, dimensions_);
        TopCandidates candidates(options.candidate_limit);

        // Check large query coordinates first.
        std::vector<std::size_t> dimension_order(dimensions_);
        std::iota(dimension_order.begin(), dimension_order.end(), 0);
        std::stable_sort(dimension_order.begin(), dimension_order.end(), [query](auto left, auto right) {
            return std::abs(query[left]) > std::abs(query[right]);
        });
        double query_l1 = 0;
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            query_l1 += std::abs(double(query[dimension]));
        }

        // Start with every row active in each selected bucket.
        Frontier frontier;
        std::uint64_t next_order = 0;
        for (std::size_t probe = 0; probe < probes; ++probe) {
            const auto entry = bucket_lookup_.find(selected_buckets[query_id * probes + probe]);
            if (entry == bucket_lookup_.end()) {
                continue;
            }
            const auto& bucket = buckets_[entry->second];
            std::vector<std::uint64_t> active(bucket.bitmap_words, ~std::uint64_t{0});
            const auto tail = bucket.rows.size() % 64;
            if (tail) {
                // The last word may have unused bits. Keep those off.
                active.back() = (std::uint64_t{1} << tail) - 1;
            }
            frontier.push({entry->second, 0, bucket.rows.size(), 0, next_order++, std::move(active)});
        }

        // A batch uses the same seeds as separate calls with seed + row.
        std::mt19937_64 random(options.seed + query_id);
        std::bernoulli_distribution explore(options.explore_probability);
        while (!frontier.empty()) {
            if (cannot_improve(frontier.best_penalty(), query_l1, dimensions_, candidates)) {
                result.stats.stop_reason = "bound";
                break;
            }
            if (options.node_budget && result.stats.nodes >= options.node_budget) {
                result.stats.stop_reason = "budget";
                break;
            }

            // Exploration changes visit order; the other branches stay queued.
            std::size_t chosen = 0;
            if (frontier.size() > 1 && options.explore_probability > 0 && explore(random)) {
                chosen = std::uniform_int_distribution<std::size_t>(1, frontier.size() - 1)(random);
                ++result.stats.random_nodes;
            }
            auto node = frontier.take(chosen);
            ++result.stats.nodes;
            const auto& bucket = buckets_[node.bucket];
            const bool score_now = node.count <= options.leaf_size || node.depth == dimensions_;
            if (options.trace) {
                // Keep the path small: counts and decisions, no document bitmaps.
                result.stats.trace.push_back({
                    node.depth, score_now ? -1 : static_cast<std::int64_t>(dimension_order[node.depth]),
                    node.penalty, query_l1 - 2 * node.penalty, node.count,
                    chosen != 0, score_now ? "score" : "split"
                });
            }

            if (score_now) {
                // Visit just the set bits, then use the same scorer as a full scan.
                for (std::size_t word = 0; word < node.active.size(); ++word) {
                    auto remaining = node.active[word];
                    while (remaining) {
                        const auto bit = static_cast<std::size_t>(std::countr_zero(remaining));
                        const auto row = word * 64 + bit;
                        candidates.offer(bucket.rows[row], table.score(bucket.codes.data() + row * code_words_));
                        ++result.stats.documents_scored;
                        remaining &= remaining - 1; // Clear the bit we just used.
                    }
                }
                continue;
            }

            // Split on one sign. Both children stay within this node's active rows.
            const auto dimension = dimension_order[node.depth];
            const auto* plane = bucket.planes.data() + dimension * bucket.bitmap_words;
            const bool prefer_one = query[dimension] >= 0;
            std::vector<std::uint64_t> matching(bucket.bitmap_words);
            std::vector<std::uint64_t> opposite(bucket.bitmap_words);
            std::size_t matching_count = 0;
            for (std::size_t word = 0; word < bucket.bitmap_words; ++word) {
                const auto preferred = prefer_one ? plane[word] : ~plane[word];
                matching[word] = node.active[word] & preferred;
                opposite[word] = node.active[word] & ~preferred;
                matching_count += std::popcount(matching[word]);
            }
            result.stats.bitplane_words += bucket.bitmap_words;
            if (matching_count) {
                frontier.push({node.bucket, node.depth + 1, matching_count, node.penalty,
                               next_order++, std::move(matching)});
            }
            if (matching_count < node.count) {
                frontier.push({node.bucket, node.depth + 1, node.count - matching_count,
                               node.penalty + std::abs(double(query[dimension])),
                               next_order++, std::move(opposite)});
            }
        }
        candidates.write_to(result);
        result.stats.elapsed_ms = milliseconds_since(start);
    }
    return results;
}

StorageInfo Index::info() const {
    StorageInfo result;
    result.documents = documents_;
    result.dimensions = dimensions_;
    result.buckets = buckets_.size();
    for (const auto& bucket : buckets_) {
        result.codes_bytes += bucket.codes.size() * sizeof(std::uint64_t);
        result.bitplanes_bytes += bucket.planes.size() * sizeof(std::uint64_t);
        result.row_ids_bytes += bucket.rows.size() * sizeof(std::int64_t);
    }
    return result;
}

} // namespace bitplane
