#pragma once

#include "index.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>

namespace bitplane::detail {

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

inline bool cannot_improve(double best_penalty, double query_l1, std::size_t dimensions,
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


} // namespace bitplane::detail
