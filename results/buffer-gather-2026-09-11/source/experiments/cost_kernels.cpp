// Small measurements of the operations used by the search index.
// This measures buffers and score loops, not retrieval recall on a large corpus.
#include "score.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Word = std::uint64_t;
using Clock = std::chrono::steady_clock;

struct Split {
    std::vector<Word> matching;
    std::vector<Word> opposite;
    std::size_t matching_count;
};

// Keep this out of the caller so the compiler cannot specialize the loop to
// the generated inputs. Both output masks are checked after the timed call.
__attribute__((noinline)) Split split_masks(const std::vector<Word>& active,
                                          const Word* plane) {
    Split result{std::vector<Word>(active.size()), std::vector<Word>(active.size()), 0};
    for (std::size_t word = 0; word < active.size(); ++word) {
        result.matching[word] = active[word] & plane[word];
        result.opposite[word] = active[word] & ~plane[word];
        result.matching_count += std::popcount(result.matching[word]);
    }
    return result;
}

struct Leaf {
    std::size_t count = 0;
    Word row_sum = 0;
};

__attribute__((noinline)) Leaf enumerate_leaf(const std::vector<Word>& active) {
    Leaf result;
    for (std::size_t word = 0; word < active.size(); ++word) {
        auto remaining = active[word];
        while (remaining) {
            const auto bit = std::countr_zero(remaining);
            result.row_sum += word * 64 + bit;
            ++result.count;
            remaining &= remaining - 1;
        }
    }
    return result;
}

__attribute__((noinline)) bitplane::QueryResult score_rows(
        const std::vector<Word>& codes, const std::vector<std::int64_t>& ids,
        const std::vector<float>& query, std::size_t top_k) {
    bitplane::detail::ScoreTable table(query.data(), query.size());
    bitplane::detail::TopCandidates best(top_k);
    const auto words = (query.size() + 63) / 64;
    for (std::size_t row = 0; row < ids.size(); ++row) {
        best.offer(ids[row], table.score(codes.data() + row * words));
    }
    bitplane::QueryResult result;
    best.write_to(result);
    return result;
}

// This is the leaf loop from Index::search, including scattered code/ID reads.
__attribute__((noinline)) bitplane::QueryResult score_leaf(
        const std::vector<Word>& codes, const std::vector<std::int64_t>& ids,
        const std::vector<float>& query, const std::vector<Word>& active, std::size_t top_k) {
    bitplane::detail::ScoreTable table(query.data(), query.size());
    bitplane::detail::TopCandidates best(top_k);
    const auto words = (query.size() + 63) / 64;
    for (std::size_t word = 0; word < active.size(); ++word) {
        auto remaining = active[word];
        while (remaining) {
            const auto row = word * 64 + std::countr_zero(remaining);
            best.offer(ids[row], table.score(codes.data() + row * words));
            remaining &= remaining - 1;
        }
    }
    bitplane::QueryResult result;
    best.write_to(result);
    return result;
}

void self_test() {
    const std::vector<Word> active{0x97, 0xf0}, plane{0x65, 0x3c};
    const auto split = split_masks(active, plane.data());
    std::size_t count = 0;
    Word row_sum = 0;
    for (std::size_t row = 0; row < active.size() * 64; ++row) {
        const Word flag = Word{1} << (row % 64);
        const bool match = (active[row / 64] & flag) && (plane[row / 64] & flag);
        if (bool(split.matching[row / 64] & flag) != match) throw std::runtime_error("split mismatch");
        if ((split.matching[row / 64] | split.opposite[row / 64]) != active[row / 64])
            throw std::runtime_error("split lost a row");
        if (match) { ++count; row_sum += row; }
    }
    const auto leaf = enumerate_leaf(split.matching);
    if (leaf.count != count || leaf.row_sum != row_sum || count != split.matching_count)
        throw std::runtime_error("leaf mismatch");

    const std::vector<float> query{.7f, -.4f, .2f, -.1f, .3f};
    const std::vector<Word> codes{3, 7, 11, 19, 23, 3};
    std::vector<std::int64_t> ids(codes.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::pair<double, std::int64_t>> reference;
    for (std::size_t row = 0; row < codes.size(); ++row) {
        double score = 0;
        for (std::size_t bit = 0; bit < query.size(); ++bit)
            score += ((codes[row] >> bit) & 1) ? double(query[bit]) : -double(query[bit]);
        reference.emplace_back(-score, ids[row]);
    }
    std::sort(reference.begin(), reference.end());
    const auto ranked = score_rows(codes, ids, query, 3);
    for (std::size_t row = 0; row < ranked.rows.size(); ++row)
        if (ranked.rows[row] != reference[row].second || ranked.scores[row] != -reference[row].first)
            throw std::runtime_error("score mismatch");
    const auto selected = score_leaf(codes, ids, query, std::vector<Word>{0b010101}, 2);
    reference.erase(std::remove_if(reference.begin(), reference.end(), [](const auto& entry) {
        return (entry.second % 2) != 0;
    }), reference.end());
    for (std::size_t row = 0; row < selected.rows.size(); ++row)
        if (selected.rows[row] != reference[row].second) throw std::runtime_error("gather mismatch");
    std::cout << "split, leaf, contiguous and gathered scorer checks passed\n";
}

void measure_path(std::size_t words, std::size_t depth, int repeats, Word seed) {
    std::mt19937_64 random(seed);
    std::vector<Word> planes(words * depth);
    for (auto& value : planes) value = random();
    std::cout << "operation,size,dimensions,repetition,step,milliseconds,count,live_payload_bytes,checksum\n";
    // The first whole path warms up the allocator and is not reported.
    for (int repetition = -1; repetition < repeats; ++repetition) {
        auto root_start = Clock::now();
        std::vector<Word> active(words, ~Word{0});
        const auto root_ms = std::chrono::duration<double, std::milli>(Clock::now() - root_start).count();
        std::vector<std::vector<Word>> pending;
        pending.reserve(depth);
        std::vector<double> times;
        std::vector<std::size_t> counts;
        for (std::size_t step = 0; step < depth; ++step) {
            const auto start = Clock::now();
            auto split = split_masks(active, planes.data() + step * words);
            pending.push_back(std::move(split.opposite));
            active = std::move(split.matching); // Also release the previous parent.
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            counts.push_back(split.matching_count);
        }
        const auto leaf_start = Clock::now();
        const auto leaf = enumerate_leaf(active);
        const auto leaf_ms = std::chrono::duration<double, std::milli>(Clock::now() - leaf_start).count();

        // Independently verify that the surviving path and all alternatives
        // partition the original rows. This pass is outside the query timings.
        std::size_t total = 0;
        for (std::size_t word = 0; word < words; ++word) {
            Word covered = active[word];
            total += std::popcount(active[word]);
            for (const auto& mask : pending) {
                if (covered & mask[word]) throw std::runtime_error("overlapping children");
                covered |= mask[word];
                total += std::popcount(mask[word]);
            }
            if (covered != ~Word{0}) throw std::runtime_error("missing rows");
        }
        if (total != words * 64 || leaf.count != counts.back()) throw std::runtime_error("count mismatch");
        const auto free_start = Clock::now();
        pending.clear();
        std::vector<Word>().swap(active);
        const auto free_ms = std::chrono::duration<double, std::milli>(Clock::now() - free_start).count();
        if (repetition >= 0) {
            const auto plane_bytes = words * depth * sizeof(Word);
            std::cout << "root," << words << ",0," << repetition << ",0," << root_ms << ",0,"
                      << plane_bytes + words * 8 << ",0\n";
            for (std::size_t step = 0; step < depth; ++step)
                std::cout << "split," << words << ",0," << repetition << ',' << step + 1 << ',' << times[step]
                          << ',' << counts[step] << ',' << plane_bytes + (step + 3) * words * 8 << ",0\n";
            std::cout << "leaf," << words << ",0," << repetition << ',' << depth << ',' << leaf_ms
                      << ',' << leaf.count << ',' << plane_bytes + (depth + 1) * words * 8 << ',' << leaf.row_sum << '\n';
            std::cout << "release," << words << ",0," << repetition << ',' << depth << ',' << free_ms
                      << ",0," << plane_bytes << ",0\n";
        }
    }
}

void measure_scores(std::size_t documents, std::size_t dimensions, std::size_t top_k, int repeats, Word seed) {
    std::mt19937_64 random(seed);
    std::vector<Word> codes(documents * ((dimensions + 63) / 64));
    std::vector<std::int64_t> ids(documents);
    std::vector<float> query(dimensions);
    for (auto& code : codes) code = random();
    std::iota(ids.begin(), ids.end(), 0);
    std::normal_distribution<float> normal;
    for (auto& value : query) value = normal(random);
    const auto bytes = codes.size() * 8 + ids.size() * 8 + query.size() * 4;
    std::cout << "operation,size,dimensions,repetition,step,milliseconds,count,live_payload_bytes,checksum\n";
    for (int repetition = -1; repetition < repeats; ++repetition) {
        const auto start = Clock::now();
        const auto result = score_rows(codes, ids, query, std::min(top_k, documents));
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        Word checksum = 0;
        for (const auto row : result.rows) checksum += row;
        if (repetition >= 0)
            std::cout << "score," << documents << ',' << dimensions << ',' << repetition << ",0,"
                      << elapsed << ',' << documents << ',' << bytes << ',' << checksum << '\n';
    }
}

void measure_gather(std::size_t documents, std::size_t dimensions, std::size_t depth,
                    std::size_t top_k, int repeats, Word seed) {
    std::mt19937_64 random(seed);
    std::vector<Word> codes(documents * ((dimensions + 63) / 64));
    std::vector<std::int64_t> ids(documents);
    std::vector<float> query(dimensions);
    std::vector<Word> active((documents + 63) / 64, ~Word{0});
    for (auto& code : codes) code = random();
    std::iota(ids.begin(), ids.end(), 0);
    std::normal_distribution<float> normal;
    for (auto& value : query) value = normal(random);
    for (auto& mask : active)
        for (std::size_t bit = 0; bit < depth; ++bit) mask &= random();
    if (documents % 64) active.back() &= (Word{1} << (documents % 64)) - 1;
    std::size_t found = 0;
    for (const auto mask : active) found += std::popcount(mask);
    const auto bytes = codes.size() * 8 + ids.size() * 8 + active.size() * 8 + query.size() * 4;
    std::cout << "operation,size,dimensions,repetition,step,milliseconds,count,live_payload_bytes,checksum\n";
    for (int repetition = -1; repetition < repeats; ++repetition) {
        const auto start = Clock::now();
        const auto result = score_leaf(codes, ids, query, active, top_k);
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (result.rows.size() != std::min(top_k, found)) throw std::runtime_error("gather count mismatch");
        Word checksum = 0;
        for (const auto row : result.rows) {
            if (!(active[row / 64] & (Word{1} << (row % 64)))) throw std::runtime_error("row outside leaf");
            checksum += row;
        }
        if (repetition >= 0)
            std::cout << "gather," << documents << ',' << dimensions << ',' << repetition << ',' << depth << ','
                      << elapsed << ',' << found << ',' << bytes << ',' << checksum << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        if (argc < 2) throw std::invalid_argument("choose path or score");
        const std::string operation = argv[1];
        const auto expected_argc = operation == "path" ? 6 : operation == "score" ? 7 : 8;
        if (argc != expected_argc || (operation != "path" && operation != "score" && operation != "gather"))
            throw std::invalid_argument("usage: path WORDS DEPTH REPEATS SEED | score ROWS DIMENSIONS K REPEATS SEED | gather ROWS DIMENSIONS DEPTH K REPEATS SEED");
        const auto size = std::stoull(argv[2]);
        const auto parameter = std::stoull(argv[3]);
        const auto repeats = std::stoi(argv[argc - 2]);
        const auto seed = std::stoull(argv[argc - 1]);
        if (size == 0 || parameter == 0 || repeats <= 0) throw std::invalid_argument("sizes and repeats must be positive");
        std::cout.precision(12);
        if (operation == "path") measure_path(size, parameter, repeats, seed);
        else {
            const auto top_k = std::stoull(argv[operation == "score" ? 4 : 5]);
            if (top_k == 0) throw std::invalid_argument("K must be positive");
            if (operation == "score") measure_scores(size, parameter, top_k, repeats, seed);
            else measure_gather(size, parameter, std::stoull(argv[4]), top_k, repeats, seed);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
