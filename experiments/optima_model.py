"""Small analytic models for the declared random-sign experiments.

Documents have independent balanced signs. Strong query weights exceed the
sum of all weak weights, so a document matching every strong sign outranks
every document with a strong mismatch. These assumptions do not describe
arbitrary real embeddings or provide a general recall estimate for IVF.
"""
import math

import numpy as np
from scipy.stats import binom, hypergeom


def ideal_group_stats(documents, strong_bits, top_k, sd_margin=3.0):
    """Count documents matching all strong signs, including the empty cases.

    Expected recall is for searching ONLY that group and scoring its rows
    exactly. Unlimited search may examine other groups and recover more.
    The suggested leaf is a mean plus a spread allowance, not a guarantee.
    """
    match_probability = 2.0**-strong_bits
    mean = documents * match_probability
    std = math.sqrt(mean * (1 - match_probability))
    # E[min(Z, K)] = sum_{i=1}^K P(Z >= i). Using min(E[Z], K)
    # instead would hide the recall lost when the group has too few rows.
    positions = np.arange(1, top_k + 1)
    expected_recall = float(binom.sf(positions - 1, documents, match_probability).sum() / top_k)
    return {
        'mean_count': mean,
        'std_count': std,
        'probability_fewer_than_k': float(binom.cdf(top_k - 1, documents, match_probability)),
        'expected_ideal_only_recall': expected_recall,
        'suggested_leaf_size': math.ceil(mean + sd_margin * std),
    }


def expected_occupied_keys(documents, key_bits):
    """Expected nonempty cells when each of 2**key_bits keys is equally likely.

    For disjoint prefix routing and local key bits, their sum gives the width
    of the combined (cluster, key) partition. Do not add overlapping bits.
    """
    if key_bits == 0:
        return float(documents > 0)
    cells = 2.0**key_bits
    # log1p/expm1 keep the subtraction accurate for a sparse key space.
    return -cells * math.expm1(documents * math.log1p(-1 / cells))


def fixed_coordinate_stats(dimensions, strong_bits, key_bits):
    """Overlap between a fixed key and uniformly changing strong positions.

    If s strong coordinates fall in the key, retaining every key pattern
    that matches them retains an expected 2**-s fraction of documents. The
    compatible-key count includes every combination of the remaining weak
    signs. Actual bound-based search can sometimes stop earlier. These are
    work predictions for that declared policy, not measured recall values.
    """
    captured = np.arange(min(strong_bits, key_bits) + 1)
    probabilities = hypergeom.pmf(captured, dimensions, strong_bits, key_bits)
    retained_fraction = float(np.sum(probabilities * 2.0**-captured))
    return {
        'probability_zero_overlap': float(hypergeom.pmf(0, dimensions, strong_bits, key_bits)),
        'expected_retained_fraction': retained_fraction,
        'expected_compatible_keys': 2.0**key_bits * retained_fraction,
    }


def branch_depth_time(cluster_documents, depth, split_word_ms, leaf_word_ms,
                      score_row_ms, fixed_ms=0.0, word_bits=64, fixed_prefix_bits=0):
    """Expected milliseconds for one preferred path followed by one leaf.

    After fixed prefix coordinates, each useful split halves the expected
    row count. Every split and the leaf still read the original mask width.
    Today's C++ traversal visits prefix-fixed coordinates too; they cost a
    split without removing rows when they occur first in the query order.

    This omits changing queue overhead and assumes enough matching rows,
    dominant strong weights, and no binding node budget. It cannot predict
    an arbitrary multi-branch search. Prices must describe separate work.
    """
    words = math.ceil(cluster_documents / word_bits)
    useful_splits = max(depth - fixed_prefix_bits, 0)
    scored_rows = cluster_documents * 2.0**-useful_splits
    return (fixed_ms + depth * words * split_word_ms
            + words * leaf_word_ms + scored_rows * score_row_ms)


def branch_depth_optimum(cluster_documents, split_word_ms, score_row_ms,
                         max_depth, word_bits=64, fixed_prefix_bits=0):
    """Integer candidates around the smooth one-path optimum.

    Both prices are positive. The caller chooses max_depth from the valid
    score/recall assumptions, then checks these depths against actual work.
    Zero is retained because traversing fixed prefix coordinates may cost
    more than the later useful cuts save. This is not a full ANN optimizer.
    """
    words = math.ceil(cluster_documents / word_bits)
    continuous = fixed_prefix_bits + math.log2(
        cluster_documents * math.log(2) * score_row_ms / (words * split_word_ms)
    )
    proposed = [0, max_depth, fixed_prefix_bits, math.floor(continuous), math.ceil(continuous)]
    candidates = sorted({min(max(int(depth), 0), max_depth) for depth in proposed})
    return {'unconstrained_depth': continuous, 'candidate_depths': candidates}
