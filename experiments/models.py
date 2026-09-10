"""Storage equations from the paper, with units kept explicit.

These return logical payload bytes. Allocator overhead and peak process RAM
are separate measurements, not hidden constants added to these equations.
"""
import math

import numpy as np
from scipy.optimize import nnls


def packed_code_bytes(documents, dimensions, word_bits=64):
    """Paper, common layout: every document occupies whole packed words."""
    words_per_document = (dimensions + word_bits - 1) // word_bits
    return documents * words_per_document * (word_bits // 8)


def bitplane_bytes(cluster_sizes, dimensions, word_bits=64):
    """Paper, method B: pad each cluster's bitmap separately, once per bit."""
    bitmap_words = sum((size + word_bits - 1) // word_bits for size in cluster_sizes)
    return dimensions * bitmap_words * (word_bits // 8)


def key_directory_bytes(occupied_keys, key_bytes=4, offset_bytes=8):
    """Paper, method C: key plus posting start/count, for occupied keys only.

    Rows are reordered with their codes. Their original IDs are the postings,
    so this implementation needs no additional N-row posting permutation.
    """
    return occupied_keys * (key_bytes + 2 * offset_bytes)


def cost_features(row, method):
    """Measured work used by the timing model; these are not CPU instructions."""
    features = {'fixed': 1.0, 'score_terms': row['documents_scored']*((row['dimensions']+3)//4)}
    if method == 'branch':
        features['bitmap_words'] = row['bitplane_words'] + row['leaf_words']
        features['nodes'] = row['nodes']
    elif method == 'keys':
        features['key_attempts'] = row['key_attempts']
        # Queue depth grows with enumeration. This is an approximate work term,
        # not a claim that every key performs exactly this many comparisons.
        keys = row['keys_generated']
        features['key_queue_work'] = keys*math.log2(keys+2)
    return features


def fit_cost_model(rows, method, loss='relative'):
    """Fit nonnegative combined costs on calibration rows only, in milliseconds."""
    names = list(cost_features(rows[0], method))
    values = np.array([list(cost_features(row, method).values()) for row in rows], dtype=float)
    observed = np.array([row['search_ms'] for row in rows])
    # The acceptance criterion is percentage error. Weight each residual by
    # 1/time so a 100 ms case does not overwhelm a 0.1 ms case solely by scale.
    if loss == 'relative':
        row_weights = 1/observed
    elif loss == 'absolute':
        row_weights = np.ones(len(observed))
    else:
        raise ValueError('loss must be relative or absolute')
    scales = np.maximum(np.linalg.norm(values*row_weights[:, None], axis=0), 1)
    normalized = values/scales
    weights, _ = nnls(normalized*row_weights[:, None], observed*row_weights)
    rank = int(np.linalg.matrix_rank(normalized))
    return {'method': method, 'loss': loss, 'coefficients': dict(zip(names, (weights/scales).tolist())),
            'training_rows': len(rows), 'matrix_rank': rank, 'identifiable': rank == len(names),
            'target': 'search_ms',
            'scope': 'API time estimated from measured work; coefficients combine compute, memory and selection costs.'}


def predict_search_ms(model, row):
    features = cost_features(row, model['method'])
    return sum(model['coefficients'][name]*value for name, value in features.items())


def mean_recall(values, top_k):
    """Aggregate individual query/seed recalls through their integer hit counts.

    Each input is recovered_neighbors / top_k. Dividing once at the end avoids
    turning an exact 99% total into 0.9899999999999999 during averaging.
    """
    return sum(round(float(value)*top_k) for value in values)/(len(values)*top_k)


def predict_native_cost(model, row):
    """Evaluate the saved complete-call model for its declared workload."""
    if model['kind']=='score_and_selection':
        count=row['documents_scored']
        return (model['fixed_ms']+count*model['per_scored_row_ms']
                +model['per_log_ratio_ms']*np.log(max(count/model['top_k'],1)))
    return model['fixed_ms']+model['per_million_ms']*row['documents']/1000000
