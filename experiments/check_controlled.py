"""Check the one-path formulas against completed controlled-run counters."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.analysis import load_cases
from experiments.controlled import predict_one_path


def key_shell_work(shell_counts, top_k, key_bits):
    """Equal strong weights order keys by their number of mismatching bits."""
    documents = 0
    keys = 0
    for errors, count in enumerate(shell_counts):
        documents += int(count)
        keys += math.comb(key_bits, errors)
        if documents >= top_k:
            return dict(keys_generated=keys, key_attempts=keys, documents_scored=documents)
    return None


def predicted_work(case, metadata, counts, shell_counts=None):
    """Return a prediction only for the dominance case derived in the paper."""
    setup = metadata['identity']
    strong_bits = setup['strong_bits']
    weak_total = (setup['dimensions'] - strong_bits) * setup['weak_weight']
    if weak_total >= 1 or case['router'] is not None:
        return None
    if (case['method'] == 'keys' and setup['support'] == 'fixed'
            and case['key_offset'] == 0 and case['key_bits'] == strong_bits
            and case['candidate_target'] == 0 and shell_counts is not None):
        predicted = key_shell_work(shell_counts, case['top_k'], strong_bits)
        if predicted is None or (case['key_limit'] and predicted['key_attempts'] > case['key_limit']):
            return None
        predicted['score_terms'] = predicted['documents_scored'] * ((case['dimensions'] + 3) // 4)
        predicted['recall'] = 1.0
        return predicted
    if counts[-1] < case['top_k']:
        return None
    if case['method'] == 'branch' and case['node_budget'] == 0:
        predicted = predict_one_path(counts, case['leaf_size'], case['top_k'], case['dimensions'])
        if predicted is None:
            return None
        depth = predicted['nodes'] - 1
        words = (counts[0] + 63) // 64
        # Every alternate child is nonempty in this model's counted path.
        if any(counts[j] == counts[j + 1] for j in range(depth)):
            return None
        predicted['peak_mask_bytes'] = (depth + 2 if depth else 1) * words * 8
        predicted['recall'] = 1.0
        return predicted
    if (case['method'] == 'keys' and setup['support'] == 'fixed'
            and case['key_offset'] == 0 and case['key_bits'] <= strong_bits):
        # The first key contains every document matching all strong bits. There are at least K,
        # so after scoring that posting, the bound rejects the next strong error.
        scored = counts[case['key_bits']]
        return dict(keys_generated=1, key_attempts=1, documents_scored=scored,
                    score_terms=scored * ((case['dimensions'] + 3) // 4), recall=1.0)
    return None


def check(folder):
    cases, failed = load_cases(folder)
    checked, skipped, mismatches = [], 0, []
    pools = {}
    # The metadata and count arrays were written from document signs during
    # preparation, before native indexing or timing.
    for item in cases:
        case = item['case']
        pool = Path(case['pool'])
        if str(pool) not in pools:
            metadata = json.loads((pool / 'pool.json').read_text())
            counts = np.load(pool / 'preferred_counts.npy').tolist()
            shells = None
            if metadata['identity']['support'] == 'fixed':
                bits = metadata['identity']['strong_bits']
                codes = np.load(pool / 'codes.npy', mmap_mode='r')
                histogram = np.bincount(codes[:, 0].astype(np.int64) & ((1 << bits) - 1), minlength=1 << bits)
                queries = np.load(pool / 'queries.npy')
                shells = []
                for query in queries:
                    ideal = sum(1 << i for i in range(bits) if query[i] >= 0)
                    distances = [int(key ^ ideal).bit_count() for key in range(1 << bits)]
                    shells.append(np.bincount(distances, weights=histogram, minlength=bits + 1).astype(np.int64).tolist())
            pools[str(pool)] = (metadata, counts, shells)
        metadata, counts_by_query, shells = pools[str(pool)]
        for row in item['queries']:
            if row['repetition'] != 0:
                continue
            predicted = predicted_work(case, metadata, counts_by_query[row['query']],
                                       None if shells is None else shells[row['query']])
            if predicted is None:
                skipped += 1
                continue
            measured = {field: row[field] for field in predicted if field != 'score_terms'}
            measured['score_terms'] = row['documents_scored'] * ((case['dimensions'] + 3) // 4)
            result = dict(case_id=item['case_id'], query=row['query'], method=case['method'],
                          documents=case['documents'], predicted=predicted, measured=measured)
            checked.append(result)
            if predicted != measured:
                mismatches.append(result)
    evidence = dict(checked=len(checked), outside_stated_conditions=skipped,
                    failed_cases=failed, mismatches=mismatches, checks=checked,
                    scope='First repetition per query. Exact counts under stated dominance/enough-match conditions; no latency fit and no deletion of other queries from the study.')
    (folder / 'work-checks.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f"Checked {len(checked)} predictions; {skipped} outside the stated conditions; {len(mismatches)} mismatches.")
    if mismatches:
        raise AssertionError('work prediction disagrees with native counters; see work-checks.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    check(parser.parse_args().folder)


if __name__ == '__main__':
    main()
