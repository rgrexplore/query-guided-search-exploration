import numpy as np

from experiments.direct_routing import DirectPrefixRouter


def test_direct_prefix_routing_matches_exhaustive_weighted_pattern_scores():
    labels=np.array([0,2,5,7],dtype=np.int64)
    router=DirectPrefixRouter(labels,3)
    queries=np.array([[.7,-.2,.4],[0,0,0],[1,1,1]],dtype=np.float32)
    signs=np.array([[1 if code>>bit&1 else -1 for bit in range(3)] for code in labels])
    expected=[]
    for query in queries:
        scores=signs @ query.astype(np.float64)
        expected.append(labels[np.lexsort((labels,-scores))[:2]])
    np.testing.assert_array_equal(router.select(queries,2),expected)
