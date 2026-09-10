import numpy as np
from experiments.components import pack_signs, reference_rows
from experiments.native_scaling import make_queries, binary_reference


def test_count_reference_matches_independent_full_scores():
    signs=np.random.default_rng(7).integers(0,2,(257,80),dtype=np.uint8)
    codes=pack_signs(signs)
    for mode in ('fixed','adaptive'):
        queries,supports=make_queries(80,3,13,mode,197)
        truth=reference_rows(signs,queries,5)
        for row,(query,support) in enumerate(zip(queries,supports,strict=True)):
            actual,count=binary_reference(codes,query,support,80,5,chunk_size=64)
            np.testing.assert_array_equal(actual,truth[row])
            assert count==int(np.sum(np.all(signs[:,support]==(query[support]>=0),axis=1)))
