import numpy as np
from experiments.summarize_native_scaling import fit_median_scaling


def test_median_fit_does_not_overweight_a_fast_query():
    rows=[dict(documents=n,query_ms=t*n/1000000)
          for n in (1000000,2000000) for t in (1,10,10)]
    coefficients=fit_median_scaling(rows)
    assert np.isclose(coefficients @ [1,4],40)


def test_median_fit_keeps_uniform_queries_correct():
    rows=[dict(documents=n,query_ms=10*n/1000000)
          for n in (1000000,2000000) for _ in range(3)]
    coefficients=fit_median_scaling(rows)
    assert np.isclose(coefficients @ [1,4],40)
