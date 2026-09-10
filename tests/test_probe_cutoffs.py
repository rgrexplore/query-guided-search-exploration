import numpy as np

from experiments.probe_cutoffs import probe_cutoff, cover_boundary_ties


def test_probe_cutoff_counts_neighbors_including_shared_clusters():
    ranks=np.array([[1,1,3],[2,4,4]])
    assert probe_cutoff(ranks,2/3)==3
    assert probe_cutoff(ranks,1)==4
    assert probe_cutoff(ranks,1/6)==1


def test_cutoff_includes_complete_score_ties_for_every_query():
    scores=np.array([[4,3,3,1],[5,4,3,3]],dtype=np.float32)
    assert cover_boundary_ties(scores,1)==1
    assert cover_boundary_ties(scores,2)==4
