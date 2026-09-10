from experiments.controlled import predict_one_path


def test_one_path_prediction_keeps_full_width_leaf_work():
    predicted=predict_one_path([1024,512,256,128],128,100,256)
    assert predicted==dict(nodes=4,bitplane_words=48,leaf_words=16,documents_scored=128,score_terms=8192)
    assert predict_one_path([1024,512,256,80],128,100,256) is None
    assert predict_one_path([1024,512,256,128],64,100,256) is None
