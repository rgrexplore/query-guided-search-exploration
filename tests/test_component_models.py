"""Small byte counts that can be checked without an index implementation."""
from experiments.models import bitplane_bytes, packed_code_bytes, key_directory_bytes, mean_recall


def test_word_padding_is_counted_per_cluster_and_per_document():
    assert packed_code_bytes(65, 5) == 520
    assert packed_code_bytes(65, 65) == 1040
    assert bitplane_bytes([65], 5) == 80
    assert bitplane_bytes([32, 33], 5) == 80
    assert bitplane_bytes([64, 1, 1], 5) == 120


def test_key_directory_counts_only_payload_of_occupied_keys():
    assert key_directory_bytes(3, key_bytes=4, offset_bytes=8) == 60
    assert key_directory_bytes(0, key_bytes=4, offset_bytes=8) == 0


def test_recall_aggregation_preserves_the_exact_neighbor_count_threshold():
    assert mean_recall([.99]*100,100)==.99
    assert mean_recall([.98]*100,100)<.99
