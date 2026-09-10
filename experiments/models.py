"""Storage equations from the paper, with units kept explicit.

These return logical payload bytes. Allocator overhead and peak process RAM
are separate measurements, not hidden constants added to these equations.
"""


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
