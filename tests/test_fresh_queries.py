from experiments.fresh_queries import select_fresh


def test_fresh_selection_excludes_old_ids_and_preserves_text_alignment():
    ids = ['old', 'a', 'b', 'c']
    texts = ['old text', 'text a', 'text b', 'text c']
    chosen_ids, chosen_texts = select_fresh(ids, texts, {'old', 'b'}, 2)
    assert chosen_ids == ['a', 'c']
    assert chosen_texts == ['text a', 'text c']
