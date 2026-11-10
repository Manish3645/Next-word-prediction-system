import numpy as np
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer

import demo_v3_preprocess as demo_v3
from predict import tokenize_v3_for_prediction
from train import build_lstm_model, build_rnn_model


def _fit_tokens(texts):
    tokenizer = Tokenizer(num_words=None, oov_token="<OOV>", filters="", lower=False, split=" ")
    tokenizer.fit_on_texts([" ".join(demo_v3.tokenize_text(text)) for text in texts])
    return tokenizer


def test_unicode_is_nfc_normalized_and_tokenized_consistently():
    decomposed = "Cafe\u0301"
    composed = "Café"
    assert demo_v3.normalize_text(decomposed) == composed
    assert demo_v3.tokenize_text(decomposed) == demo_v3.tokenize_text(composed) == ["café"]


def test_apostrophes_are_normalized_and_contractions_remain_one_token():
    assert demo_v3.tokenize_text("I’m learning") == ["i'm", "learning"]
    assert demo_v3.tokenize_text("I’m learning") == demo_v3.tokenize_text("I'm learning")


def test_punctuation_hyphens_and_wikitext_markers_follow_explicit_policy():
    assert demo_v3.tokenize_text('Well-known, "next!"') == [
        "well", "-", "known", ",", '"', "next", "!", '"'
    ]
    assert demo_v3.tokenize_text("four @-@ wheel @,@ drive @.@") == [
        "four", "-", "wheel", ",", "drive", "."
    ]
    assert demo_v3.tokenize_text("word_part") == ["word", "_", "part"]


def test_training_encoding_and_inference_use_same_canonical_tokens():
    tokenizer = _fit_tokens(["Café isn’t well-known."])
    text = "Cafe\u0301 isn’t well-known."
    expected = demo_v3.encode_text(text, tokenizer)
    inferred = tokenize_v3_for_prediction(text, tokenizer)[0].tolist()
    assert inferred[-len(expected):] == expected


def test_selected_training_examples_have_no_unexpected_oov_targets():
    selected, _ = demo_v3.select_records_for_sequence_budget(
        ["Café isn’t well-known. More words here."],
        sequence_budget=6,
        seed=7,
    )
    tokenizer = demo_v3.build_v3_tokenizer(selected)
    _, targets, stats = demo_v3.build_arrays(selected, tokenizer, sequence_length=4)
    assert stats["oov_target_count"] == 0
    assert np.all(targets != tokenizer.word_index[tokenizer.oov_token])


def test_padding_and_vocabulary_dimensions_are_correct():
    selected, _ = demo_v3.select_records_for_sequence_budget(
        ["alpha beta gamma delta epsilon."],
        sequence_budget=4,
        seed=1,
    )
    tokenizer = demo_v3.build_v3_tokenizer(selected)
    x_values, y_values, _ = demo_v3.build_arrays(selected, tokenizer, sequence_length=5)
    vocab_size = demo_v3.effective_vocabulary_size(tokenizer)
    assert tokenizer.word_index[tokenizer.oov_token] == 1
    assert x_values[0, 0] == 0
    assert np.all(y_values > 0)
    assert x_values.max() < vocab_size
    assert y_values.max() < vocab_size
    rnn = build_rnn_model(vocab_size, mask_zero=True)
    lstm = build_lstm_model(vocab_size, mask_zero=True)
    assert rnn.output_shape[-1] == lstm.output_shape[-1] == vocab_size
    assert rnn.layers[0].mask_zero
    assert lstm.layers[0].mask_zero


def test_sequences_do_not_cross_sentence_or_record_boundaries():
    selected, _ = demo_v3.select_records_for_sequence_budget(
        ["alpha beta. gamma delta."],
        sequence_budget=4,
        seed=3,
    )
    tokenizer = demo_v3.build_v3_tokenizer(selected)
    x_values, y_values, _ = demo_v3.build_arrays(selected, tokenizer, sequence_length=10)
    assert len(y_values) == 4
    assert x_values.shape == (4, 10)
    gamma_id = tokenizer.word_index["gamma"]
    assert x_values[2, -1] == gamma_id
    assert np.count_nonzero(x_values[2]) == 1
    assert demo_v3.split_text_into_sentences("alpha beta. gamma delta.") == [
        "alpha beta.",
        "gamma delta.",
    ]


def test_seeded_selection_is_reproducible_and_exact_budget():
    records = [f"record{index} alpha beta gamma delta." for index in range(12)]
    first, first_stats = demo_v3.select_records_for_sequence_budget(records, 17, seed=42)
    second, second_stats = demo_v3.select_records_for_sequence_budget(records, 17, seed=42)
    assert first == second
    assert first_stats == second_stats
    assert sum(sum(len(sentence) - 1 for sentence in record["sentences"]) for record in first) == 17


def test_duplicate_text_is_excluded_across_official_split_selections():
    seen = {}
    train_records, _ = demo_v3.select_records_for_sequence_budget(
        ["shared alpha beta."],
        2,
        seed=1,
        seen_records=seen,
    )
    for key in list(seen):
        if seen[key] == "current":
            seen[key] = "train"
    valid_records, valid_stats = demo_v3.select_records_for_sequence_budget(
        ["shared alpha beta.", "valid only epsilon zeta."],
        4,
        seed=0,
        seen_records=seen,
    )
    assert len(train_records) == 1
    assert len(valid_records) == 1
    assert valid_stats["duplicates_from_other_splits_excluded"] == 1


def test_tokenizer_mapping_hash_is_stable():
    tokenizer = _fit_tokens(["alpha beta gamma"])
    assert demo_v3.tokenizer_mapping_sha256(tokenizer) == demo_v3.tokenizer_mapping_sha256(tokenizer)


@pytest.mark.parametrize("text", ["", "   "])
def test_v3_inference_rejects_empty_or_whitespace_input(text):
    tokenizer = _fit_tokens(["known words"])
    with pytest.raises(ValueError):
        tokenize_v3_for_prediction(text, tokenizer)
