import numpy as np
import pytest

import predict
from predict import select_top_predictions, tokenize_for_prediction, validate_model_tokenizer_compatibility


def test_tokenize_for_prediction_handles_empty_string():
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3]]

    try:
        tokenize_for_prediction("", DummyTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("Empty input should raise ValueError")


def test_tokenize_for_prediction_returns_proper_shape():
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3, 4]]

    arr = tokenize_for_prediction("hello world", DummyTokenizer())
    assert arr.shape == (1, 10)


def test_select_top_predictions_returns_unique_valid_words():
    class DummyTokenizer:
        index_word = {1: "hello", 2: "world", 3: "<OOV>", 4: "<PAD>", 5: "python"}

    probabilities = np.array([0.0, 0.5, 0.2, 0.1, 0.1, 0.7])
    results = select_top_predictions(probabilities, DummyTokenizer(), top_k=5)
    assert [item["word"] for item in results] == ["python", "hello", "world"]
    assert len(results) == 3


def test_validate_model_tokenizer_compatibility_accepts_valid_pair():
    class DummyTokenizer:
        word_index = {"<OOV>": 1, "text": 2, "python": 13000}
        num_words = 12000

    class DummyModel:
        output_shape = (None, 12000)

    validate_model_tokenizer_compatibility(DummyModel(), DummyTokenizer())


def test_validate_model_tokenizer_compatibility_rejects_vocab_limit_mismatch():
    class DummyTokenizer:
        word_index = {"<OOV>": 1, "text": 2}
        num_words = 100

    class DummyModel:
        output_shape = (None, 12000)

    try:
        validate_model_tokenizer_compatibility(DummyModel(), DummyTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("A tokenizer/model vocabulary-limit mismatch should raise ValueError")


def test_validate_dynamic_tokenizer_includes_padding_and_all_token_ids():
    class DynamicTokenizer:
        word_index = {"<OOV>": 1, "alpha": 2, "beta": 3}
        num_words = None

    class MatchingModel:
        output_shape = (None, 4)

    class MismatchedModel:
        output_shape = (None, 12000)

    validate_model_tokenizer_compatibility(MatchingModel(), DynamicTokenizer())
    try:
        validate_model_tokenizer_compatibility(MismatchedModel(), DynamicTokenizer())
    except ValueError as exc:
        assert "effective vocabulary size (4" in str(exc)
    else:
        raise AssertionError("A dynamic tokenizer/model vocabulary mismatch should raise ValueError")


class DummyGenerationTokenizer:
    word_index = {"<OOV>": 1, "this": 2, "demo": 3, "is": 4, "useful": 5, ".": 6}
    index_word = {value: key for key, value in word_index.items()}
    num_words = None

    def texts_to_sequences(self, texts):
        return [
            [self.word_index[token] for token in text.split() if token in self.word_index]
            for text in texts
        ]


class DummyGenerationModel:
    output_shape = (None, 7)

    def __init__(self):
        self.calls = []

    def predict(self, inputs, verbose=0):
        self.calls.append(inputs.copy())
        return np.asarray([[0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1]])


def patch_generation_assets(monkeypatch):
    tokenizer = DummyGenerationTokenizer()
    model = DummyGenerationModel()
    monkeypatch.setattr(predict, "load_tokenizer", lambda artifact_set: tokenizer)
    monkeypatch.setattr(predict, "load_model", lambda model_name, artifact_set: model)
    monkeypatch.setattr(predict, "validate_model_tokenizer_compatibility", lambda model, tokenizer: None)
    return tokenizer, model


@pytest.mark.parametrize("strategy", ["greedy", "top_k", "temperature"])
def test_continue_sentence_uses_model_probabilities_and_requested_token_count(monkeypatch, strategy):
    _, model = patch_generation_assets(monkeypatch)
    result = predict.continue_sentence(
        "This demo",
        model_name="lstm",
        additional_words=3,
        artifact_set="demo",
        strategy=strategy,
        temperature=0.8,
        top_k=3,
        repetition_penalty=1.2,
    )

    assert result.startswith("This demo ")
    assert len(model.calls) == 3


def test_repetition_penalty_discourages_repeats_without_banning_them():
    tokenizer = DummyGenerationTokenizer()
    probabilities = np.asarray([0.0, 0.0, 0.0, 0.0, 0.65, 0.35, 0.0])

    penalized = predict._apply_repetition_penalty(
        probabilities,
        tokenizer,
        recent_token_ids=[4],
        history_token_ids=[2, 4],
        penalty=2.0,
    )
    assert penalized[4] > 0
    assert penalized[4] < probabilities[4]
    assert penalized[5] > penalized[4]


def test_top_k_sampling_only_samples_from_ranked_candidates(monkeypatch):
    tokenizer = DummyGenerationTokenizer()
    sampled = {}

    def fake_choice(candidates, p):
        sampled["candidates"] = list(candidates)
        sampled["probabilities"] = p
        return candidates[-1]

    monkeypatch.setattr(predict.np.random, "choice", fake_choice)
    selected = predict._choose_next_token(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.6, 0.3, 0.1]),
        tokenizer,
        strategy="top_k",
        temperature=1.0,
        top_k=3,
        repetition_penalty=1.0,
        recent_token_ids=[],
        history_token_ids=[],
    )

    assert selected in sampled["candidates"]
    assert len(sampled["candidates"]) == 3
    assert np.isclose(sampled["probabilities"].sum(), 1.0)


def test_repeated_short_phrase_is_downweighted():
    tokenizer = DummyGenerationTokenizer()
    probabilities = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3])
    penalized = predict._apply_repetition_penalty(
        probabilities,
        tokenizer,
        recent_token_ids=[],
        history_token_ids=[2, 5, 2],
        penalty=1.5,
    )
    assert penalized[5] < probabilities[5]


def test_generation_rejects_invalid_decoding_strategy(monkeypatch):
    patch_generation_assets(monkeypatch)
    with pytest.raises(ValueError, match="strategy"):
        predict.continue_sentence("This demo", strategy="beam")
