import json
import pickle
import re
from functools import lru_cache

import numpy as np
import tensorflow as tf

from config import (
    DEFAULT_ARTIFACT_SET,
    DEMO_V3_METADATA_PATH,
    SEQUENCE_LENGTH,
    get_artifact_paths,
)
from preprocess import clean_text


@lru_cache(maxsize=2)
def load_tokenizer(artifact_set=DEFAULT_ARTIFACT_SET):
    tokenizer_path = get_artifact_paths(artifact_set)["tokenizer"]
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    with tokenizer_path.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=8)
def load_model(model_name, artifact_set=DEFAULT_ARTIFACT_SET):
    model_name = model_name.lower()
    paths = get_artifact_paths(artifact_set)
    if artifact_set == "demo_v3":
        experiment_id = model_name.upper()
        experiment_paths = paths.get("experiment_models", {})
        if experiment_id not in experiment_paths:
            raise ValueError("Demo V3 model_name must be one of 'A', 'B', 'C', or 'D'.")
        path = experiment_paths[experiment_id]
    else:
        if model_name not in {"rnn", "lstm"}:
            raise ValueError("model_name must be 'rnn' or 'lstm'.")
        path = paths[f"{model_name}_model"]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)


def validate_model_tokenizer_compatibility(model, tokenizer):
    model_vocab_size = int(model.output_shape[-1])
    token_ids = list(tokenizer.word_index.values())
    if not token_ids:
        raise ValueError("Tokenizer has no fitted vocabulary. Run python preprocess.py and python train.py again.")

    tokenizer_vocab_limit = getattr(tokenizer, "num_words", None)
    if tokenizer_vocab_limit is None:
        expected_vocab_size = max(token_ids) + 1
        if expected_vocab_size != model_vocab_size:
            raise ValueError(
                f"Tokenizer effective vocabulary size ({expected_vocab_size}, including padding ID 0) "
                f"does not match model output size ({model_vocab_size})."
            )
    elif int(tokenizer_vocab_limit) != model_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary limit ({tokenizer_vocab_limit}) does not match model output size "
            f"({model_vocab_size}). Please retrain the tokenizer and model together."
        )
    if not any(token_id < model_vocab_size for token_id in token_ids):
        raise ValueError("Tokenizer has no token IDs supported by the model output layer.")


def validate_demo_v3_model_tokenizer_compatibility(model, tokenizer, model_id):
    from demo_v3_preprocess import effective_vocabulary_size, tokenizer_mapping_sha256

    experiment_id = str(model_id).upper()
    if experiment_id not in {"A", "B", "C", "D"}:
        raise ValueError("Demo V3 model_id must be one of 'A', 'B', 'C', or 'D'.")
    metadata = json.loads(DEMO_V3_METADATA_PATH.read_text(encoding="utf-8"))
    if effective_vocabulary_size(tokenizer) != metadata.get("vocabulary_size"):
        raise ValueError("Demo V3 tokenizer vocabulary size differs from preprocessing metadata.")
    if tokenizer_mapping_sha256(tokenizer) != metadata.get("tokenizer_mapping_sha256"):
        raise ValueError("Demo V3 tokenizer mapping does not match the model's preprocessing metadata.")
    if int(model.output_shape[-1]) != metadata["vocabulary_size"]:
        raise ValueError("Demo V3 model output size does not match the tokenizer vocabulary.")
    expected_mask = experiment_id in {"C", "D"}
    if bool(model.layers[0].mask_zero) != expected_mask:
        raise ValueError(f"Demo V3 model {experiment_id} has an unexpected padding-mask configuration.")


def tokenize_for_prediction(text, tokenizer):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text is empty. Please provide a sentence to predict the next word.")

    tokens = re.findall(r"[A-Za-z0-9']+|[.,!?;:]", cleaned.lower())
    if not tokens:
        raise ValueError("Input text does not contain usable words for prediction.")

    sequence = tokenizer.texts_to_sequences([" ".join(tokens)])[0]
    if not sequence:
        return np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32)

    context = sequence[-SEQUENCE_LENGTH:]
    if len(context) < SEQUENCE_LENGTH:
        context = [0] * (SEQUENCE_LENGTH - len(context)) + context
    return np.asarray([context], dtype=np.int32)


def tokenize_v3_for_prediction(text, tokenizer):
    from demo_v3_preprocess import encode_text

    sequence = encode_text(text, tokenizer)
    if not sequence:
        raise ValueError("Input text does not contain usable tokens for prediction.")
    context = sequence[-SEQUENCE_LENGTH:]
    if len(context) < SEQUENCE_LENGTH:
        context = [0] * (SEQUENCE_LENGTH - len(context)) + context
    return np.asarray([context], dtype=np.int32)


def select_top_predictions(probabilities, tokenizer, top_k=5):
    valid_indices = _valid_vocabulary_indices(probabilities, tokenizer)
    if not valid_indices:
        raise ValueError("No valid vocabulary entries are available in the tokenizer.")

    ranked_indices = sorted(valid_indices, key=lambda idx: float(probabilities[int(idx)]), reverse=True)
    results = []
    seen = set()
    for idx in ranked_indices:
        if idx in seen:
            continue
        seen.add(idx)
        word = tokenizer.index_word.get(int(idx), "")
        if not word or word in {"<OOV>", "<PAD>"}:
            continue
        results.append(
            {
                "word": word,
                "probability": float(probabilities[int(idx)]),
                "rank": len(results) + 1,
            }
        )
        if len(results) >= top_k:
            break

    if not results:
        raise ValueError("No valid words were predicted. Try a richer sentence.")
    return results


def _valid_vocabulary_indices(probabilities, tokenizer):
    return [
        int(idx)
        for idx in range(1, len(probabilities))
        if tokenizer.index_word.get(int(idx), "") not in {"", "<OOV>", "<PAD>"}
    ]


def _apply_repetition_penalty(probabilities, tokenizer, recent_token_ids, history_token_ids, penalty):
    adjusted = np.asarray(probabilities, dtype=np.float64).copy()
    valid_indices = _valid_vocabulary_indices(adjusted, tokenizer)
    if not valid_indices:
        raise ValueError("No valid vocabulary entries are available in the tokenizer.")

    recent_counts = {}
    for token_id in recent_token_ids:
        recent_counts[token_id] = recent_counts.get(token_id, 0) + 1

    history = [token_id for token_id in history_token_ids if token_id != 0]
    for token_id in valid_indices:
        repeats = recent_counts.get(token_id, 0)
        if repeats:
            adjusted[token_id] /= penalty ** repeats
        if history and token_id == history[-1]:
            adjusted[token_id] /= 10.0 * penalty ** 2

        repeated_phrases = 0
        for phrase_size in (2, 3, 4):
            context_size = phrase_size - 1
            if len(history) < context_size:
                continue
            candidate_phrase = tuple(history[-context_size:] + [token_id])
            existing_count = sum(
                tuple(history[start : start + phrase_size]) == candidate_phrase
                for start in range(max(0, len(history) - phrase_size))
            )
            repeated_phrases += existing_count
        if repeated_phrases:
            adjusted[token_id] /= penalty ** repeated_phrases

    return adjusted


def _temperature_distribution(probabilities, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    powered = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0
    powered[positive] = np.exp(np.log(probabilities[positive]) / temperature)
    total = powered.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The model returned an invalid probability distribution.")
    return powered / total


def _choose_next_token(
    probabilities,
    tokenizer,
    strategy,
    temperature,
    top_k,
    repetition_penalty,
    recent_token_ids,
    history_token_ids,
):
    if strategy not in {"greedy", "top_k", "temperature"}:
        raise ValueError("strategy must be 'greedy', 'top_k', or 'temperature'.")
    if not np.isfinite(temperature) or not 0.2 <= temperature <= 1.5:
        raise ValueError("temperature must be between 0.2 and 1.5.")
    if not isinstance(top_k, int) or not 3 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 3 and 20.")
    if not np.isfinite(repetition_penalty) or repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0.")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or probabilities.sum() <= 0
    ):
        raise ValueError("The model returned an invalid probability distribution.")
    adjusted = _apply_repetition_penalty(
        probabilities,
        tokenizer,
        recent_token_ids,
        history_token_ids,
        repetition_penalty,
    )
    valid_indices = _valid_vocabulary_indices(adjusted, tokenizer)
    ranked_indices = sorted(valid_indices, key=lambda idx: adjusted[idx], reverse=True)

    if strategy == "greedy":
        return ranked_indices[0]
    if strategy == "top_k":
        candidates = ranked_indices[: min(top_k, len(ranked_indices))]
    else:
        candidates = ranked_indices

    sampling_probabilities = _temperature_distribution(adjusted[candidates], temperature)
    return int(np.random.choice(candidates, p=sampling_probabilities))


def _tokenize_input(text, tokenizer, artifact_set):
    if artifact_set == "demo_v3":
        return tokenize_v3_for_prediction(text, tokenizer)
    return tokenize_for_prediction(text, tokenizer)


def _predict_probabilities(text, model, tokenizer, artifact_set=DEFAULT_ARTIFACT_SET):
    x_input = _tokenize_input(text, tokenizer, artifact_set)
    probabilities = np.asarray(model.predict(x_input, verbose=0)[0], dtype=np.float64)
    if (
        probabilities.ndim != 1
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0, rtol=1e-4, atol=1e-5)
    ):
        raise ValueError("The model returned an invalid probability distribution.")
    return probabilities


def predict_next_words(text, model_name="rnn", top_k=5, artifact_set=DEFAULT_ARTIFACT_SET):
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    tokenizer = load_tokenizer(artifact_set)
    model = load_model(model_name, artifact_set)
    if artifact_set == "demo_v3":
        validate_demo_v3_model_tokenizer_compatibility(model, tokenizer, model_name)
    else:
        validate_model_tokenizer_compatibility(model, tokenizer)

    probabilities = _predict_probabilities(text, model, tokenizer, artifact_set)
    return select_top_predictions(probabilities, tokenizer, top_k=top_k)


def continue_sentence(
    text,
    model_name="rnn",
    additional_words=10,
    artifact_set=DEFAULT_ARTIFACT_SET,
    strategy="greedy",
    temperature=1.0,
    top_k=10,
    repetition_penalty=1.2,
):
    if not isinstance(additional_words, int) or additional_words <= 0:
        raise ValueError("additional_words must be a positive integer.")

    continuation = text.strip()
    if not continuation:
        raise ValueError("Input text is empty. Please provide a sentence to continue.")

    tokenizer = load_tokenizer(artifact_set)
    model = load_model(model_name, artifact_set)
    if artifact_set == "demo_v3":
        validate_demo_v3_model_tokenizer_compatibility(model, tokenizer, model_name)
    else:
        validate_model_tokenizer_compatibility(model, tokenizer)
    generated_token_ids = []
    for _ in range(additional_words):
        probabilities = _predict_probabilities(continuation, model, tokenizer, artifact_set)
        history_token_ids = _tokenize_input(continuation, tokenizer, artifact_set)[0].tolist()
        token_id = _choose_next_token(
            probabilities,
            tokenizer,
            strategy,
            temperature,
            top_k,
            repetition_penalty,
            generated_token_ids,
            history_token_ids,
        )
        generated_token_ids.append(token_id)
        prediction = tokenizer.index_word[token_id]
        if prediction in {".", ",", "!", "?", ";", ":"}:
            continuation += prediction
        else:
            continuation += f" {prediction}"
    return continuation


if __name__ == "__main__":
    sample = "I am learning deep"
    for model_name in ("rnn", "lstm"):
        try:
            print(model_name.upper(), predict_next_words(sample, model_name=model_name, top_k=5))
        except FileNotFoundError as exc:
            print(str(exc))
        except ValueError as exc:
            print(str(exc))
