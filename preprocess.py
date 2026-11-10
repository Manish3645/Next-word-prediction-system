import argparse
import json
import pickle
import random
import re

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DEMO_DATA_DIR,
    DEMO_METADATA_PATH,
    DEMO_PROCESSED_TEST_PATH,
    DEMO_PROCESSED_TRAIN_PATH,
    DEMO_PROCESSED_VALID_PATH,
    DEMO_RAW_TEST_PATH,
    DEMO_RAW_TRAIN_PATH,
    DEMO_RAW_VALID_PATH,
    DEMO_TOKENIZER_PATH,
    DEMO_V2_DATA_DIR,
    DEMO_V2_METADATA_PATH,
    DEMO_V2_PROCESSED_TEST_PATH,
    DEMO_V2_PROCESSED_TRAIN_PATH,
    DEMO_V2_PROCESSED_VALID_PATH,
    DEMO_V2_RAW_TEST_PATH,
    DEMO_V2_RAW_TRAIN_PATH,
    DEMO_V2_RAW_VALID_PATH,
    DEMO_V2_TOKENIZER_PATH,
    RANDOM_SEED,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VALID_PATH,
    WIKITEXT2_METADATA_PATH,
    WIKITEXT2_MODELS_DIR,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VALID_PATH,
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
)


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)


def clean_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("\xa0", " ")
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("“", '"')
    cleaned = cleaned.replace("”", '"')
    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"[^\S\n]+", " ", line)
        line = re.sub(r"(?<=[A-Za-z0-9])\s*([.,!?;:])", r"\1", line)
        lines.append(line.strip())
    return "\n".join(lines).strip()


def _load_dataset(dataset_name, dataset_config):
    from datasets import load_dataset

    return load_dataset(dataset_name, dataset_config)


def split_sentences(text: str):
    if not text:
        return []
    fragments = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [clean_text(fragment) for fragment in fragments if clean_text(fragment)]
    return sentences


def tokenize_sentence(sentence: str, tokenizer):
    if not sentence:
        return []
    words = re.findall(r"[A-Za-z0-9']+|[.,!?;:]", sentence.lower())
    if not words:
        return []
    sentence_text = " ".join(words)
    return tokenizer.texts_to_sequences([sentence_text])[0]


def generate_sequence_examples(token_ids, sequence_length):
    if len(token_ids) < 2:
        return [], []

    x_rows = []
    y_rows = []
    for idx in range(len(token_ids) - 1):
        context = token_ids[max(0, idx - sequence_length + 1): idx + 1]
        if not context:
            continue
        if len(context) < sequence_length:
            context = [0] * (sequence_length - len(context)) + context
        else:
            context = context[-sequence_length:]
        x_rows.append(context)
        y_rows.append(token_ids[idx + 1])
    return x_rows, y_rows


def build_tokenizer(train_texts, vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
    )
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def effective_vocabulary_size(tokenizer):
    return max(tokenizer.word_index.values(), default=0) + 1


def save_tokenizer(tokenizer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(tokenizer, file)


def save_text_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_lines = [clean_text(line) for line in lines if clean_text(line)]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(cleaned_lines))


def select_dataset_rows(split_records, limit=None):
    """Select the first limit records without iterating Hugging Face column keys."""
    if limit is None:
        return split_records
    if limit < 0:
        raise ValueError("Sample limits must be zero or a positive integer.")

    count = min(limit, len(split_records))
    if hasattr(split_records, "select"):
        return split_records.select(range(count))
    return split_records[:count]


def extract_text_examples(split_records, limit=None):
    selected_records = select_dataset_rows(split_records, limit)
    texts = []
    for sample in selected_records:
        if isinstance(sample, dict):
            text = sample.get("text", "")
        else:
            text = str(sample)
        cleaned = clean_text(text)
        if cleaned:
            texts.append(cleaned)
    return texts


def create_processed_arrays(text_lines, tokenizer, sequence_length=SEQUENCE_LENGTH):
    x_chunks = []
    y_chunks = []

    for text in text_lines:
        for sentence in split_sentences(text):
            token_ids = tokenize_sentence(sentence, tokenizer)
            if len(token_ids) < 2:
                continue
            token_array = np.asarray(token_ids, dtype=np.int32)
            targets = token_array[1:]
            positions = np.arange(len(targets))
            x_sentence = np.zeros((len(targets), sequence_length), dtype=np.int32)
            for column in range(sequence_length):
                source_positions = positions - (sequence_length - column - 1)
                valid_positions = source_positions >= 0
                x_sentence[valid_positions, column] = token_array[source_positions[valid_positions]]
            x_chunks.append(x_sentence)
            y_chunks.append(targets)

    if not x_chunks:
        fallback_x = []
        fallback_y = []
        for text in text_lines:
            for sentence in split_sentences(text):
                token_ids = tokenize_sentence(sentence, tokenizer)
                if not token_ids:
                    continue
                fallback_x.append(np.zeros(sequence_length, dtype=np.int32))
                fallback_y.append(token_ids[-1])
        if not fallback_x:
            raise ValueError("No training sequences were generated. Check input text or sequence length.")
        x_chunks = [np.stack(fallback_x)]
        y_chunks = [np.asarray(fallback_y, dtype=np.int32)]

    x_array = np.concatenate(x_chunks, axis=0)
    y_array = np.concatenate(y_chunks, axis=0)
    return x_array, y_array


def select_text_for_sequence_budget(text_lines, sequence_budget):
    if sequence_budget <= 0:
        raise ValueError("Sequence budgets must be positive integers.")

    selected_texts = []
    sequence_count = 0
    for text in text_lines:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        selected_texts.append(cleaned)
        for sentence in split_sentences(cleaned):
            token_count = len(re.findall(r"[A-Za-z0-9']+", sentence))
            sequence_count += max(token_count - 1, 0)
        if sequence_count >= sequence_budget:
            break
    return selected_texts


def select_seeded_text_for_sequence_budget(text_lines, sequence_budget, seed):
    if sequence_budget <= 0:
        raise ValueError("Sequence budgets must be positive integers.")

    indices = list(range(len(text_lines)))
    random.Random(seed).shuffle(indices)
    selected_texts = []
    sequence_count = 0
    for index in indices:
        cleaned = clean_text(text_lines[index])
        if not cleaned:
            continue
        sentence_token_counts = [
            len(re.findall(r"[A-Za-z0-9']+", sentence))
            for sentence in split_sentences(cleaned)
        ]
        record_sequence_count = sum(max(count - 1, 0) for count in sentence_token_counts)
        if record_sequence_count <= 0:
            continue
        selected_texts.append(cleaned)
        sequence_count += record_sequence_count
        if sequence_count >= sequence_budget:
            break
    return selected_texts


def prepare_demo_v2_data(
    train_sequence_limit=20_000,
    valid_sequence_limit=10_000,
    test_sequence_limit=10_000,
    overwrite=False,
):
    set_seed()
    expected_dataset = f"{DATASET_NAME}/{DATASET_CONFIG}"
    if not WIKITEXT2_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Full WikiText-2 metadata is missing: {WIKITEXT2_METADATA_PATH}. "
            "Prepare the full dataset before creating Demo V2."
        )
    full_metadata = json.loads(WIKITEXT2_METADATA_PATH.read_text(encoding="utf-8"))
    if full_metadata.get("dataset") != expected_dataset:
        raise ValueError(f"Expected source data for {expected_dataset}.")

    source_paths = {
        "train": RAW_TRAIN_PATH,
        "validation": RAW_VALID_PATH,
        "test": RAW_TEST_PATH,
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Full WikiText-2 raw split files are missing: " + ", ".join(missing_sources)
        )

    output_paths = [
        DEMO_V2_RAW_TRAIN_PATH,
        DEMO_V2_RAW_VALID_PATH,
        DEMO_V2_RAW_TEST_PATH,
        DEMO_V2_PROCESSED_TRAIN_PATH,
        DEMO_V2_PROCESSED_VALID_PATH,
        DEMO_V2_PROCESSED_TEST_PATH,
        DEMO_V2_TOKENIZER_PATH,
        DEMO_V2_METADATA_PATH,
    ]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Demo V2 preprocessing outputs already exist and will not be overwritten. "
            "Pass overwrite=True to replace only Demo V2 outputs. Existing: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    limits = {
        "train": train_sequence_limit,
        "validation": valid_sequence_limit,
        "test": test_sequence_limit,
    }
    seeds = {
        "train": RANDOM_SEED,
        "validation": RANDOM_SEED + 1,
        "test": RANDOM_SEED + 2,
    }
    selected_splits = {}
    source_counts = {}
    for split_name, source_path in source_paths.items():
        source_texts = source_path.read_text(encoding="utf-8").splitlines()
        source_counts[split_name] = len(source_texts)
        selected_splits[split_name] = select_seeded_text_for_sequence_budget(
            source_texts,
            limits[split_name],
            seeds[split_name],
        )

    train_texts = selected_splits["train"]
    if not train_texts:
        raise ValueError("No Demo V2 training text was selected from WikiText-2.")
    tokenizer = build_tokenizer(train_texts, vocab_size=None)
    vocab_size = effective_vocabulary_size(tokenizer)
    if vocab_size <= 1:
        raise ValueError("Demo V2 training tokenizer has no usable vocabulary.")
    save_tokenizer(tokenizer, DEMO_V2_TOKENIZER_PATH)

    processed_paths = {
        "train": DEMO_V2_PROCESSED_TRAIN_PATH,
        "validation": DEMO_V2_PROCESSED_VALID_PATH,
        "test": DEMO_V2_PROCESSED_TEST_PATH,
    }
    raw_paths = {
        "train": DEMO_V2_RAW_TRAIN_PATH,
        "validation": DEMO_V2_RAW_VALID_PATH,
        "test": DEMO_V2_RAW_TEST_PATH,
    }
    sequence_counts = {}
    oov_counts = {}
    selected_counts = {}
    for split_name, texts in selected_splits.items():
        save_text_lines(raw_paths[split_name], texts)
        x_values, y_values = create_processed_arrays(texts, tokenizer, SEQUENCE_LENGTH)
        x_values = x_values[: limits[split_name]]
        y_values = y_values[: limits[split_name]]
        if not len(y_values):
            raise ValueError(f"No Demo V2 {split_name} sequences were generated.")
        if x_values.min() < 0 or y_values.min() < 0 or x_values.max() >= vocab_size or y_values.max() >= vocab_size:
            raise ValueError(f"Demo V2 {split_name} contains token IDs outside the effective vocabulary.")
        np.savez_compressed(processed_paths[split_name], x=x_values, y=y_values)
        sequence_counts[split_name] = int(len(y_values))
        oov_counts[split_name] = int(np.count_nonzero(y_values == tokenizer.word_index[tokenizer.oov_token]))
        selected_counts[split_name] = len(texts)

    summary = {
        "dataset": expected_dataset,
        "artifact_set": "demo_v2",
        "source_artifact_set": "wikitext2",
        "selection_method": "seeded randomized record-level selection within each official split",
        "selection_seeds": seeds,
        "source_record_counts": source_counts,
        "selected_record_counts": selected_counts,
        "requested_sequence_limits": limits,
        "actual_sequences": sequence_counts,
        "oov_target_counts": oov_counts,
        "oov_target_percentages": {
            split_name: 100.0 * oov_counts[split_name] / sequence_counts[split_name]
            for split_name in sequence_counts
        },
        "sequence_length": SEQUENCE_LENGTH,
        "tokenizer_fit_split": "demo_v2 training text only",
        "tokenizer_num_words": tokenizer.num_words,
        "tokenizer_word_index_size": len(tokenizer.word_index),
        "max_token_id": max(tokenizer.word_index.values()),
        "effective_vocabulary_size": vocab_size,
        "padding_id": 0,
        "oov_token": tokenizer.oov_token,
        "oov_id": int(tokenizer.word_index[tokenizer.oov_token]),
        "tokenizer_path": str(DEMO_V2_TOKENIZER_PATH),
        "processed_split_paths": {name: str(path) for name, path in processed_paths.items()},
        "raw_split_paths": {name: str(path) for name, path in raw_paths.items()},
        "source_raw_split_paths": {name: str(path) for name, path in source_paths.items()},
    }
    DEMO_V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_V2_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_V2_METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def prepare_demo_data(
    train_sequence_limit=20_000,
    valid_sequence_limit=10_000,
    test_sequence_limit=10_000,
    overwrite=False,
):
    set_seed()
    expected_dataset = f"{DATASET_NAME}/{DATASET_CONFIG}"
    if not WIKITEXT2_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Full WikiText-2 metadata is missing: {WIKITEXT2_METADATA_PATH}. "
            "Prepare the full dataset before creating the demo subset."
        )
    full_metadata = json.loads(WIKITEXT2_METADATA_PATH.read_text(encoding="utf-8"))
    if full_metadata.get("dataset") != expected_dataset:
        raise ValueError(f"Expected source data for {expected_dataset}.")

    source_paths = {
        "train": RAW_TRAIN_PATH,
        "validation": RAW_VALID_PATH,
        "test": RAW_TEST_PATH,
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Full WikiText-2 raw split files are missing: " + ", ".join(missing_sources)
        )

    demo_outputs = [
        DEMO_RAW_TRAIN_PATH,
        DEMO_RAW_VALID_PATH,
        DEMO_RAW_TEST_PATH,
        DEMO_PROCESSED_TRAIN_PATH,
        DEMO_PROCESSED_VALID_PATH,
        DEMO_PROCESSED_TEST_PATH,
        DEMO_TOKENIZER_PATH,
        DEMO_METADATA_PATH,
    ]
    existing_outputs = [path for path in demo_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Demo preprocessing outputs already exist and will not be overwritten. "
            f"Pass overwrite=True to replace only demo outputs. Existing: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    limits = {
        "train": train_sequence_limit,
        "validation": valid_sequence_limit,
        "test": test_sequence_limit,
    }
    selected_splits = {}
    for split_name, source_path in source_paths.items():
        source_texts = source_path.read_text(encoding="utf-8").splitlines()
        selected_splits[split_name] = select_text_for_sequence_budget(
            source_texts,
            limits[split_name],
        )

    train_texts = selected_splits["train"]
    if not train_texts:
        raise ValueError("No demo training text was selected from the full WikiText-2 training split.")
    tokenizer = build_tokenizer(train_texts)
    save_tokenizer(tokenizer, DEMO_TOKENIZER_PATH)

    processed_paths = {
        "train": DEMO_PROCESSED_TRAIN_PATH,
        "validation": DEMO_PROCESSED_VALID_PATH,
        "test": DEMO_PROCESSED_TEST_PATH,
    }
    raw_paths = {
        "train": DEMO_RAW_TRAIN_PATH,
        "validation": DEMO_RAW_VALID_PATH,
        "test": DEMO_RAW_TEST_PATH,
    }
    sequence_counts = {}
    text_counts = {}
    for split_name, texts in selected_splits.items():
        save_text_lines(raw_paths[split_name], texts)
        x_values, y_values = create_processed_arrays(texts, tokenizer, SEQUENCE_LENGTH)
        if len(y_values) > limits[split_name]:
            x_values = x_values[: limits[split_name]]
            y_values = y_values[: limits[split_name]]
        if not len(y_values):
            raise ValueError(f"No demo {split_name} sequences were generated from the source split.")
        np.savez_compressed(processed_paths[split_name], x=x_values, y=y_values)
        sequence_counts[split_name] = int(len(y_values))
        text_counts[split_name] = len(texts)

    summary = {
        "dataset": expected_dataset,
        "artifact_set": "demo",
        "source_artifact_set": "wikitext2",
        "sequence_length": SEQUENCE_LENGTH,
        "vocab_size": len(tokenizer.word_index) + 1,
        "tokenizer_num_words": tokenizer.num_words,
        "tokenizer_fit_split": "demo training text only",
        "requested_sequence_limits": limits,
        "actual_sequences": sequence_counts,
        "selected_text_lines": text_counts,
        "raw_split_paths": {name: str(path) for name, path in raw_paths.items()},
        "processed_split_paths": {name: str(path) for name, path in processed_paths.items()},
        "tokenizer_path": str(DEMO_TOKENIZER_PATH),
        "source_raw_split_paths": {name: str(path) for name, path in source_paths.items()},
    }
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def download_and_prepare_data(
    max_train_samples=None,
    max_valid_samples=None,
    max_test_samples=None,
    save_raw_files=True,
    overwrite=False,
):
    set_seed()
    output_paths = [
        PROCESSED_TRAIN_PATH,
        PROCESSED_VALID_PATH,
        PROCESSED_TEST_PATH,
        WIKITEXT2_METADATA_PATH,
        WIKITEXT2_MODELS_DIR / "tokenizer.pkl",
    ]
    if save_raw_files:
        output_paths.extend([RAW_TRAIN_PATH, RAW_VALID_PATH, RAW_TEST_PATH])
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "WikiText-2 preprocessing outputs already exist and will not be overwritten. "
            f"Pass overwrite=True to replace them. Existing: {paths}"
        )

    dataset = _load_dataset(DATASET_NAME, DATASET_CONFIG)

    split_texts = {"train": [], "validation": [], "test": []}
    limits = {"train": max_train_samples, "validation": max_valid_samples, "test": max_test_samples}

    for split_name, split_key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        cleaned_lines = extract_text_examples(dataset[split_key], limits[split_name])
        split_texts[split_name] = cleaned_lines

        if save_raw_files:
            target_path = {"train": RAW_TRAIN_PATH, "validation": RAW_VALID_PATH, "test": RAW_TEST_PATH}[split_name]
            save_text_lines(target_path, cleaned_lines)

    train_texts = split_texts["train"]
    if not train_texts:
        raise ValueError("The selected WikiText-2 training split contains no non-empty text.")
    tokenizer = build_tokenizer(train_texts)
    tokenizer_path = WIKITEXT2_MODELS_DIR / "tokenizer.pkl"
    save_tokenizer(tokenizer, tokenizer_path)

    processed_counts = {}
    for split_name in ("train", "validation", "test"):
        x_array, y_array = create_processed_arrays(split_texts[split_name], tokenizer, SEQUENCE_LENGTH)
        save_path = {
            "train": PROCESSED_TRAIN_PATH,
            "validation": PROCESSED_VALID_PATH,
            "test": PROCESSED_TEST_PATH,
        }[split_name]
        np.savez_compressed(save_path, x=x_array, y=y_array)
        processed_counts[split_name] = x_array.shape[0]
        del x_array, y_array

    summary = {
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "train_text_lines": len(train_texts),
        "valid_text_lines": len(split_texts["validation"]),
        "test_text_lines": len(split_texts["test"]),
        "train_sequences": processed_counts["train"],
        "valid_sequences": processed_counts["validation"],
        "test_sequences": processed_counts["test"],
        "vocab_size": len(tokenizer.word_index) + 1,
        "sequence_length": SEQUENCE_LENGTH,
        "tokenizer_path": str(tokenizer_path),
        "sample_limits": {
            "train": max_train_samples,
            "validation": max_valid_samples,
            "test": max_test_samples,
        },
    }

    WIKITEXT2_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIKITEXT2_METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess WikiText-2 raw data.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create an isolated demo subset from already-preprocessed WikiText-2 raw splits.",
    )
    parser.add_argument(
        "--demo-v2",
        action="store_true",
        help="Create a seeded, representative Demo V2 subset from existing WikiText-2 split files.",
    )
    parser.add_argument("--demo-train-sequences", type=int, default=20_000)
    parser.add_argument("--demo-valid-sequences", type=int, default=10_000)
    parser.add_argument("--demo-test-sequences", type=int, default=10_000)
    parser.add_argument(
        "--overwrite-demo",
        action="store_true",
        help="Replace existing demo outputs only.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing WikiText-2 preprocessing outputs.",
    )
    args = parser.parse_args()
    if args.demo and args.demo_v2:
        parser.error("Choose either --demo or --demo-v2.")
    if args.demo_v2:
        prepare_demo_v2_data(
            train_sequence_limit=args.demo_train_sequences,
            valid_sequence_limit=args.demo_valid_sequences,
            test_sequence_limit=args.demo_test_sequences,
            overwrite=args.overwrite_demo,
        )
    elif args.demo:
        prepare_demo_data(
            train_sequence_limit=args.demo_train_sequences,
            valid_sequence_limit=args.demo_valid_sequences,
            test_sequence_limit=args.demo_test_sequences,
            overwrite=args.overwrite_demo,
        )
    else:
        download_and_prepare_data(
            max_train_samples=args.max_train_samples,
            max_valid_samples=args.max_valid_samples,
            max_test_samples=args.max_test_samples,
            overwrite=args.overwrite,
        )
