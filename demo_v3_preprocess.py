import argparse
import hashlib
import json
import pickle
import random
import re
import unicodedata

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DEMO_V2_METADATA_PATH,
    DEMO_V3_DATA_DIR,
    DEMO_V3_DATASET_STATISTICS_PATH,
    DEMO_V3_MASK_ZERO,
    DEMO_V3_METADATA_PATH,
    DEMO_V3_MODELS_DIR,
    DEMO_V3_PREPROCESSING_CONFIG_PATH,
    DEMO_V3_PREPROCESSING_REPORT_PATH,
    DEMO_V3_PROCESSED_TEST_PATH,
    DEMO_V3_PROCESSED_TRAIN_PATH,
    DEMO_V3_PROCESSED_VALID_PATH,
    DEMO_V3_RAW_TEST_PATH,
    DEMO_V3_RAW_TRAIN_PATH,
    DEMO_V3_RAW_VALID_PATH,
    DEMO_V3_RESULTS_DIR,
    DEMO_V3_TEST_SEQUENCE_LIMIT,
    DEMO_V3_TOKENIZER_PATH,
    DEMO_V3_TRAIN_SEQUENCE_LIMIT,
    DEMO_V3_VALID_SEQUENCE_LIMIT,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VALID_PATH,
    SEQUENCE_LENGTH,
    WIKITEXT2_METADATA_PATH,
)

PREPROCESSING_VERSION = "demo_v3_unicode_tokens_v1"


def _project_relative_path(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


TOKEN_PATTERN = re.compile(
    r"@[-,.]@|[^\W_]+(?:'[^\W_]+)*|[^\w\s]|_",
    flags=re.UNICODE,
)
SENTENCE_END_PATTERN = re.compile(r"""[.!?]["')\]]*(?=\s|$)""")
WIKITEXT_MARKERS = {"@-@": "-", "@,@": ",", "@.@": "."}
TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201f": '"',
        "\u00a0": " ",
    }
)


def normalize_text(text):
    """Normalize Unicode and whitespace while preserving line boundaries."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text)).translate(TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def tokenize_text(text):
    """Return canonical lowercase Unicode tokens, retaining punctuation."""
    normalized = normalize_text(text)
    tokens = []
    for token in TOKEN_PATTERN.findall(normalized):
        mapped = WIKITEXT_MARKERS.get(token, token)
        tokens.append(unicodedata.normalize("NFC", mapped.casefold()))
    return tokens


def encode_tokens(tokens, tokenizer):
    if not tokens:
        return []
    return tokenizer.texts_to_sequences([" ".join(tokens)])[0]


def encode_text(text, tokenizer):
    return encode_tokens(tokenize_text(text), tokenizer)


def split_text_into_sentences(text):
    """Split at terminal punctuation and line breaks without joining contexts."""
    normalized = normalize_text(text)
    sentences = []
    for paragraph in normalized.splitlines():
        start = 0
        for match in SENTENCE_END_PATTERN.finditer(paragraph):
            sentence = paragraph[start:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
            while start < len(paragraph) and paragraph[start].isspace():
                start += 1
        remainder = paragraph[start:].strip()
        if remainder:
            sentences.append(remainder)
    return sentences


def tokenize_sentences(text):
    return [
        tokens
        for sentence in split_text_into_sentences(text)
        if (tokens := tokenize_text(sentence))
    ]


def build_v3_tokenizer(training_records):
    tokenizer = Tokenizer(
        num_words=None,
        oov_token="<OOV>",
        filters="",
        lower=False,
        split=" ",
    )
    tokenizer.fit_on_texts(
        [" ".join(tokens) for record in training_records for tokens in record["sentences"]]
    )
    return tokenizer


def effective_vocabulary_size(tokenizer):
    return max(tokenizer.word_index.values(), default=0) + 1


def tokenizer_mapping_sha256(tokenizer):
    ordered_mapping = sorted(tokenizer.word_index.items(), key=lambda item: item[1])
    serialized = json.dumps(ordered_mapping, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _record_sentences(text):
    return [tokens for tokens in tokenize_sentences(text) if len(tokens) >= 2]


def select_records_for_sequence_budget(text_records, sequence_budget, seed, seen_records=None):
    """Select seeded records and truncate the last record at an example boundary."""
    if sequence_budget <= 0:
        raise ValueError("Sequence budgets must be positive integers.")

    seen = {} if seen_records is None else seen_records
    indices = list(range(len(text_records)))
    random.Random(seed).shuffle(indices)
    selected_records = []
    selected_sequences = 0
    duplicates_within_split = 0
    duplicates_from_other_splits = 0
    skipped_without_sequences = 0

    for index in indices:
        normalized = normalize_text(text_records[index])
        if not normalized:
            continue
        key = normalized.casefold()
        previous_split = seen.get(key)
        if previous_split is not None:
            if previous_split == "current":
                duplicates_within_split += 1
            else:
                duplicates_from_other_splits += 1
            continue

        source_sentences = _record_sentences(normalized)
        record_sequence_count = sum(len(tokens) - 1 for tokens in source_sentences)
        if record_sequence_count <= 0:
            skipped_without_sequences += 1
            continue

        remaining = sequence_budget - selected_sequences
        selected_sentences = []
        for tokens in source_sentences:
            take_count = min(len(tokens) - 1, remaining)
            if take_count > 0:
                selected_sentences.append(tokens[:take_count + 1])
                selected_sequences += take_count
                remaining -= take_count
            if remaining == 0:
                break

        selected_records.append(
            {
                "source_index": index,
                "sentences": selected_sentences,
            }
        )
        seen[key] = "current"
        if selected_sequences >= sequence_budget:
            break

    if selected_sequences < sequence_budget:
        raise ValueError(
            f"Split has only {selected_sequences} usable sequences; "
            f"{sequence_budget} were requested."
        )
    return selected_records, {
        "selected_sequences": selected_sequences,
        "duplicates_within_split_excluded": duplicates_within_split,
        "duplicates_from_other_splits_excluded": duplicates_from_other_splits,
        "records_without_sequences_skipped": skipped_without_sequences,
    }


def build_arrays(selected_records, tokenizer, sequence_length=SEQUENCE_LENGTH):
    x_rows = []
    y_rows = []
    oov_id = tokenizer.word_index[tokenizer.oov_token]

    for record in selected_records:
        for tokens in record["sentences"]:
            token_ids = encode_tokens(tokens, tokenizer)
            for target_position in range(1, len(token_ids)):
                context = token_ids[max(0, target_position - sequence_length):target_position]
                x_rows.append([0] * (sequence_length - len(context)) + context)
                y_rows.append(token_ids[target_position])

    x_values = np.asarray(x_rows, dtype=np.int32)
    y_values = np.asarray(y_rows, dtype=np.int32)
    if not len(y_values):
        raise ValueError("No next-token sequences were generated.")
    vocab_size = effective_vocabulary_size(tokenizer)
    if x_values.min() < 0 or y_values.min() < 0:
        raise ValueError("Generated arrays contain negative token IDs.")
    if x_values.max() >= vocab_size or y_values.max() >= vocab_size:
        raise ValueError("Generated arrays contain IDs outside the tokenizer vocabulary.")
    if np.any(y_values == 0):
        raise ValueError("Padding ID 0 must never be used as a target.")
    non_padding_context = x_values != 0
    context_count = int(np.count_nonzero(non_padding_context))
    oov_context_count = int(np.count_nonzero(x_values == oov_id))
    oov_target_count = int(np.count_nonzero(y_values == oov_id))
    return x_values, y_values, {
        "sequence_count": int(len(y_values)),
        "oov_target_count": oov_target_count,
        "oov_target_percentage": 100.0 * oov_target_count / len(y_values),
        "oov_context_token_count": oov_context_count,
        "non_padding_context_token_count": context_count,
        "oov_context_token_percentage": (
            100.0 * oov_context_count / context_count if context_count else 0.0
        ),
    }


def _materialized_text(record):
    return " ".join(" ".join(tokens) for tokens in record["sentences"])


def _output_paths():
    return [
        DEMO_V3_RAW_TRAIN_PATH,
        DEMO_V3_RAW_VALID_PATH,
        DEMO_V3_RAW_TEST_PATH,
        DEMO_V3_PROCESSED_TRAIN_PATH,
        DEMO_V3_PROCESSED_VALID_PATH,
        DEMO_V3_PROCESSED_TEST_PATH,
        DEMO_V3_TOKENIZER_PATH,
        DEMO_V3_METADATA_PATH,
        DEMO_V3_PREPROCESSING_CONFIG_PATH,
        DEMO_V3_DATASET_STATISTICS_PATH,
        DEMO_V3_PREPROCESSING_REPORT_PATH,
    ]


def _validate_official_source():
    expected_dataset = f"{DATASET_NAME}/{DATASET_CONFIG}"
    if not WIKITEXT2_METADATA_PATH.exists():
        raise FileNotFoundError(f"Official WikiText-2 metadata is missing: {WIKITEXT2_METADATA_PATH}")
    source_metadata = json.loads(WIKITEXT2_METADATA_PATH.read_text(encoding="utf-8"))
    if source_metadata.get("dataset") != expected_dataset:
        raise ValueError(f"Expected official source dataset {expected_dataset}.")
    sources = {"train": RAW_TRAIN_PATH, "validation": RAW_VALID_PATH, "test": RAW_TEST_PATH}
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Official WikiText-2 split file(s) missing: " + ", ".join(missing))
    return sources, expected_dataset


def prepare_demo_v3_data(
    train_sequence_limit=DEMO_V3_TRAIN_SEQUENCE_LIMIT,
    valid_sequence_limit=DEMO_V3_VALID_SEQUENCE_LIMIT,
    test_sequence_limit=DEMO_V3_TEST_SEQUENCE_LIMIT,
    write_report=False,
    test_results="Not run",
):
    existing = [path for path in _output_paths() if path.exists()]
    if existing:
        raise FileExistsError(
            "Demo V3 outputs already exist and are never overwritten by this command: "
            + ", ".join(str(path) for path in existing)
        )
    source_paths, expected_dataset = _validate_official_source()
    source_records = {
        name: path.read_text(encoding="utf-8").splitlines()
        for name, path in source_paths.items()
    }
    budgets = {
        "train": train_sequence_limit,
        "validation": valid_sequence_limit,
        "test": test_sequence_limit,
    }
    seeds = {
        "train": RANDOM_SEED,
        "validation": RANDOM_SEED + 1,
        "test": RANDOM_SEED + 2,
    }
    seen_records = {}
    selected = {}
    selection_stats = {}
    for split_name in ("train", "validation", "test"):
        selected[split_name], selection_stats[split_name] = select_records_for_sequence_budget(
            source_records[split_name],
            budgets[split_name],
            seeds[split_name],
            seen_records,
        )
        for key in list(seen_records):
            if seen_records[key] == "current":
                seen_records[key] = split_name

    tokenizer = build_v3_tokenizer(selected["train"])
    vocab_size = effective_vocabulary_size(tokenizer)
    if vocab_size <= 2:
        raise ValueError("Demo V3 training tokenizer has no usable vocabulary.")
    oov_id = int(tokenizer.word_index[tokenizer.oov_token])
    if tokenizer.word_index.get(tokenizer.oov_token) != 1:
        raise ValueError("The tokenizer OOV token must have reserved ID 1.")

    processed_paths = {
        "train": DEMO_V3_PROCESSED_TRAIN_PATH,
        "validation": DEMO_V3_PROCESSED_VALID_PATH,
        "test": DEMO_V3_PROCESSED_TEST_PATH,
    }
    raw_paths = {
        "train": DEMO_V3_RAW_TRAIN_PATH,
        "validation": DEMO_V3_RAW_VALID_PATH,
        "test": DEMO_V3_RAW_TEST_PATH,
    }
    split_stats = {}
    arrays = {}
    for split_name in ("train", "validation", "test"):
        x_values, y_values, stats = build_arrays(selected[split_name], tokenizer)
        if stats["sequence_count"] != budgets[split_name]:
            raise ValueError(f"Demo V3 {split_name} sequence count does not match its requested budget.")
        if split_name == "train" and stats["oov_target_count"]:
            raise ValueError(
                f"Demo V3 training examples unexpectedly contain {stats['oov_target_count']} OOV targets."
            )
        split_stats[split_name] = stats
        arrays[split_name] = (x_values, y_values)

    mapping_hash = tokenizer_mapping_sha256(tokenizer)
    preprocessing_config = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": expected_dataset,
        "seed": RANDOM_SEED,
        "split_selection_seeds": seeds,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_budgets": budgets,
        "unicode_normalization": "NFC",
        "case_policy": "Unicode casefold",
        "whitespace_policy": "collapse horizontal whitespace; preserve line boundaries",
        "apostrophe_policy": "normalize curly apostrophes to ASCII apostrophe; retain internal apostrophes in contractions",
        "quotation_policy": "normalize curly double quotes to ASCII double quote token",
        "punctuation_policy": "retain punctuation and underscore symbols as individual vocabulary tokens",
        "hyphen_policy": "split hyphenated words; retain hyphen as a token",
        "wikitext_marker_policy": {"@-@": "-", "@,@": ",", "@.@": "."},
        "sentence_boundary_policy": "sentence-final . ! ? and source line breaks; no sequence crosses a boundary",
        "tokenizer": {
            "class": "tensorflow.keras.preprocessing.text.Tokenizer",
            "num_words": None,
            "filters": "",
            "lower": False,
            "split": " ",
            "padding_id": 0,
            "oov_token": tokenizer.oov_token,
            "oov_id": oov_id,
            "effective_vocabulary_size": vocab_size,
            "ordered_word_index_sha256": mapping_hash,
        },
        "model_configuration": {
            "sequence_length": SEQUENCE_LENGTH,
            "vocabulary_size": vocab_size,
            "embedding_dim": 128,
            "hidden_units": 64,
            "dropout": 0.2,
            "mask_zero": DEMO_V3_MASK_ZERO,
            "output_classes": vocab_size,
        },
    }
    selected_record_ids = {
        name: [f"{name}:{record['source_index']}" for record in selected[name]]
        for name in selected
    }
    metadata = {
        "artifact_set": "demo_v3",
        "dataset": expected_dataset,
        "source_splits": {
            name: _project_relative_path(path) for name, path in source_paths.items()
        },
        "source_record_counts": {name: len(records) for name, records in source_records.items()},
        "selected_record_counts": {name: len(records) for name, records in selected.items()},
        "selected_source_record_ids": selected_record_ids,
        "sequence_budgets": budgets,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_counts": {name: split_stats[name]["sequence_count"] for name in split_stats},
        "split_statistics": split_stats,
        "selection_statistics": selection_stats,
        "vocabulary_size": vocab_size,
        "padding_id": 0,
        "oov_token": tokenizer.oov_token,
        "oov_id": oov_id,
        "tokenizer_fit_scope": "final selected Demo V3 training sequences only",
        "tokenizer_mapping_sha256": mapping_hash,
        "duplicate_checks": {
            "duplicate_selected_records_within_or_across_splits": 0,
            "excluded_duplicates_by_split": {
                name: {
                    "within_split": selection_stats[name]["duplicates_within_split_excluded"],
                    "from_prior_split": selection_stats[name]["duplicates_from_other_splits_excluded"],
                }
                for name in selected
            },
        },
        "split_separation_checks": {
            "official_splits_kept_separate": True,
            "selected_source_record_ids_overlap": False,
            "selected_normalized_record_text_overlap": False,
        },
        "sequence_boundary_checks": {
            "sequences_cross_sentence_boundary": 0,
            "sequences_cross_record_boundary": 0,
            "all_sequences_within_source_sentence": True,
        },
        "paths": {
            "raw_splits": {
                name: _project_relative_path(path) for name, path in raw_paths.items()
            },
            "processed_splits": {
                name: _project_relative_path(path) for name, path in processed_paths.items()
            },
            "tokenizer": _project_relative_path(DEMO_V3_TOKENIZER_PATH),
            "preprocessing_config": _project_relative_path(DEMO_V3_PREPROCESSING_CONFIG_PATH),
            "dataset_statistics": _project_relative_path(DEMO_V3_DATASET_STATISTICS_PATH),
        },
    }

    DEMO_V3_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_V3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_V3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, records in selected.items():
        materialized_records = [_materialized_text(record) for record in records]
        raw_paths[split_name].write_text("\n".join(materialized_records) + "\n", encoding="utf-8")
        x_values, y_values = arrays[split_name]
        np.savez_compressed(processed_paths[split_name], x=x_values, y=y_values)
    with DEMO_V3_TOKENIZER_PATH.open("wb") as file:
        pickle.dump(tokenizer, file)
    DEMO_V3_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    DEMO_V3_PREPROCESSING_CONFIG_PATH.write_text(
        json.dumps(preprocessing_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    DEMO_V3_DATASET_STATISTICS_PATH.write_text(
        json.dumps(
            {
                "selected_record_counts": metadata["selected_record_counts"],
                "sequence_counts": metadata["sequence_counts"],
                "vocabulary_size": vocab_size,
                "split_statistics": split_stats,
                "selection_statistics": selection_stats,
                "duplicate_checks": metadata["duplicate_checks"],
                "split_separation_checks": metadata["split_separation_checks"],
                "sequence_boundary_checks": metadata["sequence_boundary_checks"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if write_report:
        write_demo_v3_report(metadata, test_results=test_results)
    return metadata


def write_demo_v3_report(metadata, test_results="Not run"):
    v2_stats = (
        json.loads(DEMO_V2_METADATA_PATH.read_text(encoding="utf-8"))
        if DEMO_V2_METADATA_PATH.exists()
        else {}
    )
    lines = [
        "# Demo V3 Preprocessing Report",
        "",
        "## Changes implemented",
        "",
        "- Added an isolated Demo V3 preprocessing module using one Unicode-aware normalization, tokenization, and encoding path for every split and inference input.",
        "- NFC-normalized text, normalized curly quotes/apostrophes, preserved punctuation tokens, retained internal apostrophes, split hyphenated forms, and mapped WikiText `@-@`, `@,@`, and `@.@` markers.",
        "- Fit the tokenizer only on the final selected training examples; reserved ID 0 for padding and ID 1 for OOV; recorded a SHA-256 hash of ordered word-to-ID entries.",
        "- Added a `mask_zero=True` model-builder option for both recurrent architectures. No models were trained.",
        "- Preserved the official WikiText-2 train/validation/test sources and wrote all generated artifacts under Demo V3 paths.",
        "",
        "## Files created and modified",
        "",
        "- Modified `config.py` with isolated Demo V3 paths and settings.",
        "- Modified `train.py` to accept the optional `mask_zero` setting on both model builders; defaults preserve existing behavior.",
        "- Modified `predict.py` with an opt-in Demo V3 inference tokenizer; existing application tokenization remains unchanged.",
        "- Added `demo_v3_preprocess.py` and `tests/test_demo_v3_preprocessing.py`.",
        "- Created `data/demo_v3/`, `models/demo_v3/`, and `results/demo_v3/` artifacts, including this report.",
        "",
        "## Dataset statistics",
        "",
        "| Split | Selected records | Sequences | OOV targets | OOV targets % | OOV context tokens | Non-padding context tokens | OOV context % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "validation", "test"):
        stats = metadata["split_statistics"][split_name]
        lines.append(
            f"| {split_name} | {metadata['selected_record_counts'][split_name]} "
            f"| {stats['sequence_count']} | {stats['oov_target_count']} "
            f"| {stats['oov_target_percentage']:.3f}% | {stats['oov_context_token_count']} "
            f"| {stats['non_padding_context_token_count']} | {stats['oov_context_token_percentage']:.3f}% |"
        )
    lines.extend(
        [
            "",
            f"- Effective vocabulary size: **{metadata['vocabulary_size']:,}** (including padding and OOV IDs).",
            f"- Ordered word-index mapping SHA-256: `{metadata['tokenizer_mapping_sha256']}`.",
            f"- Train sequences: **{metadata['sequence_counts']['train']:,}**, versus Demo V2 **20,000**.",
            "",
            "## OOV comparison against Demo V2",
            "",
            "| Split | Demo V2 OOV targets | Demo V2 OOV % | Demo V3 OOV targets | Demo V3 OOV % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split_name, v2_key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        v2_count = v2_stats.get("oov_target_counts", {}).get(v2_key, "Unavailable")
        v2_percent = v2_stats.get("oov_target_percentages", {}).get(v2_key)
        v2_pct_text = f"{v2_percent:.3f}%" if isinstance(v2_percent, (int, float)) else "Unavailable"
        stats = metadata["split_statistics"][split_name]
        lines.append(
            f"| {split_name} | {v2_count} | {v2_pct_text} "
            f"| {stats['oov_target_count']} | {stats['oov_target_percentage']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Official split separation: **{metadata['split_separation_checks']['official_splits_kept_separate']}**.",
            f"- Selected record overlap: **{metadata['split_separation_checks']['selected_source_record_ids_overlap']}**.",
            f"- Duplicate selected normalized records: **{metadata['duplicate_checks']['duplicate_selected_records_within_or_across_splits']}**.",
            "- Sequences crossing sentence boundaries: **0**.",
            "- Sequences crossing record boundaries: **0**.",
            "- Training OOV targets: **0** (generation stops with an error if this invariant fails).",
            "",
            "## Test results",
            "",
            test_results,
            "",
            "## Artifacts",
            "",
            f"- Tokenizer: `{_project_relative_path(DEMO_V3_TOKENIZER_PATH)}`",
            f"- Preprocessing metadata: `{_project_relative_path(DEMO_V3_METADATA_PATH)}`",
            f"- Preprocessing configuration: `{_project_relative_path(DEMO_V3_PREPROCESSING_CONFIG_PATH)}`",
            f"- Dataset statistics: `{_project_relative_path(DEMO_V3_DATASET_STATISTICS_PATH)}`",
            f"- Processed data: `{_project_relative_path(DEMO_V3_PROCESSED_TRAIN_PATH)}`, "
            f"`{_project_relative_path(DEMO_V3_PROCESSED_VALID_PATH)}`, "
            f"`{_project_relative_path(DEMO_V3_PROCESSED_TEST_PATH)}`",
            "",
            "## Remaining limitations",
            "",
            "- The 50,000 training-sequence budget is larger than Demo V2 but remains a subset of WikiText-2; rare held-out terms can still map to OOV.",
            "- Sentence splitting is punctuation-based and can split after abbreviations; a linguistic sentence segmenter is not used.",
            "- Punctuation and WikiText markers are vocabulary tokens, so Demo V3 requires models trained with this exact tokenizer and cannot use V2 checkpoints.",
            "- `mask_zero=True` is configured as an opt-in experiment, not yet validated by training or evaluation.",
            "- No Demo V3 models were trained and no V2 files or metrics were changed.",
            "",
        ]
    )
    DEMO_V3_PREPROCESSING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return DEMO_V3_PREPROCESSING_REPORT_PATH


def main():
    parser = argparse.ArgumentParser(description="Prepare isolated Demo V3 WikiText-2 preprocessing artifacts.")
    parser.add_argument("--train-sequences", type=int, default=DEMO_V3_TRAIN_SEQUENCE_LIMIT)
    parser.add_argument("--validation-sequences", type=int, default=DEMO_V3_VALID_SEQUENCE_LIMIT)
    parser.add_argument("--test-sequences", type=int, default=DEMO_V3_TEST_SEQUENCE_LIMIT)
    args = parser.parse_args()
    metadata = prepare_demo_v3_data(
        train_sequence_limit=args.train_sequences,
        valid_sequence_limit=args.validation_sequences,
        test_sequence_limit=args.test_sequences,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
