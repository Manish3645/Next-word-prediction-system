import pickle

import numpy as np

import preprocess
from preprocess import (
    build_tokenizer,
    clean_text,
    create_processed_arrays,
    download_and_prepare_data,
    extract_text_examples,
)


def test_clean_text_preserves_meaningful_punctuation():
    text = "  Hello, world!  "
    cleaned = clean_text(text)
    assert cleaned == "Hello, world!"


def test_create_processed_arrays_returns_fixed_shape():
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3, 4, 5, 6]]

    x_array, y_array = create_processed_arrays(["hello world this is a test"], DummyTokenizer(), 10)
    assert x_array.shape[1] == 10
    assert y_array.shape[0] == x_array.shape[0]


def test_generated_targets_are_non_empty():
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3, 4]]

    x_array, y_array = create_processed_arrays(["hello world is fun"], DummyTokenizer(), 10)
    assert y_array.size > 0


def test_clean_text_preserves_paragraph_boundaries():
    assert clean_text("First paragraph.\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."


def test_dataset_sample_limits_select_rows_not_column_names():
    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, indices):
            return FakeSplit([self.rows[index] for index in indices])

        def __iter__(self):
            return iter(self.rows)

    split = FakeSplit(
        [
            {"text": "first example"},
            {"text": "second example"},
            {"text": "third example"},
        ]
    )

    assert extract_text_examples(split, limit=2) == ["first example", "second example"]
    assert extract_text_examples(split) == ["first example", "second example", "third example"]


def test_download_preprocessing_keeps_splits_and_fits_tokenizer_on_train_only(tmp_path, monkeypatch):
    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, indices):
            return FakeSplit([self.rows[index] for index in indices])

        def __iter__(self):
            return iter(self.rows)

    fake_dataset = {
        "train": FakeSplit(
            [
                {"text": "trainword alpha beta gamma delta"},
                {"text": "ignored train row content"},
            ]
        ),
        "validation": FakeSplit([{"text": "validationonly omega sigma"}]),
        "test": FakeSplit([{"text": "testonly theta lambda"}]),
    }
    data_dir = tmp_path / "wikitext2"
    models_dir = tmp_path / "models"

    monkeypatch.setattr(preprocess, "_load_dataset", lambda *_args: fake_dataset)
    monkeypatch.setattr(preprocess, "RAW_TRAIN_PATH", data_dir / "train.txt")
    monkeypatch.setattr(preprocess, "RAW_VALID_PATH", data_dir / "valid.txt")
    monkeypatch.setattr(preprocess, "RAW_TEST_PATH", data_dir / "test.txt")
    monkeypatch.setattr(preprocess, "PROCESSED_TRAIN_PATH", data_dir / "processed_train.npz")
    monkeypatch.setattr(preprocess, "PROCESSED_VALID_PATH", data_dir / "processed_validation.npz")
    monkeypatch.setattr(preprocess, "PROCESSED_TEST_PATH", data_dir / "processed_test.npz")
    monkeypatch.setattr(preprocess, "WIKITEXT2_METADATA_PATH", data_dir / "preprocessing_metadata.json")
    monkeypatch.setattr(preprocess, "WIKITEXT2_MODELS_DIR", models_dir)

    summary = download_and_prepare_data(
        max_train_samples=1,
        max_valid_samples=1,
        max_test_samples=1,
    )

    assert summary["train_text_lines"] == 1
    assert summary["valid_text_lines"] == 1
    assert summary["test_text_lines"] == 1
    assert "trainword" in (data_dir / "train.txt").read_text(encoding="utf-8")
    assert "validationonly" in (data_dir / "valid.txt").read_text(encoding="utf-8")
    assert "testonly" in (data_dir / "test.txt").read_text(encoding="utf-8")
    assert "ignored" not in (data_dir / "train.txt").read_text(encoding="utf-8")

    with (models_dir / "tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    assert "trainword" in tokenizer.word_index
    assert "validationonly" not in tokenizer.word_index
    assert "testonly" not in tokenizer.word_index

    train_arrays = np.load(data_dir / "processed_train.npz")
    valid_arrays = np.load(data_dir / "processed_validation.npz")
    test_arrays = np.load(data_dir / "processed_test.npz")
    assert train_arrays["x"].shape[1] == valid_arrays["x"].shape[1] == test_arrays["x"].shape[1] == 10
    metadata = (data_dir / "preprocessing_metadata.json").read_text(encoding="utf-8")
    assert "Salesforce/wikitext/wikitext-2-raw-v1" in metadata


def test_build_tokenizer_only_contains_training_vocabulary():
    tokenizer = build_tokenizer(["trainword alpha beta"])
    assert "trainword" in tokenizer.word_index
    assert "validationword" not in tokenizer.word_index


def test_sequence_examples_do_not_cross_sentence_boundaries():
    class DummyTokenizer:
        word_ids = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4}

        def texts_to_sequences(self, texts):
            return [[self.word_ids[word] for word in texts[0].split() if word in self.word_ids]]

    x_array, y_array = create_processed_arrays(
        ["alpha beta. gamma delta"],
        DummyTokenizer(),
        sequence_length=3,
    )

    assert x_array.tolist() == [[0, 0, 1], [0, 0, 3]]
    assert y_array.tolist() == [2, 4]


def test_sequence_budget_selects_deterministic_text_prefix():
    selected = preprocess.select_text_for_sequence_budget(
        ["alpha beta gamma", "unused words after target"],
        sequence_budget=2,
    )
    assert selected == ["alpha beta gamma"]


def test_demo_preprocessing_isolated_and_fits_tokenizer_on_demo_train_only(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    demo_data_dir = tmp_path / "demo-data"
    demo_models_dir = tmp_path / "demo-models"
    source_dir.mkdir()
    full_metadata_path = source_dir / "preprocessing_metadata.json"
    full_metadata_path.write_text(
        '{"dataset": "Salesforce/wikitext/wikitext-2-raw-v1"}',
        encoding="utf-8",
    )
    raw_source_paths = {
        "train": source_dir / "train.txt",
        "validation": source_dir / "valid.txt",
        "test": source_dir / "test.txt",
    }
    raw_source_paths["train"].write_text(
        "trainalpha trainbeta trainonly trainmore trainlast\n",
        encoding="utf-8",
    )
    raw_source_paths["validation"].write_text(
        "validomega validdelta validmore validnext\n",
        encoding="utf-8",
    )
    raw_source_paths["test"].write_text(
        "testomega testdelta testmore testnext\n",
        encoding="utf-8",
    )
    original_train_data = raw_source_paths["train"].read_bytes()

    demo_paths = {
        "DEMO_DATA_DIR": demo_data_dir,
        "DEMO_RAW_TRAIN_PATH": demo_data_dir / "train.txt",
        "DEMO_RAW_VALID_PATH": demo_data_dir / "valid.txt",
        "DEMO_RAW_TEST_PATH": demo_data_dir / "test.txt",
        "DEMO_PROCESSED_TRAIN_PATH": demo_data_dir / "processed_train.npz",
        "DEMO_PROCESSED_VALID_PATH": demo_data_dir / "processed_validation.npz",
        "DEMO_PROCESSED_TEST_PATH": demo_data_dir / "processed_test.npz",
        "DEMO_METADATA_PATH": demo_data_dir / "preprocessing_metadata.json",
        "DEMO_TOKENIZER_PATH": demo_models_dir / "tokenizer.pkl",
    }
    for name, value in demo_paths.items():
        monkeypatch.setattr(preprocess, name, value)
    monkeypatch.setattr(preprocess, "WIKITEXT2_METADATA_PATH", full_metadata_path)
    monkeypatch.setattr(preprocess, "RAW_TRAIN_PATH", raw_source_paths["train"])
    monkeypatch.setattr(preprocess, "RAW_VALID_PATH", raw_source_paths["validation"])
    monkeypatch.setattr(preprocess, "RAW_TEST_PATH", raw_source_paths["test"])

    summary = preprocess.prepare_demo_data(
        train_sequence_limit=3,
        valid_sequence_limit=2,
        test_sequence_limit=2,
    )

    assert summary["actual_sequences"] == {"train": 3, "validation": 2, "test": 2}
    assert summary["tokenizer_fit_split"] == "demo training text only"
    assert raw_source_paths["train"].read_bytes() == original_train_data
    with demo_paths["DEMO_TOKENIZER_PATH"].open("rb") as file:
        tokenizer = pickle.load(file)
    assert "trainalpha" in tokenizer.word_index
    assert "validomega" not in tokenizer.word_index
    assert "testomega" not in tokenizer.word_index
    for split_name, expected_count in (("train", 3), ("validation", 2), ("test", 2)):
        arrays = np.load(
            {
                "train": demo_paths["DEMO_PROCESSED_TRAIN_PATH"],
                "validation": demo_paths["DEMO_PROCESSED_VALID_PATH"],
                "test": demo_paths["DEMO_PROCESSED_TEST_PATH"],
            }[split_name]
        )
        assert arrays["x"].shape == (expected_count, 10)
        assert arrays["y"].shape == (expected_count,)


def test_seeded_sequence_budget_selection_is_representative_and_repeatable():
    rows = [f"record{index} alpha beta gamma delta" for index in range(20)]
    first = preprocess.select_seeded_text_for_sequence_budget(rows, 20, seed=42)
    second = preprocess.select_seeded_text_for_sequence_budget(rows, 20, seed=42)
    assert first == second
    assert len(first) > 1
    assert first[0] != rows[0]


def test_demo_v2_preprocessing_uses_separate_splits_and_training_only_tokenizer(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    v2_data_dir = tmp_path / "demo-v2-data"
    v2_models_dir = tmp_path / "demo-v2-models"
    source_dir.mkdir()
    full_metadata_path = source_dir / "preprocessing_metadata.json"
    full_metadata_path.write_text(
        '{"dataset": "Salesforce/wikitext/wikitext-2-raw-v1"}',
        encoding="utf-8",
    )
    source_paths = {
        "train": source_dir / "train.txt",
        "validation": source_dir / "valid.txt",
        "test": source_dir / "test.txt",
    }
    source_paths["train"].write_text(
        "trainingtoken alpha beta gamma delta\ntrainonly epsilon zeta eta theta\n",
        encoding="utf-8",
    )
    source_paths["validation"].write_text(
        "validationtoken alpha beta gamma delta\n",
        encoding="utf-8",
    )
    source_paths["test"].write_text(
        "testtoken alpha beta gamma delta\n",
        encoding="utf-8",
    )

    demo_paths = {
        "DEMO_V2_DATA_DIR": v2_data_dir,
        "DEMO_V2_RAW_TRAIN_PATH": v2_data_dir / "train.txt",
        "DEMO_V2_RAW_VALID_PATH": v2_data_dir / "valid.txt",
        "DEMO_V2_RAW_TEST_PATH": v2_data_dir / "test.txt",
        "DEMO_V2_PROCESSED_TRAIN_PATH": v2_data_dir / "processed_train.npz",
        "DEMO_V2_PROCESSED_VALID_PATH": v2_data_dir / "processed_validation.npz",
        "DEMO_V2_PROCESSED_TEST_PATH": v2_data_dir / "processed_test.npz",
        "DEMO_V2_METADATA_PATH": v2_models_dir / "metadata.json",
        "DEMO_V2_TOKENIZER_PATH": v2_models_dir / "tokenizer.pkl",
    }
    for name, value in demo_paths.items():
        monkeypatch.setattr(preprocess, name, value)
    monkeypatch.setattr(preprocess, "WIKITEXT2_METADATA_PATH", full_metadata_path)
    monkeypatch.setattr(preprocess, "RAW_TRAIN_PATH", source_paths["train"])
    monkeypatch.setattr(preprocess, "RAW_VALID_PATH", source_paths["validation"])
    monkeypatch.setattr(preprocess, "RAW_TEST_PATH", source_paths["test"])
    monkeypatch.setattr(preprocess, "RANDOM_SEED", 42)

    summary = preprocess.prepare_demo_v2_data(
        train_sequence_limit=4,
        valid_sequence_limit=2,
        test_sequence_limit=2,
    )

    assert summary["actual_sequences"] == {"train": 4, "validation": 2, "test": 2}
    assert summary["tokenizer_fit_split"] == "demo_v2 training text only"
    assert summary["effective_vocabulary_size"] == summary["max_token_id"] + 1
    with demo_paths["DEMO_V2_TOKENIZER_PATH"].open("rb") as file:
        tokenizer = pickle.load(file)
    assert tokenizer.num_words is None
    assert {"trainingtoken", "trainonly"} & tokenizer.word_index.keys()
    assert "validationtoken" not in tokenizer.word_index
    assert "testtoken" not in tokenizer.word_index

    for split_name in ("train", "validation", "test"):
        arrays = np.load(
            {
                "train": demo_paths["DEMO_V2_PROCESSED_TRAIN_PATH"],
                "validation": demo_paths["DEMO_V2_PROCESSED_VALID_PATH"],
                "test": demo_paths["DEMO_V2_PROCESSED_TEST_PATH"],
            }[split_name]
        )
        assert arrays["x"].shape[1] == 10
        assert arrays["x"].max() < summary["effective_vocabulary_size"]
        assert arrays["y"].max() < summary["effective_vocabulary_size"]


def test_effective_vocabulary_size_includes_padding_and_oov():
    class DummyTokenizer:
        word_index = {"<OOV>": 1, "alpha": 2, "beta": 3}

    assert preprocess.effective_vocabulary_size(DummyTokenizer()) == 4
