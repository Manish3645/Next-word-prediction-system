import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui_common import (
    ARTIFACT_SET,
    V3_MODEL_LABELS,
    append_word,
    better_model,
    get_demo_paths,
    missing_runtime_assets,
    read_json_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = [
    "ui_pages/1_Next_Word_Prediction.py",
    "ui_pages/2_Sentence_Continuation.py",
    "ui_pages/3_Model_Comparison.py",
    "ui_pages/4_Performance_Dashboard.py",
    "ui_pages/5_About.py",
]


def test_streamlit_entry_point_renders_navigation():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.sidebar is not None


@pytest.mark.parametrize("page_file", PAGE_FILES)
def test_streamlit_pages_render_without_errors(page_file):
    app = AppTest.from_file(str(PROJECT_ROOT / page_file), default_timeout=60).run()
    assert not app.exception


def test_prediction_page_uses_real_model_and_appends_selected_suggestion():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "ui_pages/1_Next_Word_Prediction.py"),
        default_timeout=60,
    ).run()
    app.text_area[0].set_value("The history of")
    app.button[0].click().run()

    assert not app.exception
    predictions = app.session_state["prediction_results"]
    assert len(predictions) == 5
    assert all(0.0 <= item["probability"] <= 1.0 for item in predictions)
    first_word = predictions[0]["word"]

    app.button[1].click().run()

    assert not app.exception
    assert app.text_area[0].value == append_word("The history of", first_word)
    assert len(app.session_state["prediction_results"]) == 5


def test_sentence_continuation_page_exposes_all_decoding_controls():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "ui_pages/2_Sentence_Continuation.py"),
        default_timeout=60,
    ).run()
    assert not app.exception
    assert [element.label for element in app.selectbox] == [
        "Artifact set",
        "Model",
        "Decoding strategy",
    ]
    assert [element.label for element in app.slider] == [
        "Words to generate",
        "Temperature",
        "Top-K candidates",
        "Repetition penalty",
    ]


def test_inference_pages_keep_demo_v2_available():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "ui_pages/1_Next_Word_Prediction.py"),
        default_timeout=60,
    ).run()
    app.selectbox[0].select("Demo V2").run()
    assert not app.exception
    assert list(app.selectbox[1].options) == ["SimpleRNN", "LSTM"]


def test_comparison_table_matches_saved_demo_v3_metrics():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "ui_pages/3_Model_Comparison.py"),
        default_timeout=60,
    ).run()
    assert not app.exception
    actual = json.loads(get_demo_paths()["metrics"].read_text(encoding="utf-8"))
    rows = app.dataframe[0].value
    assert rows.iloc[0]["Test loss"] == f"{actual['experiments']['A']['overall_test']['test_loss']:.4f}"
    assert rows.iloc[0]["Top-1 accuracy"] == f"{actual['experiments']['A']['overall_test']['top1_accuracy'] * 100:.2f}%"


def test_ui_reads_saved_demo_v3_metrics_without_changing_them():
    metrics_path = get_demo_paths()["metrics"]
    expected = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert read_json_artifact(metrics_path) == expected


def test_metric_winner_directions_use_actual_values():
    metrics = {
        "rnn": {"test_loss": 2.0, "top1_accuracy": 0.3},
        "lstm": {"test_loss": 1.0, "top1_accuracy": 0.2},
    }
    assert better_model(metrics, "test_loss") == "LSTM"
    assert better_model(metrics, "top1_accuracy") == "SimpleRNN"


def test_demo_v3_metric_winners_use_nested_saved_values():
    experiments = read_json_artifact(get_demo_paths()["metrics"])["experiments"]
    assert better_model(experiments, "test_loss") == V3_MODEL_LABELS["A"]
    assert better_model(experiments, "top1_accuracy") == V3_MODEL_LABELS["A"]


def test_appending_suggestions_attaches_punctuation():
    assert append_word("A short prompt", "word") == "A short prompt word"
    assert append_word("A short prompt ", ".") == "A short prompt."


def test_missing_runtime_files_are_reported(tmp_path):
    paths = {
        "tokenizer": tmp_path / "tokenizer.pkl",
        "rnn_model": tmp_path / "rnn.keras",
        "lstm_model": tmp_path / "lstm.keras",
    }
    missing = missing_runtime_assets(paths)
    assert {label for label, _ in missing} == {
        "Demo V2 tokenizer",
        "SimpleRNN checkpoint",
        "LSTM checkpoint",
    }


def test_missing_demo_v3_checkpoints_are_reported(tmp_path):
    paths = {
        "tokenizer": tmp_path / "tokenizer.pkl",
        "metadata": tmp_path / "preprocessing_metadata.json",
        "experiment_models": {
            experiment_id: tmp_path / experiment_id / "best_model.keras"
            for experiment_id in ("A", "B", "C", "D")
        },
    }
    missing = missing_runtime_assets(paths)
    assert len(missing) == 6
    assert {label for label, _ in missing} == {
        "Demo V3 tokenizer",
        "Demo V3 preprocessing metadata",
        "Experiment A checkpoint",
        "Experiment B checkpoint",
        "Experiment C checkpoint",
        "Experiment D checkpoint",
    }
