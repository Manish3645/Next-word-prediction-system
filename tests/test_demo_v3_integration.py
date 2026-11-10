import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

import predict
from config import DEMO_V3_METADATA_PATH
from ui_common import V3_MODEL_LABELS, get_demo_paths, read_json_artifact


EXPERIMENT_IDS = tuple(V3_MODEL_LABELS)


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_demo_v3_checkpoints_load_and_predict_top_five(experiment_id):
    paths = get_demo_paths()
    assert paths["experiment_models"][experiment_id].is_file()
    tokenizer = predict.load_tokenizer("demo_v3")
    model = predict.load_model(experiment_id, "demo_v3")
    predict.validate_demo_v3_model_tokenizer_compatibility(model, tokenizer, experiment_id)

    predictions = predict.predict_next_words(
        "The history of",
        model_name=experiment_id,
        top_k=5,
        artifact_set="demo_v3",
    )
    assert len(predictions) == 5
    assert len({item["word"] for item in predictions}) == 5
    assert all(item["word"] in tokenizer.word_index for item in predictions)
    assert all(0.0 <= item["probability"] <= 1.0 for item in predictions)
    assert model.output_shape[-1] == len(tokenizer.word_index) + 1


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
@pytest.mark.parametrize("strategy", ["greedy", "top_k", "temperature"])
def test_demo_v3_continuation_strategies_use_real_model_and_vocab(
    monkeypatch, experiment_id, strategy
):
    tokenizer = predict.load_tokenizer("demo_v3")
    actual_model = predict.load_model(experiment_id, "demo_v3")
    selected_ids = []
    model_calls = []
    original_choose = predict._choose_next_token
    original_load_model = predict.load_model

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.output_shape = model.output_shape
            self.layers = model.layers

        def predict(self, inputs, verbose=0):
            model_calls.append(inputs.copy())
            return self.model.predict(inputs, verbose=verbose)

    wrapped_model = CountingModel(actual_model)
    monkeypatch.setattr(
        predict,
        "load_model",
        lambda model_name, artifact_set: wrapped_model
        if artifact_set == "demo_v3"
        else original_load_model(model_name, artifact_set),
    )

    def choose_and_record(*args, **kwargs):
        token_id = original_choose(*args, **kwargs)
        selected_ids.append(token_id)
        return token_id

    monkeypatch.setattr(predict, "_choose_next_token", choose_and_record)
    result = predict.continue_sentence(
        "The history of",
        model_name=experiment_id,
        additional_words=1,
        artifact_set="demo_v3",
        strategy=strategy,
        temperature=0.8,
        top_k=5,
        repetition_penalty=1.2,
    )
    assert result.startswith("The history of")
    assert len(selected_ids) == 1
    assert selected_ids[0] in tokenizer.index_word
    assert selected_ids[0] not in {0, 1}
    assert len(model_calls) == 1


def test_demo_v3_dashboard_and_comparison_use_saved_results():
    metrics = read_json_artifact(get_demo_paths()["metrics"])
    assert set(metrics["experiments"]) == set(EXPERIMENT_IDS)
    assert all(metrics["experiments"][key]["test_arrays_identical_across_experiments"] for key in EXPERIMENT_IDS)
    for experiment_id in EXPERIMENT_IDS:
        assert metrics["experiments"][experiment_id]["model_output_vocabulary_size"] == metrics["vocabulary_size"]
        assert np.isfinite(metrics["experiments"][experiment_id]["overall_test"]["test_loss"])
        assert 0.0 <= metrics["experiments"][experiment_id]["overall_test"]["top5_accuracy"] <= 1.0


def test_deployment_metadata_contains_only_portable_relative_paths():
    metadata = json.loads(DEMO_V3_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = read_json_artifact(get_demo_paths()["metrics"])
    paths = list(metadata["source_splits"].values())
    paths.extend(metadata["paths"]["raw_splits"].values())
    paths.extend(metadata["paths"]["processed_splits"].values())
    paths.extend(
        [
            metadata["paths"]["tokenizer"],
            metadata["paths"]["preprocessing_config"],
            metadata["paths"]["dataset_statistics"],
        ]
    )
    for experiment in metrics["experiments"].values():
        paths.extend([experiment["checkpoint_path"], experiment["history_path"]])
    assert all(not Path(value).is_absolute() for value in paths)
    assert all(not PureWindowsPath(value).is_absolute() for value in paths)
