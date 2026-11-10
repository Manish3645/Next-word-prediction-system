import json
import pickle

import numpy as np
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

import train as training
from preprocess import build_tokenizer, effective_vocabulary_size
from train import build_lstm_model, build_rnn_model, load_preprocessed_data, train_model


def test_model_input_shape_is_valid():
    model = Sequential(
        [
            Embedding(input_dim=200, output_dim=32),
            SimpleRNN(16),
            Dense(200, activation="softmax"),
        ]
    )
    model.build((None, 10))
    x = np.random.randint(0, 200, (2, 10))
    y = model(x)
    assert y.shape[0] == 2


def test_lstm_model_outputs_probabilities():
    model = Sequential(
        [
            Embedding(input_dim=200, output_dim=32),
            LSTM(16),
            Dense(200, activation="softmax"),
        ]
    )
    model.build((None, 10))
    x = np.random.randint(0, 200, (3, 10))
    probs = model(x)
    assert probs.shape == (3, 200)


def test_project_model_builders_have_configured_input_and_output_shapes():
    for model in (build_rnn_model(), build_lstm_model()):
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 12000)


def test_demo_v2_model_builders_use_dynamic_vocabulary_size():
    vocab_size = 57
    rnn = build_rnn_model(vocab_size=vocab_size)
    lstm = build_lstm_model(vocab_size=vocab_size)
    assert rnn.input_shape == lstm.input_shape == (None, 10)
    assert rnn.output_shape == lstm.output_shape == (None, vocab_size)
    assert rnn.layers[0].input_dim == lstm.layers[0].input_dim == vocab_size


def test_demo_v2_loader_and_models_share_training_tokenizer_vocab(tmp_path, monkeypatch):
    tokenizer = build_tokenizer(["alpha beta gamma delta"], vocab_size=None)
    vocab_size = effective_vocabulary_size(tokenizer)
    tokenizer_path = tmp_path / "tokenizer.pkl"
    with tokenizer_path.open("wb") as file:
        pickle.dump(tokenizer, file)
    train_path = tmp_path / "processed_train.npz"
    valid_path = tmp_path / "processed_validation.npz"
    x_train = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 2, 3]], dtype=np.int32)
    y_train = np.asarray([4], dtype=np.int32)
    x_valid = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 2, 3]], dtype=np.int32)
    y_valid = np.asarray([4], dtype=np.int32)
    np.savez_compressed(train_path, x=x_train, y=y_train)
    np.savez_compressed(valid_path, x=x_valid, y=y_valid)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
                "tokenizer_fit_split": "demo_v2 training text only",
                "effective_vocabulary_size": vocab_size,
                "actual_sequences": {"train": 1, "validation": 1},
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "tokenizer": tokenizer_path,
        "processed_train": train_path,
        "processed_valid": valid_path,
        "metadata": metadata_path,
    }
    monkeypatch.setattr(training, "get_artifact_paths", lambda _artifact_set: paths)

    loaded_x_train, loaded_y_train, loaded_x_valid, loaded_y_valid = load_preprocessed_data("demo_v2")
    rnn = build_rnn_model(vocab_size)
    lstm = build_lstm_model(vocab_size)

    assert np.array_equal(loaded_x_train, x_train)
    assert np.array_equal(loaded_y_train, y_train)
    assert np.array_equal(loaded_x_valid, x_valid)
    assert np.array_equal(loaded_y_valid, y_valid)
    assert loaded_y_train.max() < vocab_size
    assert loaded_x_valid.max() < vocab_size
    assert loaded_y_valid.max() < vocab_size
    assert rnn.output_shape == lstm.output_shape == (None, vocab_size)
    assert paths["tokenizer"] == tokenizer_path


def test_training_checkpoints_are_independent_and_loadable(tmp_path):
    x_train = np.asarray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    y_train = np.asarray([2, 3, 4, 5], dtype=np.int32)

    checkpoint_paths = {}
    for model_name, recurrent_layer in (("rnn", SimpleRNN), ("lstm", LSTM)):
        model = Sequential(
            [
                Embedding(input_dim=8, output_dim=4),
                recurrent_layer(4),
                Dense(8, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        checkpoint_path = tmp_path / f"{model_name}_checkpoint.keras"
        train_model(model, x_train, y_train, x_train, y_train, checkpoint_path, epochs=1)
        checkpoint_paths[model_name] = checkpoint_path

    assert checkpoint_paths["rnn"].exists()
    assert checkpoint_paths["lstm"].exists()
    assert checkpoint_paths["rnn"] != checkpoint_paths["lstm"]
    assert load_model(checkpoint_paths["rnn"]).output_shape == (None, 8)
    assert load_model(checkpoint_paths["lstm"]).output_shape == (None, 8)


def test_completed_rnn_checkpoint_and_history_survive_lstm_failure(tmp_path, monkeypatch):
    rnn_path = tmp_path / "models" / "rnn_model.keras"
    lstm_path = tmp_path / "models" / "lstm_model.keras"
    rnn_history_path = tmp_path / "results" / "rnn_history.json"
    lstm_history_path = tmp_path / "results" / "lstm_history.json"
    results_dir = tmp_path / "results"

    paths = {
        "rnn_model": rnn_path,
        "lstm_model": lstm_path,
        "rnn_history": rnn_history_path,
        "lstm_history": lstm_history_path,
        "train_history": results_dir / "training_history.json",
        "training_plot": results_dir / "training_history.png",
    }
    monkeypatch.setattr(training, "get_artifact_paths", lambda _artifact_set: paths)
    monkeypatch.setattr(training, "set_random_seeds", lambda: None)
    x_train = np.asarray([[0, 0, 1], [0, 1, 2]], dtype=np.int32)
    y_train = np.asarray([2, 3], dtype=np.int32)
    x_valid = np.asarray([[0, 0, 1]], dtype=np.int32)
    y_valid = np.asarray([2], dtype=np.int32)
    monkeypatch.setattr(
        training,
        "load_preprocessed_data",
        lambda _artifact_set: (x_train, y_train, x_valid, y_valid),
    )
    monkeypatch.setattr(training, "build_rnn_model", lambda: "rnn")
    monkeypatch.setattr(training, "build_lstm_model", lambda: "lstm")
    training_inputs = []

    def fake_train_model(model, x_values, y_values, valid_x, valid_y, checkpoint_path, epochs=1):
        training_inputs.append((x_values, y_values, valid_x, valid_y))
        if model == "lstm":
            raise RuntimeError("simulated LSTM failure")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("completed RNN checkpoint", encoding="utf-8")
        return {
            "loss": [1.0],
            "val_loss": [1.0],
            "accuracy": [0.5],
            "val_accuracy": [0.5],
            "training_time_seconds": 1.0,
        }, 1.0

    monkeypatch.setattr(training, "train_model", fake_train_model)

    try:
        training.main()
    except RuntimeError as exc:
        assert str(exc) == "simulated LSTM failure"
    else:
        raise AssertionError("Expected the simulated LSTM training failure")

    assert rnn_path.exists()
    assert rnn_history_path.exists()
    assert not lstm_path.exists()
    assert not lstm_history_path.exists()
    assert len(training_inputs) == 2
    for training_input in training_inputs:
        assert training_input[0] is x_train
        assert training_input[1] is y_train
        assert training_input[2] is x_valid
        assert training_input[3] is y_valid
