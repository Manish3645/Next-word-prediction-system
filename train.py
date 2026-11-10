import argparse
import json
import pickle
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, SimpleRNN

from config import (
    BATCH_SIZE,
    DEMO_V2_EPOCHS,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_UNITS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    get_artifact_paths,
)


def load_preprocessed_data(artifact_set="wikitext2"):
    paths = get_artifact_paths(artifact_set)
    metadata_path = paths["metadata"]
    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(
            f"{artifact_set} preprocessing metadata is missing: {metadata_path}. "
            "Run the matching preprocessing command before training."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_dataset = "Salesforce/wikitext/wikitext-2-raw-v1"
    if metadata.get("dataset") != expected_dataset:
        raise ValueError(f"Expected WikiText-2 preprocessing metadata for {expected_dataset}.")
    if not paths["tokenizer"].exists():
        raise FileNotFoundError(f"{artifact_set} tokenizer is missing: {paths['tokenizer']}")
    with paths["tokenizer"].open("rb") as file:
        tokenizer = pickle.load(file)
    if artifact_set == "demo_v2":
        if metadata.get("tokenizer_fit_split") != "demo_v2 training text only":
            raise ValueError("Demo V2 tokenizer must be fitted on its training split only.")
        expected_vocab_size = max(tokenizer.word_index.values(), default=0) + 1
        if metadata.get("effective_vocabulary_size") != expected_vocab_size:
            raise ValueError("Demo V2 metadata does not match the tokenizer's effective vocabulary size.")
    else:
        expected_vocab_size = VOCAB_SIZE
        if getattr(tokenizer, "num_words", None) != VOCAB_SIZE:
            raise ValueError(
                f"Expected a tokenizer limited to {VOCAB_SIZE} words before training."
            )
    for path in (paths["processed_train"], paths["processed_valid"]):
        if not path.exists():
            raise FileNotFoundError(f"Preprocessed training data is missing: {path}")

    with np.load(paths["processed_train"]) as train_data, np.load(paths["processed_valid"]) as valid_data:
        x_train, y_train = train_data["x"], train_data["y"]
        x_valid, y_valid = valid_data["x"], valid_data["y"]
    for split_name, x_values, y_values in (
        ("training", x_train, y_train),
        ("validation", x_valid, y_valid),
    ):
        if x_values.ndim != 2 or x_values.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"{split_name.title()} input sequences must have length {SEQUENCE_LENGTH}.")
        if len(x_values) != len(y_values) or not len(y_values):
            raise ValueError(f"{split_name.title()} inputs and targets must be non-empty and aligned.")
        if x_values.min() < 0 or y_values.min() < 0:
            raise ValueError(f"{split_name.title()} data contains negative token IDs.")
        if x_values.max() >= expected_vocab_size or y_values.max() >= expected_vocab_size:
            raise ValueError(f"{split_name.title()} data contains token IDs outside the model vocabulary.")
    if artifact_set == "demo_v2":
        expected_counts = metadata.get("actual_sequences", {})
        if len(y_train) != expected_counts.get("train") or len(y_valid) != expected_counts.get("validation"):
            raise ValueError("Demo V2 arrays do not match preprocessing metadata sequence counts.")
    return x_train, y_train, x_valid, y_valid


def build_rnn_model(vocab_size=VOCAB_SIZE, mask_zero=False):
    model = Sequential(
        [
            Input(shape=(SEQUENCE_LENGTH,), dtype="int32"),
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=mask_zero),
            SimpleRNN(HIDDEN_UNITS),
            Dropout(DROPOUT),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model(vocab_size=VOCAB_SIZE, mask_zero=False):
    model = Sequential(
        [
            Input(shape=(SEQUENCE_LENGTH,), dtype="int32"),
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=mask_zero),
            LSTM(HIDDEN_UNITS),
            Dropout(DROPOUT),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_training_plot(histories, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, history in histories.items():
        axes[0].plot(history["loss"], label=f"{model_name} train loss")
        axes[0].plot(history["val_loss"], label=f"{model_name} val loss")
        axes[0].set_title("Loss comparison")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(history["accuracy"], label=f"{model_name} train acc")
        axes[1].plot(history["val_accuracy"], label=f"{model_name} val acc")
        axes[1].set_title("Accuracy comparison")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_model(model, x_train, y_train, x_valid, y_valid, checkpoint_path, epochs=EPOCHS):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time
    history_dict = history.history
    history_dict["training_time_seconds"] = elapsed
    return history_dict, elapsed


def save_model_history(histories, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(histories, file, indent=2)


def save_model_specific_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def set_random_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)


def main(overwrite=False, artifact_set="wikitext2"):
    if artifact_set not in {"wikitext2", "demo", "demo_v2"}:
        raise ValueError("Training artifact_set must be 'wikitext2', 'demo', or 'demo_v2'.")
    paths = get_artifact_paths(artifact_set)
    if artifact_set == "demo_v2":
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        model_vocab_size = int(metadata["effective_vocabulary_size"])
        model_specs = (
            ("rnn", lambda: build_rnn_model(model_vocab_size), paths["rnn_model"], paths["rnn_history"]),
            ("lstm", lambda: build_lstm_model(model_vocab_size), paths["lstm_model"], paths["lstm_history"]),
        )
        training_epochs = DEMO_V2_EPOCHS
    else:
        model_specs = (
            ("rnn", build_rnn_model, paths["rnn_model"], paths["rnn_history"]),
            ("lstm", build_lstm_model, paths["lstm_model"], paths["lstm_history"]),
        )
        training_epochs = EPOCHS
    existing_models = [
        path for _, _, path, _ in model_specs if path.exists()
    ]
    if len(existing_models) == len(model_specs) and not overwrite:
        paths = ", ".join(str(path) for path in existing_models)
        raise FileExistsError(
            f"Both {artifact_set} model checkpoints already exist and will not be overwritten. "
            f"Pass --overwrite to retrain them. Existing: {paths}"
        )

    set_random_seeds()
    x_train, y_train, x_valid, y_valid = load_preprocessed_data(artifact_set)

    all_histories = {}
    for model_name, model_builder, checkpoint_path, history_path in model_specs:
        if checkpoint_path.exists() and not overwrite:
            if history_path.exists():
                all_histories[model_name] = json.loads(history_path.read_text(encoding="utf-8"))
            print(f"Keeping existing {model_name.upper()} checkpoint: {checkpoint_path}")
            continue

        model = model_builder()
        history, elapsed = train_model(
            model,
            x_train,
            y_train,
            x_valid,
            y_valid,
            checkpoint_path,
            epochs=training_epochs,
        )
        all_histories[model_name] = history
        save_model_specific_history(history, history_path)
        save_model_history(all_histories, paths["train_history"])
        save_training_plot(all_histories, paths["training_plot"])
        print(f"{model_name.upper()} training time: {elapsed:.2f} seconds")
        print(f"Saved best {model_name.upper()} checkpoint to {checkpoint_path}")

    print(f"Training artifacts are saved under {paths['train_history'].parent} and {paths['rnn_model'].parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN and LSTM models on prepared WikiText-2 data.")
    parser.add_argument("--artifact-set", choices=("wikitext2", "demo", "demo_v2"), default="wikitext2")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model checkpoints for the selected artifact set.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite, artifact_set=args.artifact_set)
