import argparse
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    get_artifact_paths,
)


def load_model_and_data(artifact_set="synthetic"):
    paths = get_artifact_paths(artifact_set)
    model_paths = {"rnn": paths["rnn_model"], "lstm": paths["lstm_model"]}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Trained {name.upper()} model is missing: {path}")

    if not paths["processed_test"].exists():
        raise FileNotFoundError(f"Processed test data is missing: {paths['processed_test']}")
    if paths["metadata"] is not None:
        if not paths["metadata"].exists():
            raise FileNotFoundError(f"Preprocessing metadata is missing: {paths['metadata']}")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("dataset") != "Salesforce/wikitext/wikitext-2-raw-v1":
            raise ValueError(f"Unexpected dataset metadata for {artifact_set} evaluation.")
        if metadata.get("sequence_length") != SEQUENCE_LENGTH:
            raise ValueError("Preprocessing sequence length does not match the model configuration.")
        if not paths["tokenizer"].exists():
            raise FileNotFoundError(f"Tokenizer is missing: {paths['tokenizer']}")
        with paths["tokenizer"].open("rb") as file:
            tokenizer = pickle.load(file)
        if artifact_set == "demo_v2":
            expected_vocab_size = max(tokenizer.word_index.values(), default=0) + 1
            if metadata.get("tokenizer_fit_split") != "demo_v2 training text only":
                raise ValueError("Demo V2 tokenizer metadata does not confirm training-split-only fitting.")
            if metadata.get("effective_vocabulary_size") != expected_vocab_size:
                raise ValueError("Demo V2 metadata does not match the tokenizer's effective vocabulary size.")
        else:
            expected_vocab_size = VOCAB_SIZE
            if getattr(tokenizer, "num_words", None) != VOCAB_SIZE:
                raise ValueError(f"Tokenizer vocabulary limit must be {VOCAB_SIZE}.")
            if artifact_set == "demo" and metadata.get("tokenizer_fit_split") != "demo training text only":
                raise ValueError("Demo tokenizer metadata does not confirm training-split-only fitting.")
    else:
        expected_vocab_size = VOCAB_SIZE

    with np.load(paths["processed_test"]) as test_data:
        x_test = test_data["x"]
        y_test = test_data["y"]
    if x_test.ndim != 2 or x_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"Test inputs must have shape (samples, {SEQUENCE_LENGTH}).")
    if y_test.ndim != 1 or len(x_test) != len(y_test) or not len(y_test):
        raise ValueError("Test inputs and targets must be non-empty and aligned.")
    if (
        x_test.min() < 0
        or y_test.min() < 0
        or x_test.max() >= expected_vocab_size
        or y_test.max() >= expected_vocab_size
    ):
        raise ValueError("Test data contains token IDs outside the model vocabulary.")
    if artifact_set == "demo" and len(y_test) != 10_000:
        raise ValueError("Demo evaluation requires the prepared 10,000-sequence test split.")
    if artifact_set == "demo_v2":
        expected_test_count = metadata.get("actual_sequences", {}).get("test")
        if expected_test_count != len(y_test):
            raise ValueError("Demo V2 test data does not match preprocessing metadata.")

    model_map = {name: tf.keras.models.load_model(path) for name, path in model_paths.items()}
    for name, model in model_map.items():
        if int(model.output_shape[-1]) != expected_vocab_size:
            raise ValueError(
                f"{name.upper()} output vocabulary ({model.output_shape[-1]}) "
                f"does not match expected size {expected_vocab_size}."
            )
    return model_map, x_test, y_test, model_paths, paths


def compute_topk_accuracy(y_true, probabilities):
    y_true = np.asarray(y_true)
    total = 0
    for idx, label in enumerate(y_true):
        top5 = np.argsort(probabilities[idx])[::-1][:5]
        if int(label) in top5:
            total += 1
    return total / len(y_true)


def evaluate_model(model, x_test, y_test):
    start_time = time.perf_counter()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    evaluation_time = time.perf_counter() - start_time
    top1 = float(acc)
    top5 = float(compute_topk_accuracy(y_test, probabilities))
    perplexity = float(np.exp(loss))
    return {
        "test_loss": float(loss),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "perplexity": perplexity,
        "evaluation_time_seconds": float(evaluation_time),
    }


def save_metrics(metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_model_comparison_plot(metrics, path):
    labels = ["RNN", "LSTM"]
    test_loss = [metrics["rnn"]["test_loss"], metrics["lstm"]["test_loss"]]
    top1 = [metrics["rnn"]["top1_accuracy"], metrics["lstm"]["top1_accuracy"]]
    perplexity = [metrics["rnn"]["perplexity"], metrics["lstm"]["perplexity"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(labels, test_loss, color=["steelblue", "darkorange"])
    axes[0].set_title("Test loss")
    axes[1].bar(labels, top1, color=["steelblue", "darkorange"])
    axes[1].set_title("Top-1 accuracy")
    axes[2].bar(labels, perplexity, color=["steelblue", "darkorange"])
    axes[2].set_title("Perplexity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(artifact_set="synthetic"):
    model_map, x_test, y_test, model_paths, paths = load_model_and_data(artifact_set)
    history = json.loads(paths["train_history"].read_text(encoding="utf-8")) if paths["train_history"].exists() else {}

    metrics = {}
    for name, model in model_map.items():
        model_metrics = evaluate_model(model, x_test, y_test)
        model_history = history.get(name, {})
        model_metrics["training_time_seconds"] = model_history.get("training_time_seconds", 0.0)
        validation_losses = model_history.get("val_loss", [])
        if validation_losses:
            model_metrics["best_epoch"] = int(np.argmin(validation_losses)) + 1
        model_metrics["model_size_bytes"] = model_paths[name].stat().st_size
        model_metrics["vocabulary_size"] = int(model.output_shape[-1])
        metrics[name] = model_metrics

    save_metrics(metrics, paths["metrics"])
    save_model_comparison_plot(metrics, paths["comparison_plot"])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained next-word prediction models.")
    parser.add_argument(
        "--artifact-set",
        choices=("synthetic", "wikitext2", "demo", "demo_v2"),
        default="synthetic",
    )
    args = parser.parse_args()
    main(artifact_set=args.artifact_set)
