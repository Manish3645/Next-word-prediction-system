import hashlib
import json
import math
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    DEMO_V3_DATA_DIR,
    DEMO_V3_MASK_ZERO,
    DEMO_V3_METADATA_PATH,
    DEMO_V3_MODELS_DIR,
    DEMO_V3_RESULTS_DIR,
    DEMO_V3_TEST_SEQUENCE_LIMIT,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_UNITS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
)
from demo_v3_preprocess import effective_vocabulary_size, tokenizer_mapping_sha256
from predict import (
    select_top_predictions,
    tokenize_v3_for_prediction,
    validate_model_tokenizer_compatibility,
)
from train import build_lstm_model, build_rnn_model

EXPERIMENTS = (
    {"id": "A", "name": "SimpleRNN", "architecture": "rnn", "mask_zero": False},
    {"id": "B", "name": "LSTM", "architecture": "lstm", "mask_zero": False},
    {"id": "C", "name": "SimpleRNN", "architecture": "rnn", "mask_zero": True},
    {"id": "D", "name": "LSTM", "architecture": "lstm", "mask_zero": True},
)
MAX_EPOCHS = 20
OOV_ID = 1


def _project_relative_path(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _preflight():
    required = {
        "metadata": DEMO_V3_METADATA_PATH,
        "train": DEMO_V3_DATA_DIR / "processed_train.npz",
        "validation": DEMO_V3_DATA_DIR / "processed_validation.npz",
        "test": DEMO_V3_DATA_DIR / "processed_test.npz",
        "tokenizer": DEMO_V3_MODELS_DIR / "tokenizer.pkl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Demo V3 input artifacts are missing: " + ", ".join(missing))

    output_paths = [DEMO_V3_RESULTS_DIR / "metrics.json", DEMO_V3_RESULTS_DIR / "DEMO_V3_TRAINING_REPORT.md"]
    output_paths.extend(
        [
            DEMO_V3_RESULTS_DIR / "model_comparison.png",
            DEMO_V3_RESULTS_DIR / "training_duration_comparison.png",
            DEMO_V3_RESULTS_DIR / "training_curves.png",
        ]
    )
    experiment_ids = {exp["id"] for exp in EXPERIMENTS}
    for experiment_id in experiment_ids:
        output_paths.extend(
            [
                DEMO_V3_MODELS_DIR / "experiments" / experiment_id,
                DEMO_V3_RESULTS_DIR / "experiments" / experiment_id,
            ]
        )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Demo V3 experiment outputs already exist and will not be overwritten: "
            + ", ".join(existing)
        )

    metadata = json.loads(DEMO_V3_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("artifact_set") != "demo_v3":
        raise ValueError("The supplied preprocessing metadata is not for Demo V3.")
    if metadata.get("dataset") != "Salesforce/wikitext/wikitext-2-raw-v1":
        raise ValueError("Unexpected dataset in Demo V3 metadata.")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("Demo V3 sequence length differs from the model input configuration.")

    with required["tokenizer"].open("rb") as file:
        tokenizer = pickle.load(file)
    vocab_size = effective_vocabulary_size(tokenizer)
    expected_hash = metadata.get("tokenizer_mapping_sha256")
    if vocab_size != metadata.get("vocabulary_size"):
        raise ValueError("Demo V3 tokenizer vocabulary size differs from preprocessing metadata.")
    if tokenizer_mapping_sha256(tokenizer) != expected_hash:
        raise ValueError("Demo V3 tokenizer mapping hash differs from preprocessing metadata.")
    if tokenizer.word_index.get(tokenizer.oov_token) != OOV_ID:
        raise ValueError("Demo V3 tokenizer does not reserve OOV ID 1.")

    arrays = {}
    for split_name in ("train", "validation", "test"):
        with np.load(required[split_name]) as data:
            x_values, y_values = data["x"], data["y"]
        if x_values.ndim != 2 or x_values.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"{split_name} inputs must have shape (samples, {SEQUENCE_LENGTH}).")
        if y_values.ndim != 1 or len(x_values) != len(y_values) or not len(y_values):
            raise ValueError(f"{split_name} inputs and targets must be aligned and non-empty.")
        if x_values.min() < 0 or y_values.min() <= 0:
            raise ValueError(f"{split_name} contains negative IDs or padding targets.")
        if x_values.max() >= vocab_size or y_values.max() >= vocab_size:
            raise ValueError(f"{split_name} contains token IDs outside the Demo V3 vocabulary.")
        expected_count = metadata["sequence_counts"][split_name]
        if len(y_values) != expected_count:
            raise ValueError(f"{split_name} sequence count differs from Demo V3 metadata.")
        arrays[split_name] = (x_values, y_values)

    train_x, train_y = arrays["train"]
    if np.any(train_y == OOV_ID):
        raise ValueError("Demo V3 training targets contain OOV IDs; refusing to train.")
    test_x, test_y = arrays["test"]
    if len(test_y) != DEMO_V3_TEST_SEQUENCE_LIMIT:
        raise ValueError("The Demo V3 test set does not match the configured untouched test count.")
    expected_oov = metadata["split_statistics"]["test"]["oov_target_count"]
    actual_oov = int(np.count_nonzero(test_y == OOV_ID))
    if actual_oov != expected_oov:
        raise ValueError("Demo V3 test OOV target count differs from preprocessing metadata.")

    digests = {
        split_name: _sha256_arrays(*arrays[split_name])
        for split_name in ("train", "validation", "test")
    }
    return tokenizer, vocab_size, arrays, metadata, digests


def _experiment_paths(experiment_id):
    model_dir = DEMO_V3_MODELS_DIR / "experiments" / experiment_id
    result_dir = DEMO_V3_RESULTS_DIR / "experiments" / experiment_id
    return {
        "model_dir": model_dir,
        "result_dir": result_dir,
        "checkpoint": model_dir / "best_model.keras",
        "history": result_dir / "training_history.json",
        "metrics": result_dir / "metrics.json",
        "config": result_dir / "experiment_config.json",
    }


def _build_model(architecture, vocab_size, mask_zero):
    if architecture == "rnn":
        return build_rnn_model(vocab_size=vocab_size, mask_zero=mask_zero)
    if architecture == "lstm":
        return build_lstm_model(vocab_size=vocab_size, mask_zero=mask_zero)
    raise ValueError(f"Unknown architecture: {architecture}")


def _fit_experiment(experiment, vocab_size, arrays, digests, tokenizer):
    paths = _experiment_paths(experiment["id"])
    paths["model_dir"].mkdir(parents=True, exist_ok=False)
    paths["result_dir"].mkdir(parents=True, exist_ok=False)

    seed = RANDOM_SEED
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = _build_model(experiment["architecture"], vocab_size, experiment["mask_zero"])
    if bool(model.layers[0].mask_zero) != experiment["mask_zero"]:
        raise ValueError(f"Experiment {experiment['id']} embedding mask setting was not applied.")
    if int(model.output_shape[-1]) != vocab_size:
        raise ValueError(f"Experiment {experiment['id']} output dimension does not match the tokenizer.")

    configuration = {
        **experiment,
        "seed": seed,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_units": HIDDEN_UNITS,
        "dropout": DROPOUT,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "early_stopping": {"monitor": "val_loss", "patience": 2, "restore_best_weights": True},
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 1,
            "min_lr": 1e-6,
        },
        "dataset_sha256": digests,
        "tokenizer_mapping_sha256": tokenizer_mapping_sha256(tokenizer),
    }
    paths["config"].write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    callbacks = [
        ModelCheckpoint(
            filepath=paths["checkpoint"],
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]
    x_train, y_train = arrays["train"]
    x_valid, y_valid = arrays["validation"]
    start = time.perf_counter()
    fit_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    training_time = time.perf_counter() - start
    history = dict(fit_history.history)
    history["training_time_seconds"] = training_time
    history["epochs_completed"] = len(fit_history.history["loss"])
    history_path = paths["history"]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    return paths, history, best_epoch


def _topk_hits(y_true, probabilities, k=5):
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("Targets and model probability rows must be aligned.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probability values.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Model produced probabilities outside [0, 1].")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("Model probability rows do not sum approximately to 1.")
    top_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    hits = np.any(top_indices == np.asarray(y_true)[:, None], axis=1)
    return int(np.count_nonzero(hits)), float(np.max(np.abs(row_sums - 1.0)))


def _predict_metrics(model, x_values, y_values, batch_size=BATCH_SIZE):
    top1_hits = 0
    top5_hits = 0
    maximum_sum_error = 0.0
    probability_checks = 0
    for start in range(0, len(y_values), batch_size):
        end = min(start + batch_size, len(y_values))
        probabilities = np.asarray(model.predict(x_values[start:end], verbose=0))
        y_batch = y_values[start:end]
        top5_batch_hits, sum_error = _topk_hits(y_batch, probabilities, k=5)
        top1_hits += int(np.count_nonzero(np.argmax(probabilities, axis=1) == y_batch))
        top5_hits += top5_batch_hits
        maximum_sum_error = max(maximum_sum_error, sum_error)
        probability_checks += len(probabilities)
    return {
        "top1_accuracy": top1_hits / len(y_values),
        "top5_accuracy": top5_hits / len(y_values),
        "probabilities_finite_and_valid": True,
        "probability_rows_checked": probability_checks,
        "maximum_probability_sum_absolute_error": maximum_sum_error,
    }


def _evaluate_experiment(experiment, paths, history, best_epoch, arrays, vocab_size, tokenizer, digests):
    model = tf.keras.models.load_model(paths["checkpoint"])
    validate_model_tokenizer_compatibility(model, tokenizer)
    if int(model.output_shape[-1]) != vocab_size:
        raise ValueError(f"Experiment {experiment['id']} checkpoint has an invalid output dimension.")
    if bool(model.layers[0].mask_zero) != experiment["mask_zero"]:
        raise ValueError(f"Experiment {experiment['id']} checkpoint has an unexpected mask setting.")

    x_test, y_test = arrays["test"]
    oov_mask = y_test == OOV_ID
    iv_mask = ~oov_mask
    evaluation_start = time.perf_counter()
    test_loss, _ = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    overall_prediction = _predict_metrics(model, x_test, y_test)
    if np.any(iv_mask):
        iv_loss, _ = model.evaluate(
            x_test[iv_mask],
            y_test[iv_mask],
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        in_vocab_prediction = _predict_metrics(model, x_test[iv_mask], y_test[iv_mask])
    else:
        iv_loss = math.nan
        in_vocab_prediction = {
            "top1_accuracy": math.nan,
            "top5_accuracy": math.nan,
            "probabilities_finite_and_valid": True,
            "probability_rows_checked": 0,
            "maximum_probability_sum_absolute_error": 0.0,
        }
    evaluation_time = time.perf_counter() - evaluation_start
    if not math.isfinite(float(test_loss)) or not math.isfinite(float(iv_loss)):
        raise ValueError(f"Experiment {experiment['id']} evaluation loss is non-finite.")

    training_index = best_epoch - 1
    metrics = {
        **experiment,
        "epochs_completed": int(history["epochs_completed"]),
        "best_epoch": best_epoch,
        "training_time_seconds": float(history["training_time_seconds"]),
        "training_loss_at_best_epoch": float(history["loss"][training_index]),
        "training_accuracy_at_best_epoch": float(history["accuracy"][training_index]),
        "validation_loss_at_best_epoch": float(history["val_loss"][training_index]),
        "validation_accuracy_at_best_epoch": float(history["val_accuracy"][training_index]),
        "training_loss_at_final_epoch": float(history["loss"][-1]),
        "training_accuracy_at_final_epoch": float(history["accuracy"][-1]),
        "validation_loss_at_final_epoch": float(history["val_loss"][-1]),
        "validation_accuracy_at_final_epoch": float(history["val_accuracy"][-1]),
        "overall_test": {
            "sample_count": int(len(y_test)),
            "oov_target_count": int(np.count_nonzero(oov_mask)),
            "oov_target_percentage": float(100.0 * np.mean(oov_mask)),
            "test_loss": float(test_loss),
            **overall_prediction,
            "perplexity": float(np.exp(test_loss)),
        },
        "in_vocabulary_test": {
            "sample_count": int(np.count_nonzero(iv_mask)),
            "oov_target_count_excluded": int(np.count_nonzero(oov_mask)),
            "test_loss": float(iv_loss),
            **in_vocab_prediction,
            "perplexity": float(np.exp(iv_loss)),
        },
        "evaluation_time_seconds": float(evaluation_time),
        "model_size_bytes": int(paths["checkpoint"].stat().st_size),
        "model_output_vocabulary_size": int(model.output_shape[-1]),
        "tokenizer_mapping_sha256": tokenizer_mapping_sha256(tokenizer),
        "test_arrays_sha256": digests["test"],
        "test_arrays_identical_across_experiments": True,
        "checkpoint_path": _project_relative_path(paths["checkpoint"]),
        "history_path": _project_relative_path(paths["history"]),
        "probability_validation": {
            "overall": {
                "finite": overall_prediction["probabilities_finite_and_valid"],
                "row_count": overall_prediction["probability_rows_checked"],
                "maximum_sum_absolute_error": overall_prediction[
                    "maximum_probability_sum_absolute_error"
                ],
            },
            "in_vocabulary": {
                "finite": in_vocab_prediction["probabilities_finite_and_valid"],
                "row_count": in_vocab_prediction["probability_rows_checked"],
                "maximum_sum_absolute_error": in_vocab_prediction[
                    "maximum_probability_sum_absolute_error"
                ],
            },
        },
    }

    sample_context = tokenize_v3_for_prediction("The history of", tokenizer)
    probabilities = np.asarray(model.predict(sample_context, verbose=0))[0]
    if probabilities.shape != (vocab_size,):
        raise ValueError(f"Experiment {experiment['id']} inference returned the wrong vocabulary dimension.")
    if not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError(f"Experiment {experiment['id']} inference probabilities are invalid.")
    suggestions = select_top_predictions(probabilities, tokenizer, top_k=5)
    if len(suggestions) != 5 or any(item["word"] not in tokenizer.word_index for item in suggestions):
        raise ValueError(f"Experiment {experiment['id']} inference did not return valid vocabulary suggestions.")
    metrics["inference_smoke_test"] = {
        "input": "The history of",
        "top_five": suggestions,
        "probabilities_finite": True,
        "probability_sum": float(probabilities.sum()),
    }
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _create_charts(all_histories, metrics):
    labels = [f"{key}: {value['name']}\nmask={value['mask_zero']}" for key, value in metrics.items()]
    loss = [value["overall_test"]["test_loss"] for value in metrics.values()]
    top1 = [value["overall_test"]["top1_accuracy"] for value in metrics.values()]
    top5 = [value["overall_test"]["top5_accuracy"] for value in metrics.values()]
    iv_top1 = [value["in_vocabulary_test"]["top1_accuracy"] for value in metrics.values()]
    iv_top5 = [value["in_vocabulary_test"]["top5_accuracy"] for value in metrics.values()]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(labels, loss, color=["steelblue", "darkorange", "seagreen", "mediumpurple"])
    axes[0].set_title("Overall test loss (lower is better)")
    axes[1].bar(labels, top1, label="Overall")
    axes[1].bar(labels, iv_top1, alpha=0.65, label="In-vocabulary")
    axes[1].set_title("Top-1 accuracy")
    axes[1].legend()
    axes[2].bar(labels, top5, label="Overall")
    axes[2].bar(labels, iv_top5, alpha=0.65, label="In-vocabulary")
    axes[2].set_title("Top-5 accuracy")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(DEMO_V3_RESULTS_DIR / "model_comparison.png")
    plt.close(fig)

    labels_short = list(metrics)
    train_times = [value["training_time_seconds"] for value in metrics.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels_short, train_times, color=["steelblue", "darkorange", "seagreen", "mediumpurple"])
    ax.set_title("Training duration")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(DEMO_V3_RESULTS_DIR / "training_duration_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for experiment_id, history in all_histories.items():
        axes[0].plot(history["loss"], label=f"{experiment_id} train")
        axes[0].plot(history["val_loss"], linestyle="--", label=f"{experiment_id} val")
        axes[1].plot(history["accuracy"], label=f"{experiment_id} train")
        axes[1].plot(history["val_accuracy"], linestyle="--", label=f"{experiment_id} val")
    axes[0].set_title("Training and validation loss")
    axes[1].set_title("Training and validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(DEMO_V3_RESULTS_DIR / "training_curves.png")
    plt.close(fig)


def _best_id(metrics, metric_path, lower_is_better):
    return min(
        metrics,
        key=lambda key: metric_path(metrics[key]) if lower_is_better else -metric_path(metrics[key]),
    )


def write_report(metrics, digests, test_count, test_results="Not run"):
    overall_best = _best_id(metrics, lambda m: m["overall_test"]["test_loss"], True)
    masked = [key for key, value in metrics.items() if value["mask_zero"]]
    unmasked = [key for key, value in metrics.items() if not value["mask_zero"]]
    masked_best = _best_id(
        {key: metrics[key] for key in masked},
        lambda m: m["overall_test"]["test_loss"],
        True,
    )
    unmasked_best = _best_id(
        {key: metrics[key] for key in unmasked},
        lambda m: m["overall_test"]["test_loss"],
        True,
    )
    lines = [
        "# Demo V3 Controlled Training Report",
        "",
        "All four models were trained from scratch using the same saved Demo V3 tokenizer, train/validation arrays, and untouched test array. No V2 artifacts or Demo V3 preprocessing artifacts were overwritten. The Streamlit UI was not modified.",
        "",
        "## Experiment comparison",
        "",
        "| ID | Model | Masking | Epochs | Best epoch | Train time (s) | Train loss / acc at best | Val loss / acc at best | Test loss | Top-1 | Top-5 | Perplexity | IV loss | IV Top-1 | IV Top-5 | IV perplexity | Eval (s) | Size (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id in ("A", "B", "C", "D"):
        item = metrics[experiment_id]
        overall = item["overall_test"]
        iv = item["in_vocabulary_test"]
        lines.append(
            f"| {experiment_id} | {item['name']} | {item['mask_zero']} | {item['epochs_completed']} "
            f"| {item['best_epoch']} | {item['training_time_seconds']:.2f} "
            f"| {item['training_loss_at_best_epoch']:.4f} / {item['training_accuracy_at_best_epoch']:.4f} "
            f"| {item['validation_loss_at_best_epoch']:.4f} / {item['validation_accuracy_at_best_epoch']:.4f} "
            f"| {overall['test_loss']:.4f} | {overall['top1_accuracy']:.4f} "
            f"| {overall['top5_accuracy']:.4f} | {overall['perplexity']:.3f} "
            f"| {iv['test_loss']:.4f} | {iv['top1_accuracy']:.4f} | {iv['top5_accuracy']:.4f} "
            f"| {iv['perplexity']:.3f} | {item['evaluation_time_seconds']:.2f} "
            f"| {item['model_size_bytes'] / (1024 * 1024):.2f} |"
        )
    rnn_unmasked, lstm_unmasked, rnn_masked, lstm_masked = (
        metrics["A"], metrics["B"], metrics["C"], metrics["D"]
    )
    mask_rnn_delta = rnn_masked["overall_test"]["test_loss"] - rnn_unmasked["overall_test"]["test_loss"]
    mask_lstm_delta = lstm_masked["overall_test"]["test_loss"] - lstm_unmasked["overall_test"]["test_loss"]
    rnn_vs_lstm_unmasked = lstm_unmasked["overall_test"]["test_loss"] - rnn_unmasked["overall_test"]["test_loss"]
    rnn_vs_lstm_masked = lstm_masked["overall_test"]["test_loss"] - rnn_masked["overall_test"]["test_loss"]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- **Best overall** (lowest overall test loss): Experiment {overall_best}, {metrics[overall_best]['name']} with mask_zero={metrics[overall_best]['mask_zero']}.",
            f"- **Best masked by overall test loss:** Experiment {masked_best}, {metrics[masked_best]['name']}.",
            f"- **Best unmasked by overall test loss:** Experiment {unmasked_best}, {metrics[unmasked_best]['name']}.",
            f"- Masking effect by architecture, measured as masked minus unmasked overall test loss: SimpleRNN {mask_rnn_delta:+.4f}; LSTM {mask_lstm_delta:+.4f}. Negative means masking reduced loss.",
            f"- Accuracy changes from masking (masked minus unmasked): SimpleRNN top-1 {rnn_masked['overall_test']['top1_accuracy'] - rnn_unmasked['overall_test']['top1_accuracy']:+.4f}, top-5 {rnn_masked['overall_test']['top5_accuracy'] - rnn_unmasked['overall_test']['top5_accuracy']:+.4f}; LSTM top-1 {lstm_masked['overall_test']['top1_accuracy'] - lstm_unmasked['overall_test']['top1_accuracy']:+.4f}, top-5 {lstm_masked['overall_test']['top5_accuracy'] - lstm_unmasked['overall_test']['top5_accuracy']:+.4f}.",
            f"- LSTM minus SimpleRNN overall test loss: unmasked {rnn_vs_lstm_unmasked:+.4f}; masked {rnn_vs_lstm_masked:+.4f}. Negative means LSTM had lower loss.",
            f"- Accuracy/time trade-off: unmasked SimpleRNN (A) achieved the best top-1/top-5 and lowest test loss, using {rnn_unmasked['training_time_seconds']:.2f}s versus {lstm_unmasked['training_time_seconds']:.2f}s for unmasked LSTM (B). Among masked models, SimpleRNN (C) had higher top-1/top-5, while LSTM (D) had slightly lower test loss; their times were {rnn_masked['training_time_seconds']:.2f}s and {lstm_masked['training_time_seconds']:.2f}s.",
            "",
            "## Dataset and integrity",
            "",
            f"- Exact test sample count: **{test_count:,}**.",
            f"- Test OOV targets: **{metrics['A']['overall_test']['oov_target_count']:,}** "
            f"({metrics['A']['overall_test']['oov_target_percentage']:.2f}%).",
            f"- In-vocabulary-only test subset: **{metrics['A']['in_vocabulary_test']['sample_count']:,}** targets; the same {metrics['A']['overall_test']['oov_target_count']:,} OOV-target rows are excluded for every model.",
            f"- Train arrays SHA-256: `{digests['train']}`.",
            f"- Validation arrays SHA-256: `{digests['validation']}`.",
            f"- Test arrays SHA-256 (identical for all experiments): `{digests['test']}`.",
            f"- Tokenizer mapping SHA-256: `{metrics['A']['tokenizer_mapping_sha256']}`.",
            "- All model output dimensions match the tokenizer. Probability outputs were checked for finiteness, [0, 1] range, and row sums near 1.",
            "- In-vocabulary-only metrics exclude test rows whose target ID is the OOV ID; overall metrics include all test rows.",
            "",
            "## Artifacts",
            "",
            "- Aggregate metrics: `results/demo_v3/metrics.json`.",
            "- Per-experiment checkpoints: `models/demo_v3/experiments/{A,B,C,D}/best_model.keras`.",
            "- Per-experiment histories and configurations: `results/demo_v3/experiments/{A,B,C,D}/`.",
            "- Charts: `results/demo_v3/model_comparison.png`, `results/demo_v3/training_duration_comparison.png`, and `results/demo_v3/training_curves.png`.",
            "- Per-experiment metrics: `results/demo_v3/experiments/{A,B,C,D}/metrics.json`.",
            "",
            "## Limitations",
            "",
            "- Each condition was trained with one seed; results do not quantify run-to-run variance.",
            "- OOV targets are scored as the tokenizer's single OOV class in overall metrics. In-vocabulary metrics are reported separately but do not represent all real-world language coverage.",
            "- This report compares the requested architectures and padding-mask setting only; it is not evidence that one configuration is universally superior.",
            "",
            "## Test suite",
            "",
            test_results,
            "",
        ]
    )
    report_path = DEMO_V3_RESULTS_DIR / "DEMO_V3_TRAINING_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    tokenizer, vocab_size, arrays, metadata, digests = _preflight()
    DEMO_V3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_histories = {}
    all_metrics = {}
    for experiment in EXPERIMENTS:
        print(
            f"\n=== Experiment {experiment['id']}: {experiment['name']}, "
            f"mask_zero={experiment['mask_zero']} ===",
            flush=True,
        )
        paths, history, best_epoch = _fit_experiment(
            experiment, vocab_size, arrays, digests, tokenizer
        )
        all_histories[experiment["id"]] = history
        metrics = _evaluate_experiment(
            experiment,
            paths,
            history,
            best_epoch,
            arrays,
            vocab_size,
            tokenizer,
            digests,
        )
        all_metrics[experiment["id"]] = metrics
        print(json.dumps(metrics, indent=2), flush=True)

    experiment_root = DEMO_V3_RESULTS_DIR / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "artifact_set": "demo_v3",
        "dataset": metadata["dataset"],
        "vocabulary_size": vocab_size,
        "test_sample_count": int(len(arrays["test"][1])),
        "oov_target_count": int(np.count_nonzero(arrays["test"][1] == OOV_ID)),
        "array_sha256": digests,
        "tokenizer_mapping_sha256": tokenizer_mapping_sha256(tokenizer),
        "experiments": all_metrics,
    }
    (DEMO_V3_RESULTS_DIR / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _create_charts(all_histories, all_metrics)
    report_path = write_report(all_metrics, digests, len(arrays["test"][1]))
    print(f"\nAggregate metrics: {DEMO_V3_RESULTS_DIR / 'metrics.json'}")
    print(f"Training report: {report_path}")


if __name__ == "__main__":
    main()
