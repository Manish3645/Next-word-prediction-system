import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
WIKITEXT2_DATA_DIR = DATA_DIR / "wikitext2"
WIKITEXT2_MODELS_DIR = MODELS_DIR / "wikitext2"
WIKITEXT2_RESULTS_DIR = RESULTS_DIR / "wikitext2"
DEMO_DATA_DIR = DATA_DIR / "demo"
DEMO_MODELS_DIR = MODELS_DIR / "demo"
DEMO_RESULTS_DIR = RESULTS_DIR / "demo"
DEMO_V2_DATA_DIR = DATA_DIR / "demo_v2"
DEMO_V2_MODELS_DIR = MODELS_DIR / "demo_v2"
DEMO_V2_RESULTS_DIR = RESULTS_DIR / "demo_v2"
DEMO_V3_DATA_DIR = DATA_DIR / "demo_v3"
DEMO_V3_MODELS_DIR = MODELS_DIR / "demo_v3"
DEMO_V3_RESULTS_DIR = RESULTS_DIR / "demo_v3"

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
RANDOM_SEED = 42

VOCAB_SIZE = 12_000
SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 128
HIDDEN_UNITS = 64
DROPOUT = 0.2
BATCH_SIZE = 128
EPOCHS = 5
DEMO_V2_EPOCHS = 20
DEMO_V3_TRAIN_SEQUENCE_LIMIT = 50_000
DEMO_V3_VALID_SEQUENCE_LIMIT = 10_000
DEMO_V3_TEST_SEQUENCE_LIMIT = 10_000
DEMO_V3_MASK_ZERO = True

TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
RNN_MODEL_PATH = MODELS_DIR / "rnn_model.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
TRAIN_HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVAL_METRICS_PATH = RESULTS_DIR / "metrics.json"
TRAINING_PLOT_PATH = RESULTS_DIR / "training_history.png"
MODEL_COMPARISON_PLOT_PATH = RESULTS_DIR / "model_comparison.png"

RAW_TRAIN_PATH = WIKITEXT2_DATA_DIR / "train.txt"
RAW_VALID_PATH = WIKITEXT2_DATA_DIR / "valid.txt"
RAW_TEST_PATH = WIKITEXT2_DATA_DIR / "test.txt"
PROCESSED_TRAIN_PATH = WIKITEXT2_DATA_DIR / "processed_train.npz"
PROCESSED_VALID_PATH = WIKITEXT2_DATA_DIR / "processed_validation.npz"
PROCESSED_TEST_PATH = WIKITEXT2_DATA_DIR / "processed_test.npz"
WIKITEXT2_METADATA_PATH = WIKITEXT2_DATA_DIR / "preprocessing_metadata.json"

WIKITEXT2_TOKENIZER_PATH = WIKITEXT2_MODELS_DIR / "tokenizer.pkl"
WIKITEXT2_RNN_MODEL_PATH = WIKITEXT2_MODELS_DIR / "rnn_model.keras"
WIKITEXT2_LSTM_MODEL_PATH = WIKITEXT2_MODELS_DIR / "lstm_model.keras"
WIKITEXT2_TRAIN_HISTORY_PATH = WIKITEXT2_RESULTS_DIR / "training_history.json"
WIKITEXT2_RNN_HISTORY_PATH = WIKITEXT2_RESULTS_DIR / "rnn_training_history.json"
WIKITEXT2_LSTM_HISTORY_PATH = WIKITEXT2_RESULTS_DIR / "lstm_training_history.json"
WIKITEXT2_TRAINING_PLOT_PATH = WIKITEXT2_RESULTS_DIR / "training_history.png"
WIKITEXT2_EVAL_METRICS_PATH = WIKITEXT2_RESULTS_DIR / "metrics.json"
WIKITEXT2_MODEL_COMPARISON_PLOT_PATH = WIKITEXT2_RESULTS_DIR / "model_comparison.png"

DEMO_PROCESSED_TRAIN_PATH = DEMO_DATA_DIR / "processed_train.npz"
DEMO_PROCESSED_VALID_PATH = DEMO_DATA_DIR / "processed_validation.npz"
DEMO_PROCESSED_TEST_PATH = DEMO_DATA_DIR / "processed_test.npz"
DEMO_METADATA_PATH = DEMO_DATA_DIR / "preprocessing_metadata.json"
DEMO_RAW_TRAIN_PATH = DEMO_DATA_DIR / "train.txt"
DEMO_RAW_VALID_PATH = DEMO_DATA_DIR / "valid.txt"
DEMO_RAW_TEST_PATH = DEMO_DATA_DIR / "test.txt"
DEMO_TOKENIZER_PATH = DEMO_MODELS_DIR / "tokenizer.pkl"
DEMO_RNN_MODEL_PATH = DEMO_MODELS_DIR / "rnn_model.keras"
DEMO_LSTM_MODEL_PATH = DEMO_MODELS_DIR / "lstm_model.keras"
DEMO_TRAIN_HISTORY_PATH = DEMO_RESULTS_DIR / "training_history.json"
DEMO_RNN_HISTORY_PATH = DEMO_RESULTS_DIR / "rnn_training_history.json"
DEMO_LSTM_HISTORY_PATH = DEMO_RESULTS_DIR / "lstm_training_history.json"
DEMO_TRAINING_PLOT_PATH = DEMO_RESULTS_DIR / "training_history.png"
DEMO_EVAL_METRICS_PATH = DEMO_RESULTS_DIR / "metrics.json"
DEMO_MODEL_COMPARISON_PLOT_PATH = DEMO_RESULTS_DIR / "model_comparison.png"

DEMO_V2_PROCESSED_TRAIN_PATH = DEMO_V2_DATA_DIR / "processed_train.npz"
DEMO_V2_PROCESSED_VALID_PATH = DEMO_V2_DATA_DIR / "processed_validation.npz"
DEMO_V2_PROCESSED_TEST_PATH = DEMO_V2_DATA_DIR / "processed_test.npz"
DEMO_V2_RAW_TRAIN_PATH = DEMO_V2_DATA_DIR / "train.txt"
DEMO_V2_RAW_VALID_PATH = DEMO_V2_DATA_DIR / "valid.txt"
DEMO_V2_RAW_TEST_PATH = DEMO_V2_DATA_DIR / "test.txt"
DEMO_V2_TOKENIZER_PATH = DEMO_V2_MODELS_DIR / "tokenizer.pkl"
DEMO_V2_METADATA_PATH = DEMO_V2_MODELS_DIR / "metadata.json"
DEMO_V2_RNN_MODEL_PATH = DEMO_V2_MODELS_DIR / "rnn" / "rnn_model.keras"
DEMO_V2_LSTM_MODEL_PATH = DEMO_V2_MODELS_DIR / "lstm" / "lstm_model.keras"
DEMO_V2_TRAIN_HISTORY_PATH = DEMO_V2_RESULTS_DIR / "training_history.json"
DEMO_V2_RNN_HISTORY_PATH = DEMO_V2_RESULTS_DIR / "rnn_training_history.json"
DEMO_V2_LSTM_HISTORY_PATH = DEMO_V2_RESULTS_DIR / "lstm_training_history.json"
DEMO_V2_TRAINING_PLOT_PATH = DEMO_V2_RESULTS_DIR / "training_history.png"
DEMO_V2_EVAL_METRICS_PATH = DEMO_V2_RESULTS_DIR / "metrics.json"
DEMO_V2_MODEL_COMPARISON_PLOT_PATH = DEMO_V2_RESULTS_DIR / "model_comparison.png"
DEMO_V2_PREDICTION_AUDIT_PATH = DEMO_V2_RESULTS_DIR / "prediction_audit.json"

DEMO_V3_PROCESSED_TRAIN_PATH = DEMO_V3_DATA_DIR / "processed_train.npz"
DEMO_V3_PROCESSED_VALID_PATH = DEMO_V3_DATA_DIR / "processed_validation.npz"
DEMO_V3_PROCESSED_TEST_PATH = DEMO_V3_DATA_DIR / "processed_test.npz"
DEMO_V3_RAW_TRAIN_PATH = DEMO_V3_DATA_DIR / "train.txt"
DEMO_V3_RAW_VALID_PATH = DEMO_V3_DATA_DIR / "valid.txt"
DEMO_V3_RAW_TEST_PATH = DEMO_V3_DATA_DIR / "test.txt"
DEMO_V3_TOKENIZER_PATH = DEMO_V3_MODELS_DIR / "tokenizer.pkl"
DEMO_V3_METADATA_PATH = DEMO_V3_MODELS_DIR / "preprocessing_metadata.json"
DEMO_V3_PREPROCESSING_CONFIG_PATH = DEMO_V3_MODELS_DIR / "preprocessing_config.json"
DEMO_V3_DATASET_STATISTICS_PATH = DEMO_V3_RESULTS_DIR / "dataset_statistics.json"
DEMO_V3_PREPROCESSING_REPORT_PATH = DEMO_V3_RESULTS_DIR / "DEMO_V3_PREPROCESSING_REPORT.md"
DEMO_V3_RNN_MODEL_PATH = DEMO_V3_MODELS_DIR / "rnn_model.keras"
DEMO_V3_LSTM_MODEL_PATH = DEMO_V3_MODELS_DIR / "lstm_model.keras"


DEFAULT_ARTIFACT_SET = os.environ.get("NEXT_WORD_ARTIFACT_SET", "synthetic").lower()


def get_artifact_paths(artifact_set="synthetic"):
    """Return paths for one isolated model/data artifact set."""
    if artifact_set == "synthetic":
        return {
            "tokenizer": TOKENIZER_PATH,
            "rnn_model": RNN_MODEL_PATH,
            "lstm_model": LSTM_MODEL_PATH,
            "processed_train": DATA_DIR / "processed_train.npz",
            "processed_valid": DATA_DIR / "processed_validation.npz",
            "processed_test": DATA_DIR / "processed_test.npz",
            "metadata": None,
            "train_history": TRAIN_HISTORY_PATH,
            "rnn_history": RESULTS_DIR / "rnn_training_history.json",
            "lstm_history": RESULTS_DIR / "lstm_training_history.json",
            "training_plot": TRAINING_PLOT_PATH,
            "metrics": EVAL_METRICS_PATH,
            "comparison_plot": MODEL_COMPARISON_PLOT_PATH,
        }
    if artifact_set == "wikitext2":
        return {
            "tokenizer": WIKITEXT2_TOKENIZER_PATH,
            "rnn_model": WIKITEXT2_RNN_MODEL_PATH,
            "lstm_model": WIKITEXT2_LSTM_MODEL_PATH,
            "processed_train": PROCESSED_TRAIN_PATH,
            "processed_valid": PROCESSED_VALID_PATH,
            "processed_test": PROCESSED_TEST_PATH,
            "metadata": WIKITEXT2_METADATA_PATH,
            "train_history": WIKITEXT2_TRAIN_HISTORY_PATH,
            "rnn_history": WIKITEXT2_RNN_HISTORY_PATH,
            "lstm_history": WIKITEXT2_LSTM_HISTORY_PATH,
            "training_plot": WIKITEXT2_TRAINING_PLOT_PATH,
            "metrics": WIKITEXT2_EVAL_METRICS_PATH,
            "comparison_plot": WIKITEXT2_MODEL_COMPARISON_PLOT_PATH,
        }
    if artifact_set == "demo":
        return {
            "tokenizer": DEMO_TOKENIZER_PATH,
            "rnn_model": DEMO_RNN_MODEL_PATH,
            "lstm_model": DEMO_LSTM_MODEL_PATH,
            "processed_train": DEMO_PROCESSED_TRAIN_PATH,
            "processed_valid": DEMO_PROCESSED_VALID_PATH,
            "processed_test": DEMO_PROCESSED_TEST_PATH,
            "metadata": DEMO_METADATA_PATH,
            "train_history": DEMO_TRAIN_HISTORY_PATH,
            "rnn_history": DEMO_RNN_HISTORY_PATH,
            "lstm_history": DEMO_LSTM_HISTORY_PATH,
            "training_plot": DEMO_TRAINING_PLOT_PATH,
            "metrics": DEMO_EVAL_METRICS_PATH,
            "comparison_plot": DEMO_MODEL_COMPARISON_PLOT_PATH,
        }
    if artifact_set == "demo_v2":
        return {
            "tokenizer": DEMO_V2_TOKENIZER_PATH,
            "rnn_model": DEMO_V2_RNN_MODEL_PATH,
            "lstm_model": DEMO_V2_LSTM_MODEL_PATH,
            "processed_train": DEMO_V2_PROCESSED_TRAIN_PATH,
            "processed_valid": DEMO_V2_PROCESSED_VALID_PATH,
            "processed_test": DEMO_V2_PROCESSED_TEST_PATH,
            "metadata": DEMO_V2_METADATA_PATH,
            "train_history": DEMO_V2_TRAIN_HISTORY_PATH,
            "rnn_history": DEMO_V2_RNN_HISTORY_PATH,
            "lstm_history": DEMO_V2_LSTM_HISTORY_PATH,
            "training_plot": DEMO_V2_TRAINING_PLOT_PATH,
            "metrics": DEMO_V2_EVAL_METRICS_PATH,
            "comparison_plot": DEMO_V2_MODEL_COMPARISON_PLOT_PATH,
            "prediction_audit": DEMO_V2_PREDICTION_AUDIT_PATH,
        }
    if artifact_set == "demo_v3":
        return {
            "tokenizer": DEMO_V3_TOKENIZER_PATH,
            "rnn_model": DEMO_V3_MODELS_DIR / "experiments" / "A" / "best_model.keras",
            "lstm_model": DEMO_V3_MODELS_DIR / "experiments" / "B" / "best_model.keras",
            "experiment_models": {
                "A": DEMO_V3_MODELS_DIR / "experiments" / "A" / "best_model.keras",
                "B": DEMO_V3_MODELS_DIR / "experiments" / "B" / "best_model.keras",
                "C": DEMO_V3_MODELS_DIR / "experiments" / "C" / "best_model.keras",
                "D": DEMO_V3_MODELS_DIR / "experiments" / "D" / "best_model.keras",
            },
            "experiment_histories": {
                experiment_id: DEMO_V3_RESULTS_DIR
                / "experiments"
                / experiment_id
                / "training_history.json"
                for experiment_id in ("A", "B", "C", "D")
            },
            "processed_train": DEMO_V3_PROCESSED_TRAIN_PATH,
            "processed_valid": DEMO_V3_PROCESSED_VALID_PATH,
            "processed_test": DEMO_V3_PROCESSED_TEST_PATH,
            "metadata": DEMO_V3_METADATA_PATH,
            "train_history": DEMO_V3_RESULTS_DIR / "training_history.json",
            "rnn_history": DEMO_V3_RESULTS_DIR / "rnn_training_history.json",
            "lstm_history": DEMO_V3_RESULTS_DIR / "lstm_training_history.json",
            "training_plot": DEMO_V3_RESULTS_DIR / "training_history.png",
            "metrics": DEMO_V3_RESULTS_DIR / "metrics.json",
            "comparison_plot": DEMO_V3_RESULTS_DIR / "model_comparison.png",
        }
    raise ValueError("artifact_set must be 'synthetic', 'wikitext2', 'demo', 'demo_v2', or 'demo_v3'")
