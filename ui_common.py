import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import predict
from config import get_artifact_paths
ARTIFACT_SET = "demo_v3"
MODEL_LABELS = {"rnn": "SimpleRNN", "lstm": "LSTM"}
V3_MODEL_LABELS = {
    "A": "SimpleRNN · unmasked",
    "B": "LSTM · unmasked",
    "C": "SimpleRNN · masked",
    "D": "LSTM · masked",
}
V3_MODEL_COLORS = {"A": "#67e8f9", "B": "#a78bfa", "C": "#22c55e", "D": "#fb923c"}
METRIC_DIRECTIONS = {
    "test_loss": "lower",
    "top1_accuracy": "higher",
    "top5_accuracy": "higher",
    "perplexity": "lower",
    "training_time_seconds": "lower",
    "evaluation_time_seconds": "lower",
    "model_size_bytes": "lower",
}

THEME_CSS = """
<style>
    :root {
        color-scheme: dark;
        --panel: rgba(15, 23, 42, .78);
        --panel-border: rgba(148, 163, 184, .16);
        --muted: #9fb0c7;
        --accent: #67e8f9;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(14, 116, 144, .20), transparent 38%),
            radial-gradient(ellipse at 92% 10%, rgba(79, 70, 229, .16), transparent 36%),
            linear-gradient(145deg, #070b16 0%, #0b1220 48%, #080d18 100%);
        color: #e5edf8;
    }
    [data-testid="stSidebar"] {
        background: rgba(7, 12, 24, .96);
        border-right: 1px solid var(--panel-border);
    }
    h1, h2, h3 { letter-spacing: -.025em; }
    h1 { font-weight: 720; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 18px 20px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--panel-border);
        border-radius: 18px;
    }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(103, 232, 249, .30);
        transition: border-color .15s ease, transform .15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }
    .eyebrow {
        color: var(--accent);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .muted { color: var(--muted); }
</style>
"""


@st.cache_resource(show_spinner="Loading Demo V3 tokenizer…")
def load_demo_v3_tokenizer():
    return predict.load_tokenizer(ARTIFACT_SET)


@st.cache_resource(show_spinner="Loading Demo V3 model…")
def load_demo_v3_model(model_name):
    if model_name not in V3_MODEL_LABELS:
        raise ValueError("Choose one of the four Demo V3 experiments.")
    model = predict.load_model(model_name, ARTIFACT_SET)
    predict.validate_demo_v3_model_tokenizer_compatibility(
        model, load_demo_v3_tokenizer(), model_name
    )
    return model


@st.cache_resource(show_spinner="Loading Demo V2 tokenizer…")
def load_demo_v2_tokenizer():
    return predict.load_tokenizer("demo_v2")


@st.cache_resource(show_spinner="Loading Demo V2 model…")
def load_demo_v2_model(model_name):
    if model_name not in MODEL_LABELS:
        raise ValueError("Choose either SimpleRNN or LSTM.")
    model = predict.load_model(model_name, "demo_v2")
    predict.validate_model_tokenizer_compatibility(model, load_demo_v2_tokenizer())
    return model


@st.cache_data(show_spinner=False)
def _read_json(path_string, modified_ns):
    del modified_ns
    with Path(path_string).open("r", encoding="utf-8") as file:
        return json.load(file)


def read_json_artifact(path):
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime_ns)


def get_demo_paths():
    return get_artifact_paths(ARTIFACT_SET)


def missing_runtime_assets(paths):
    if "experiment_models" in paths:
        required = [
            ("tokenizer", "Demo V3 tokenizer"),
            ("metadata", "Demo V3 preprocessing metadata"),
        ]
        required.extend(
            (f"experiment:{experiment_id}", f"Experiment {experiment_id} checkpoint")
            for experiment_id in V3_MODEL_LABELS
        )
        return [
            (
                label,
                paths["tokenizer"]
                if key == "tokenizer"
                else paths["metadata"]
                if key == "metadata"
                else paths["experiment_models"][key.split(":")[1]],
            )
            for key, label in required
            if not (
                paths["tokenizer"].is_file()
                if key == "tokenizer"
                else paths["metadata"].is_file()
                if key == "metadata"
                else paths["experiment_models"][key.split(":")[1]].is_file()
            )
        ]
    return [
        (label, paths[key])
        for key, label in (
            ("tokenizer", "Demo V2 tokenizer"),
            ("rnn_model", "SimpleRNN checkpoint"),
            ("lstm_model", "LSTM checkpoint"),
        )
        if not paths[key].is_file()
    ]


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### ✦ WikiText-2 Lab")
    st.sidebar.caption("Demo V3 · four model experiments")
    st.sidebar.markdown("---")


def show_asset_status(artifact_set=ARTIFACT_SET):
    paths = get_artifact_paths(artifact_set)
    missing = missing_runtime_assets(paths)
    if missing:
        display_name = "Demo V3" if artifact_set == "demo_v3" else "Demo V2"
        st.error(f"Some {display_name} runtime files are missing. Add the trained files listed below.")
        for label, path in missing:
            st.code(f"{label}: {path}", language=None)
        return False
    return True


def show_missing_artifact(path, label):
    st.info(f"{label} is not available yet.")
    st.code(str(path), language=None)


def format_metric(metric_name, value):
    if metric_name in {"top1_accuracy", "top5_accuracy"}:
        return f"{value * 100:.2f}%"
    if metric_name == "model_size_bytes":
        return f"{value / (1024 * 1024):.2f} MB"
    if metric_name in {"training_time_seconds", "evaluation_time_seconds"}:
        return f"{value:.2f} s"
    return f"{value:.4f}"


def better_model(metrics, metric_name):
    direction = METRIC_DIRECTIONS[metric_name]
    labels = V3_MODEL_LABELS if any(key in metrics for key in V3_MODEL_LABELS) else MODEL_LABELS
    available = []
    for name, values in metrics.items():
        if name not in labels:
            continue
        metric_value = values.get(metric_name)
        if metric_value is None:
            metric_value = values.get("overall_test", {}).get(metric_name)
        if metric_value is not None:
            available.append((name, metric_value))
    if len(available) < 2:
        return None
    key = (lambda pair: pair[1]) if direction == "lower" else (lambda pair: -pair[1])
    winner_value = min(available, key=key)[1]
    winners = [name for name, value in available if value == winner_value]
    return "Tie" if len(winners) > 1 else labels[winners[0]]


def append_word(text, word):
    if word in {".", ",", "!", "?", ";", ":"}:
        return text.rstrip() + word
    return f"{text.rstrip()} {word}".strip()


def make_history_figure(history, metric, validation_metric, title, ylabel):
    fig, axis = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#0b1220")
    axis.set_facecolor("#0f172a")
    model_ids = tuple(V3_MODEL_LABELS) if any(key in history for key in V3_MODEL_LABELS) else ("rnn", "lstm")
    labels = V3_MODEL_LABELS if model_ids and model_ids[0] in V3_MODEL_LABELS else MODEL_LABELS
    colors = V3_MODEL_COLORS if model_ids and model_ids[0] in V3_MODEL_LABELS else {
        "rnn": "#67e8f9",
        "lstm": "#a78bfa",
    }
    for model_name in model_ids:
        values = history.get(model_name, {})
        train_values = values.get(metric, [])
        valid_values = values.get(validation_metric, [])
        if train_values:
            axis.plot(
                range(1, len(train_values) + 1),
                train_values,
                color=colors[model_name],
                linewidth=2,
                label=f"{labels[model_name]} train",
            )
        if valid_values:
            axis.plot(
                range(1, len(valid_values) + 1),
                valid_values,
                color=colors[model_name],
                linewidth=2,
                linestyle="--",
                label=f"{labels[model_name]} validation",
            )
    axis.set_title(title, color="#e5edf8", pad=12)
    axis.set_xlabel("Epoch", color="#b6c4d8")
    axis.set_ylabel(ylabel, color="#b6c4d8")
    axis.tick_params(colors="#b6c4d8")
    axis.grid(color="#334155", alpha=.45)
    for spine in axis.spines.values():
        spine.set_color("#334155")
    axis.legend(frameon=False, labelcolor="#dbeafe")
    fig.tight_layout()
    return fig


def make_duration_figure(metrics):
    fig, axis = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#0b1220")
    axis.set_facecolor("#0f172a")
    experiment_metrics = metrics.get("experiments", metrics)
    if any(name in experiment_metrics for name in V3_MODEL_LABELS):
        names = [name for name in V3_MODEL_LABELS if name in experiment_metrics]
        labels = [V3_MODEL_LABELS[name] for name in names]
        values = [experiment_metrics[name].get("training_time_seconds", 0.0) for name in names]
        colors = [V3_MODEL_COLORS[name] for name in names]
    else:
        names = [name for name in ("rnn", "lstm") if name in experiment_metrics]
        labels = [MODEL_LABELS[name] for name in names]
        values = [experiment_metrics[name].get("training_time_seconds", 0.0) for name in names]
        colors = ["#67e8f9" if name == "rnn" else "#a78bfa" for name in names]
    axis.bar(labels, values, color=colors, width=.52)
    axis.set_title("Training duration", color="#e5edf8", pad=12)
    axis.set_ylabel("Seconds", color="#b6c4d8")
    axis.tick_params(colors="#b6c4d8")
    axis.grid(axis="y", color="#334155", alpha=.45)
    for spine in axis.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig
