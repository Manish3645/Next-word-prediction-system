import streamlit as st

from ui_common import (
    V3_MODEL_LABELS,
    apply_theme,
    better_model,
    format_metric,
    get_demo_paths,
    read_json_artifact,
    show_missing_artifact,
)


DISPLAY_METRICS = (
    ("test_loss", "Test loss"),
    ("top1_accuracy", "Top-1 accuracy"),
    ("top5_accuracy", "Top-5 accuracy"),
    ("perplexity", "Perplexity"),
    ("training_time_seconds", "Training time"),
    ("model_size_bytes", "Model size"),
)


def main():
    apply_theme()
    st.markdown('<p class="eyebrow">Demo V3 · Test-set results</p>', unsafe_allow_html=True)
    st.title("Model Comparison")
    st.caption("All values are read from the saved Demo V3 evaluation results; no metrics are recomputed in the UI.")

    paths = get_demo_paths()
    metrics = read_json_artifact(paths["metrics"])
    if metrics is None:
        show_missing_artifact(paths["metrics"], "Demo V3 metrics")
        return
    experiments = metrics.get("experiments", {})
    if not all(experiment_id in experiments for experiment_id in V3_MODEL_LABELS):
        st.error("The saved Demo V3 metrics do not contain all four experiments.")
        return

    columns = st.columns(4)
    for column, (metric_name, title) in zip(
        columns,
        (
            ("test_loss", "Best test loss"),
            ("top1_accuracy", "Best top-1"),
            ("top5_accuracy", "Best top-5"),
            ("training_time_seconds", "Fastest training"),
        ),
    ):
        winner = better_model(experiments, metric_name)
        column.metric(title, winner or "—")

    rows = []
    for experiment_id, label in V3_MODEL_LABELS.items():
        values = experiments[experiment_id]
        overall = values.get("overall_test", {})
        rows.append(
            {
                "Experiment": experiment_id,
                "Model": label,
                "Masking": "Enabled" if values.get("mask_zero") else "Disabled",
                "Test loss": format_metric("test_loss", overall["test_loss"]),
                "Top-1 accuracy": format_metric("top1_accuracy", overall["top1_accuracy"]),
                "Top-5 accuracy": format_metric("top5_accuracy", overall["top5_accuracy"]),
                "Perplexity": format_metric("perplexity", overall["perplexity"]),
                "Training time": format_metric("training_time_seconds", values["training_time_seconds"]),
                "Model size": format_metric("model_size_bytes", values["model_size_bytes"]),
                "Best epoch": values["best_epoch"],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("In-vocabulary test results")
    st.caption(
        f"The shared test set contains {metrics.get('oov_target_count', '—')} OOV targets "
        f"out of {metrics.get('test_sample_count', '—')}. The following scores exclude those targets."
    )
    iv_rows = []
    for experiment_id, label in V3_MODEL_LABELS.items():
        values = experiments[experiment_id].get("in_vocabulary_test", {})
        iv_rows.append(
            {
                "Experiment": experiment_id,
                "Model": label,
                "Samples": values.get("sample_count", "—"),
                "Test loss": format_metric("test_loss", values["test_loss"]) if "test_loss" in values else "—",
                "Top-1 accuracy": format_metric("top1_accuracy", values["top1_accuracy"]) if "top1_accuracy" in values else "—",
                "Top-5 accuracy": format_metric("top5_accuracy", values["top5_accuracy"]) if "top5_accuracy" in values else "—",
                "Perplexity": format_metric("perplexity", values["perplexity"]) if "perplexity" in values else "—",
            }
        )
    st.dataframe(iv_rows, use_container_width=True, hide_index=True)
    st.caption("The experiments share the same tokenizer, vocabulary, train/validation data, and untouched test arrays.")

    chart = paths["comparison_plot"]
    if chart.is_file():
        st.subheader("Saved comparison chart")
        st.image(str(chart), use_column_width=True)
    else:
        st.info(f"Comparison chart is missing: {chart}")


main()
