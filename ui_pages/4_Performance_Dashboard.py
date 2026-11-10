import matplotlib.pyplot as plt
import streamlit as st

from ui_common import (
    V3_MODEL_LABELS,
    apply_theme,
    better_model,
    format_metric,
    get_demo_paths,
    make_duration_figure,
    make_history_figure,
    read_json_artifact,
    show_missing_artifact,
)


def main():
    apply_theme()
    st.markdown('<p class="eyebrow">Demo V3 · Training and evaluation</p>', unsafe_allow_html=True)
    st.title("Performance Dashboard")
    st.caption("Training curves, test performance, and training duration from the four saved Demo V3 experiments.")

    paths = get_demo_paths()
    metrics = read_json_artifact(paths["metrics"])
    if metrics is None:
        show_missing_artifact(paths["metrics"], "Demo V3 metrics")
        return
    experiments = metrics.get("experiments", {})
    histories = {
        experiment_id: read_json_artifact(history_path)
        for experiment_id, history_path in paths["experiment_histories"].items()
    }
    missing_histories = [
        experiment_id for experiment_id, history in histories.items() if history is None
    ]
    if missing_histories:
        for experiment_id in missing_histories:
            show_missing_artifact(
                paths["experiment_histories"][experiment_id],
                f"Experiment {experiment_id} training history",
            )
        return

    cards = st.columns(4)
    for column, metric_name in zip(
        cards,
        ("test_loss", "top1_accuracy", "top5_accuracy", "training_time_seconds"),
    ):
        winner = better_model(experiments, metric_name)
        column.metric(f"Best {metric_name.replace('_', ' ')}", winner or "—")

    overall_winner = better_model(experiments, "test_loss")
    if overall_winner:
        winning_id = min(
            experiments,
            key=lambda experiment_id: experiments[experiment_id]["overall_test"]["test_loss"],
        )
        winning_test = experiments[winning_id]["overall_test"]
        st.info(
            f"{overall_winner} is the overall winner by test loss; it also has "
            f"{format_metric('top1_accuracy', winning_test['top1_accuracy'])} "
            "top-1 and "
            f"{format_metric('top5_accuracy', winning_test['top5_accuracy'])} "
            "top-5 accuracy."
        )

    for metric, validation_metric, title, ylabel in (
        ("loss", "val_loss", "Training and validation loss", "Sparse categorical crossentropy"),
        ("accuracy", "val_accuracy", "Training and validation accuracy", "Accuracy"),
    ):
        figure = make_history_figure(
            histories,
            metric,
            validation_metric,
            title,
            ylabel,
        )
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    left, right = st.columns([1.2, 1])
    with left:
        if paths["comparison_plot"].is_file():
            st.subheader("Test-set comparison")
            st.image(str(paths["comparison_plot"]), use_column_width=True)
        else:
            st.info(f"Comparison chart is missing: {paths['comparison_plot']}")
    with right:
        st.subheader("Training duration")
        duration_figure = make_duration_figure(metrics)
        st.pyplot(duration_figure, use_container_width=True)
        plt.close(duration_figure)

    st.subheader("Test metrics")
    rows = []
    for experiment_id, label in V3_MODEL_LABELS.items():
        values = experiments.get(experiment_id, {})
        overall = values.get("overall_test", {})
        rows.append(
            {
                "Experiment": experiment_id,
                "Model": label,
                "Test loss": format_metric("test_loss", overall["test_loss"]) if "test_loss" in overall else "—",
                "Top-1": format_metric("top1_accuracy", overall["top1_accuracy"]) if "top1_accuracy" in overall else "—",
                "Top-5": format_metric("top5_accuracy", overall["top5_accuracy"]) if "top5_accuracy" in overall else "—",
                "Perplexity": format_metric("perplexity", overall["perplexity"]) if "perplexity" in overall else "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


main()
