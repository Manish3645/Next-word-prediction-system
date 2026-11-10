from html import escape

import streamlit as st

from config import get_artifact_paths
from predict import continue_sentence
from ui_common import (
    MODEL_LABELS,
    V3_MODEL_LABELS,
    apply_theme,
    load_demo_v2_model,
    load_demo_v2_tokenizer,
    load_demo_v3_model,
    load_demo_v3_tokenizer,
    show_asset_status,
)

DECODING_STRATEGIES = {
    "Greedy Decoding": "greedy",
    "Top-K Sampling": "top_k",
    "Temperature-based Sampling": "temperature",
}


def main():
    apply_theme()
    st.markdown('<p class="eyebrow">Demo V3 and V2 · Autoregressive generation</p>', unsafe_allow_html=True)
    st.title("Sentence Continuation")
    st.caption("The selected model predicts one token at a time; each token is added back to the context.")

    artifact_labels = {"Demo V3": "demo_v3", "Demo V2": "demo_v2"}
    artifact_label = st.selectbox("Artifact set", list(artifact_labels), key="continuation_artifact_set")
    artifact_set = artifact_labels[artifact_label]
    paths = get_artifact_paths(artifact_set)
    assets_ready = show_asset_status(artifact_set)
    model_labels = V3_MODEL_LABELS if artifact_set == "demo_v3" else MODEL_LABELS
    with st.form("continuation_form"):
        prompt = st.text_area(
            "Incomplete sentence",
            height=130,
            placeholder="For example: The history of",
        )
        selected_label = st.selectbox("Model", list(model_labels.values()))
        word_count = st.slider("Words to generate", min_value=1, max_value=30, value=10)
        strategy_label = st.selectbox("Decoding strategy", list(DECODING_STRATEGIES))
        temperature = st.slider(
            "Temperature",
            min_value=0.2,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Lower values favor more likely tokens; higher values increase sampling diversity.",
        )
        top_k = st.slider(
            "Top-K candidates",
            min_value=3,
            max_value=20,
            value=10,
            help="Used by Top-K Sampling; sampling is restricted to this many highest-probability tokens.",
        )
        repetition_penalty = st.slider(
            "Repetition penalty",
            min_value=1.0,
            max_value=2.5,
            value=1.2,
            step=0.1,
            help="Down-weights recent tokens and repeated short phrases without banning vocabulary words.",
        )
        submitted = st.form_submit_button("Continue sentence", type="primary", disabled=not assets_ready)

    if submitted:
        if not prompt.strip():
            st.warning("Enter an incomplete sentence first.")
        else:
            model_name = next(key for key, label in model_labels.items() if label == selected_label)
            try:
                if artifact_set == "demo_v3":
                    load_demo_v3_tokenizer()
                    load_demo_v3_model(model_name)
                else:
                    load_demo_v2_tokenizer()
                    load_demo_v2_model(model_name)
                completed = continue_sentence(
                    prompt,
                    model_name=model_name,
                    additional_words=word_count,
                    artifact_set=artifact_set,
                    strategy=DECODING_STRATEGIES[strategy_label],
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                st.subheader("Generated continuation")
                st.markdown(
                    f'<div style="padding:1.2rem 1.4rem;border:1px solid rgba(148,163,184,.2);'
                    f'border-radius:16px;background:rgba(15,23,42,.78);font-size:1.15rem;line-height:1.8">'
                    f'{escape(completed)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Strategy: {strategy_label} · Model: {selected_label} · "
                    f"Generated tokens: {word_count}"
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                st.error(f"Sentence continuation could not be generated: {exc}")

    if not assets_ready:
        checkpoint_path = (
            paths["experiment_models"]["A"].parent.parent
            if artifact_set == "demo_v3"
            else paths["rnn_model"].parent
        )
        st.info(f"Checkpoints are stored under: {checkpoint_path}")


main()
