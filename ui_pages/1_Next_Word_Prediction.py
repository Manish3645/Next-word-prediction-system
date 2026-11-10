import streamlit as st

from predict import predict_next_words
from config import get_artifact_paths
from ui_common import (
    MODEL_LABELS,
    V3_MODEL_LABELS,
    append_word,
    apply_theme,
    load_demo_v2_model,
    load_demo_v2_tokenizer,
    load_demo_v3_model,
    load_demo_v3_tokenizer,
    show_asset_status,
)


def select_suggestion(word):
    current_text = st.session_state.get("prediction_input", "")
    st.session_state["prediction_input"] = append_word(current_text, word)
    st.session_state["predict_after_suggestion"] = True


def main():
    apply_theme()
    st.markdown('<p class="eyebrow">Demo V3 and V2 · Language modeling</p>', unsafe_allow_html=True)
    st.title("Next Word Prediction")
    st.caption("Choose Demo V3's four experiments or use the preserved Demo V2 RNN/LSTM checkpoints.")
    artifact_labels = {"Demo V3": "demo_v3", "Demo V2": "demo_v2"}
    artifact_label = st.selectbox(
        "Artifact set",
        list(artifact_labels),
        key="prediction_artifact_set",
    )
    artifact_set = artifact_labels[artifact_label]
    paths = get_artifact_paths(artifact_set)
    assets_ready = show_asset_status(artifact_set)
    model_labels = V3_MODEL_LABELS if artifact_set == "demo_v3" else MODEL_LABELS
    with st.form("next_word_form"):
        prompt = st.text_area(
            "Your text",
            key="prediction_input",
            height=130,
            placeholder="Type a few words to get started…",
            help="Demo V3 uses the shared Unicode tokenizer; Demo V2 retains its original preprocessing.",
        )
        selected_label = st.selectbox("Model", list(model_labels.values()), key="prediction_model")
        submitted = st.form_submit_button("Predict next words", type="primary", disabled=not assets_ready)

    predict_after_suggestion = st.session_state.pop("predict_after_suggestion", False)
    if submitted or predict_after_suggestion:
        if not prompt.strip():
            st.warning("Enter some text before requesting a prediction.")
            st.session_state.pop("prediction_results", None)
        else:
            try:
                model_name = next(key for key, label in model_labels.items() if label == selected_label)
                if artifact_set == "demo_v3":
                    load_demo_v3_tokenizer()
                    load_demo_v3_model(model_name)
                else:
                    load_demo_v2_tokenizer()
                    load_demo_v2_model(model_name)
                st.session_state["prediction_results"] = predict_next_words(
                    prompt,
                    model_name=model_name,
                    top_k=5,
                    artifact_set=artifact_set,
                )
                st.session_state["prediction_results_model"] = model_name
                st.session_state["prediction_results_artifact_set"] = artifact_set
            except (FileNotFoundError, OSError, ValueError) as exc:
                st.error(f"Prediction could not be generated: {exc}")

    results = st.session_state.get("prediction_results")
    if results:
        result_model = st.session_state.get("prediction_results_model", "A")
        result_artifact = st.session_state.get("prediction_results_artifact_set", "demo_v3")
        result_labels = V3_MODEL_LABELS if result_artifact == "demo_v3" else MODEL_LABELS
        st.subheader(f"Suggestions · {result_labels[result_model]}")
        st.caption("Probabilities are the selected model's original softmax values; bars are not rescaled.")
        for item in results:
            col_word, col_probability, col_bar = st.columns([2.4, 1, 3])
            with col_word:
                st.button(
                    f"＋ {item['word']}",
                    key=f"suggestion-{result_artifact}-{result_model}-{item['rank']}-{item['word']}",
                    use_container_width=True,
                    on_click=select_suggestion,
                    args=(item["word"],),
                )
            with col_probability:
                st.write(f"{item['probability'] * 100:.2f}%")
            with col_bar:
                st.progress(min(max(item["probability"], 0.0), 1.0))
        st.caption("Select a word to append it and automatically request the next suggestion.")

    if not assets_ready:
        st.info("Provide the checkpoints and tokenizer listed above to enable inference.")
        st.code(f"Tokenizer: {paths['tokenizer']}", language=None)


main()
