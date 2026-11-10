import streamlit as st

from ui_common import apply_theme


def main():
    apply_theme()
    st.markdown('<p class="eyebrow">Project overview</p>', unsafe_allow_html=True)
    st.title("About the Project")
    st.caption("A controlled comparison of SimpleRNN and LSTM next-word models, with and without padding masks.")

    st.subheader("Dataset and preprocessing")
    st.write(
        "Demo V3 uses the official `Salesforce/wikitext` dataset configuration "
        "`wikitext-2-raw-v1`. It preserves the official train, validation, and test splits, "
        "uses deterministic sequence selection, and fits one tokenizer on selected training examples only."
    )
    st.write(
        "Text is normalized with Unicode NFC, consistent whitespace, and normalized curly quotation marks. "
        "Training, validation, test, and inference use the same Unicode-aware tokenizer. Sentence and paragraph "
        "boundaries are retained so generated training sequences do not cross unrelated text boundaries. "
        "The V3 tokenizer reserves token ID 0 for padding and ID 1 for OOV."
    )

    st.subheader("The four model experiments")
    st.write(
        "Every experiment uses the same V3 tokenizer, vocabulary, training/validation arrays, and held-out test "
        "arrays. The configurations are A: SimpleRNN without masking, B: LSTM without masking, "
        "C: SimpleRNN with `mask_zero=True`, and D: LSTM with `mask_zero=True`. Each model has a 128-dimensional "
        "embedding, 64 recurrent units, 0.2 dropout, and a vocabulary-sized softmax output."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("SimpleRNN")
        st.write(
            "A Simple Recurrent Neural Network processes tokens sequentially and carries a hidden state forward. "
            "It is a compact baseline, although long-range information can be difficult to retain."
        )
    with right:
        st.subheader("LSTM")
        st.write(
            "A Long Short-Term Memory network adds gated memory cells to control which information is retained "
            "or discarded over time."
        )

    st.subheader("Next-word prediction and continuation")
    st.write(
        "For next-word prediction, the app normalizes and tokenizes the input with the V3 pipeline, pads or crops "
        "it to the model context length, then displays the five highest-probability vocabulary tokens and their "
        "actual model probabilities. Selecting a suggestion appends it to the prompt and runs another prediction."
    )
    st.write(
        "Sentence continuation is autoregressive: each generated token is fed back into the same selected model. "
        "Greedy decoding chooses the highest adjusted probability; Top-K sampling samples among the highest-ranked "
        "candidates; temperature sampling adjusts the full distribution. The repetition penalty down-weights recent "
        "tokens and short repeated phrases but does not prohibit legitimate repetition."
    )

    st.subheader("Evaluation")
    st.write(
        "Saved evaluation results include test loss, top-1 and top-5 accuracy, perplexity, training and evaluation "
        "time, and checkpoint size. Overall results include OOV targets; in-vocabulary-only metrics exclude those "
        "targets. All four experiments were evaluated on the same untouched test arrays and their output "
        "probabilities were validated as finite distributions."
    )
    st.info(
        "In the completed Demo V3 experiment, A (unmasked SimpleRNN) had the best overall test loss "
        "(6.9816), top-1 accuracy (11.33%), and top-5 accuracy (24.72%). This is a result for this specific "
        "subset and training run, not a general claim that SimpleRNN outperforms LSTM."
    )

    st.subheader("Runtime behavior and limitations")
    st.write(
        "The prediction and continuation pages can use either Demo V3's four experiments or the preserved "
        "Demo V2 RNN/LSTM artifacts. Comparison and dashboard pages show the completed Demo V3 experiment. "
        "The application only loads saved artifacts; it does not train models or download the dataset at startup. "
        "The model vocabulary is limited, so unseen words map to OOV. Recurrent models may still produce locally "
        "repetitive or ungrammatical continuations."
    )
    st.write("**Stack:** Python, TensorFlow/Keras, NumPy, Matplotlib, and Streamlit.")


main()
