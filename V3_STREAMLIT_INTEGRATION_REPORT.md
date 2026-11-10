# Demo V3 Streamlit Integration Report

## Summary

Integrated the four completed Demo V3 experiments into the existing Streamlit pages without training models or changing any model, tokenizer, dataset, metric, or preprocessing artifact. Demo V2 checkpoints remain available through the artifact selector on the prediction and continuation pages. The comparison and performance dashboard show saved Demo V3 results.

The existing `app.py` navigation already registered all five pages, so it required no change.

## Features

- Added direct checkpoint routing for experiments A-D:
  - A: SimpleRNN, `mask_zero=False`
  - B: LSTM, `mask_zero=False`
  - C: SimpleRNN, `mask_zero=True`
  - D: LSTM, `mask_zero=True`
- Cached the V3 tokenizer and each model. Before inference, the app verifies the vocabulary size, output dimension, tokenizer mapping hash, and expected masking configuration.
- Routed V3 input and autoregressive generation through the canonical V3 Unicode tokenization pipeline. Demo V2 inference remains routed through its legacy tokenizer and model artifacts.
- Preserved top-five next-word predictions with actual softmax probabilities and clickable suggestions.
- Preserved sentence continuation with Greedy, Top-K, and Temperature sampling, a 1-30 token count, temperature, Top-K, and repetition-penalty controls.
- Added V3 comparison tables using the saved overall and in-vocabulary test results, including loss, top-1/top-5 accuracy, perplexity, training time, model size, masking, and best epoch.
- Added V3 training-loss/accuracy curves, comparison and duration charts, and a test-performance summary.
- Updated About text and project documentation. Generated continuation text is HTML-escaped before rendering.
- Missing tokenizer/checkpoints are reported with their expected paths; the app does not download data or train at startup.

## Artifact verification

The app loads these actual checkpoints:

- `models/demo_v3/experiments/A/best_model.keras`
- `models/demo_v3/experiments/B/best_model.keras`
- `models/demo_v3/experiments/C/best_model.keras`
- `models/demo_v3/experiments/D/best_model.keras`

The V3 tokenizer is `models/demo_v3/tokenizer.pkl`; its vocabulary/output dimension is 9,593 and its ordered word-index SHA-256 is `f2b73d7ae243444d65804fcdc4b3ef7de507a3f81c44a5f674f5d1f209ab3790`. Each checkpoint's output dimension and mask configuration were verified against that tokenizer and experiment ID. The integration tests ran real top-five inference on all four checkpoints and exercised all three decoding strategies with all four models.

Saved comparison data is read from `results/demo_v3/metrics.json`; history curves are read from each experiment's `training_history.json`. No metric values are generated or substituted in the UI.

## Files changed

- `config.py` — explicit V3 experiment checkpoint and history paths.
- `predict.py` — V3 checkpoint selection, compatibility validation, and V3 tokenization for both prediction and continuation.
- `ui_common.py` — cached V2/V3 loaders, artifact validation, labels, and metric/chart helpers.
- `ui_pages/1_Next_Word_Prediction.py` — Demo V2/V3 selector and four V3 model choices with suggestions.
- `ui_pages/2_Sentence_Continuation.py` — Demo V2/V3 selector, four V3 model choices, and decoding controls.
- `ui_pages/3_Model_Comparison.py` — four-experiment V3 comparison and saved overall/in-vocabulary results.
- `ui_pages/4_Performance_Dashboard.py` — V3 training curves, duration chart, metrics, and winner explanation.
- `ui_pages/5_About.py` — V3 dataset, model, preprocessing, decoding, and evaluation explanations.
- `tests/test_streamlit_app.py` — V3 metrics, table values, missing-artifact handling, and V2 selector coverage.
- `tests/test_demo_v3_integration.py` — actual checkpoint, top-five, tokenizer, and all-model/strategy inference coverage.
- `README.md` — setup, V3 training reproduction, measured comparison, runtime artifacts, and deployment instructions.
- `V3_STREAMLIT_INTEGRATION_REPORT.md` — this report.

No changes were made to `app.py`, V2/V3 training or preprocessing code, model checkpoints, tokenizer, data arrays, or saved metrics.

## Validation

- Complete test suite: **81 passed**, with **13 pre-existing Matplotlib/PyParsing deprecation warnings**.
- Real checkpoint inference: all four models returned five unique vocabulary words with valid probabilities.
- Sentence continuation: all four models passed Greedy, Top-K, and Temperature decoding with one requested generated token; selected token IDs were valid, non-padding, non-OOV vocabulary IDs.
- Suggestion UI: AppTest verified clicking a suggestion appends it and requests another prediction.
- Comparison UI: displayed test-loss and top-1 values matched `metrics.json`.
- Streamlit startup: launched on port 8502 and verified `/_stcore/health` returned **HTTP 200, `ok`**. Port 8501 was already occupied, so no process on that port was disturbed. The verification server was stopped afterward.

## Remaining limitations

- Demo V3 results are from a single training run and the selected WikiText-2 subset; they do not establish general model superiority.
- Test targets include 1,434 OOV cases (14.34%); the comparison page also presents in-vocabulary-only scores.
- Demo V2 is available for next-word prediction and sentence continuation. Comparison and dashboard pages are intentionally V3-specific.
- Deployment must include all four V3 checkpoints, the matching V3 tokenizer and metadata, saved metrics, charts, and experiment histories. Large checkpoints may require Git LFS or another artifact-delivery approach.
