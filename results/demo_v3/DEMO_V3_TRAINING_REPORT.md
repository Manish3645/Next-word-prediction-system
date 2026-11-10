# Demo V3 Controlled Training Report

All four models were trained from scratch using the same saved Demo V3 tokenizer, train/validation arrays, and untouched test array. No V2 artifacts or Demo V3 preprocessing artifacts were overwritten. The Streamlit UI was not modified.

## Experiment comparison

| ID | Model | Masking | Epochs | Best epoch | Train time (s) | Train loss / acc at best | Val loss / acc at best | Test loss | Top-1 | Top-5 | Perplexity | IV loss | IV Top-1 | IV Top-5 | IV perplexity | Eval (s) | Size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | SimpleRNN | False | 4 | 2 | 60.21 | 6.5742 / 0.1046 | 6.9873 / 0.1233 | 6.9816 | 0.1133 | 0.2472 | 1076.665 | 6.0668 | 0.1323 | 0.2886 | 431.303 | 16.82 | 21.36 |
| B | LSTM | False | 3 | 1 | 47.59 | 7.2551 / 0.0638 | 7.1226 / 0.0959 | 7.0809 | 0.0898 | 0.2066 | 1189.088 | 6.3071 | 0.1048 | 0.2412 | 548.449 | 17.39 | 21.78 |
| C | SimpleRNN | True | 4 | 2 | 58.47 | 6.7385 / 0.0916 | 7.1442 / 0.1149 | 7.1171 | 0.1107 | 0.2465 | 1232.907 | 6.1414 | 0.1292 | 0.2878 | 464.722 | 16.47 | 21.36 |
| D | LSTM | True | 3 | 1 | 49.12 | 7.3459 / 0.0664 | 7.1509 / 0.0956 | 7.1042 | 0.0900 | 0.2148 | 1217.020 | 6.3420 | 0.1051 | 0.2508 | 567.914 | 16.78 | 21.78 |

## Findings

- **Best overall** (lowest overall test loss): Experiment A, SimpleRNN with mask_zero=False.
- **Best masked by overall test loss:** Experiment D, LSTM.
- **Best unmasked by overall test loss:** Experiment A, SimpleRNN.
- Masking effect by architecture, measured as masked minus unmasked overall test loss: SimpleRNN +0.1355; LSTM +0.0232. Negative means masking reduced loss.
- Accuracy changes from masking (masked minus unmasked): SimpleRNN top-1 -0.0026, top-5 -0.0007; LSTM top-1 +0.0002, top-5 +0.0082.
- LSTM minus SimpleRNN overall test loss: unmasked +0.0993; masked -0.0130. Negative means LSTM had lower loss.
- Accuracy/time trade-off: unmasked SimpleRNN (A) achieved the best top-1/top-5 and lowest test loss, using 60.21s versus 47.59s for unmasked LSTM (B). Among masked models, SimpleRNN (C) had higher top-1/top-5, while LSTM (D) had slightly lower test loss; their times were 58.47s and 49.12s.

## Dataset and integrity

- Exact test sample count: **10,000**.
- Test OOV targets: **1,434** (14.34%).
- In-vocabulary-only test subset: **8,566** targets; the same 1,434 OOV-target rows are excluded for every model.
- Train arrays SHA-256: `96996f14916b05474b7b8f2aaca63b1cf6034fb68d50c19ed1f1f52e013f0525`.
- Validation arrays SHA-256: `e3787368f1b965b97abf9d0efa87cb8a5b0b875e4f78e3574c1382c562d56624`.
- Test arrays SHA-256 (identical for all experiments): `2bc8690ee754a4756e6d0f7631388c76324601611e0f1e06816d52fe637254f9`.
- Tokenizer mapping SHA-256: `f2b73d7ae243444d65804fcdc4b3ef7de507a3f81c44a5f674f5d1f209ab3790`.
- All model output dimensions match the tokenizer. Probability outputs were checked for finiteness, [0, 1] range, and row sums near 1.
- In-vocabulary-only metrics exclude test rows whose target ID is the OOV ID; overall metrics include all test rows.

## Artifacts

- Aggregate metrics: `results/demo_v3/metrics.json`.
- Per-experiment checkpoints: `models/demo_v3/experiments/{A,B,C,D}/best_model.keras`.
- Per-experiment histories and configurations: `results/demo_v3/experiments/{A,B,C,D}/`.
- Charts: `results/demo_v3/model_comparison.png`, `results/demo_v3/training_duration_comparison.png`, and `results/demo_v3/training_curves.png`.
- Per-experiment metrics: `results/demo_v3/experiments/{A,B,C,D}/metrics.json`.

## Limitations

- Each condition was trained with one seed; results do not quantify run-to-run variance.
- OOV targets are scored as the tokenizer's single OOV class in overall metrics. In-vocabulary metrics are reported separately but do not represent all real-world language coverage.
- This report compares the requested architectures and padding-mask setting only; it is not evidence that one configuration is universally superior.

## Test suite

Full suite: 60 passed, 13 Matplotlib/PyParsing deprecation warnings.
