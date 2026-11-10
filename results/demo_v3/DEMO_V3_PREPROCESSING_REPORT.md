# Demo V3 Preprocessing Report

## Changes implemented

- Added an isolated Demo V3 preprocessing module using one Unicode-aware normalization, tokenization, and encoding path for every split and inference input.
- NFC-normalized text, normalized curly quotes/apostrophes, preserved punctuation tokens, retained internal apostrophes, split hyphenated forms, and mapped WikiText `@-@`, `@,@`, and `@.@` markers.
- Fit the tokenizer only on the final selected training examples; reserved ID 0 for padding and ID 1 for OOV; recorded a SHA-256 hash of ordered word-to-ID entries.
- Added a `mask_zero=True` model-builder option for both recurrent architectures. No models were trained.
- Preserved the official WikiText-2 train/validation/test sources and wrote all generated artifacts under Demo V3 paths.

## Files created and modified

- Modified `config.py` with isolated Demo V3 paths and settings.
- Modified `train.py` to accept the optional `mask_zero` setting on both model builders; defaults preserve existing behavior.
- Modified `predict.py` with an opt-in Demo V3 inference tokenizer; existing application tokenization remains unchanged.
- Added `demo_v3_preprocess.py` and `tests/test_demo_v3_preprocessing.py`.
- Created `data/demo_v3/`, `models/demo_v3/`, and `results/demo_v3/` artifacts, including this report.

## Dataset statistics

| Split | Selected records | Sequences | OOV targets | OOV targets % | OOV context tokens | Non-padding context tokens | OOV context % |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 578 | 50000 | 0 | 0.000% | 0 | 406958 | 0.000% |
| validation | 105 | 10000 | 1451 | 14.510% | 12197 | 82386 | 14.805% |
| test | 109 | 10000 | 1434 | 14.340% | 11692 | 80749 | 14.479% |

- Effective vocabulary size: **9,593** (including padding and OOV IDs).
- Ordered word-index mapping SHA-256: `f2b73d7ae243444d65804fcdc4b3ef7de507a3f81c44a5f674f5d1f209ab3790`.
- Train sequences: **50,000**, versus Demo V2 **20,000**.

## OOV comparison against Demo V2

| Split | Demo V2 OOV targets | Demo V2 OOV % | Demo V3 OOV targets | Demo V3 OOV % |
|---|---:|---:|---:|---:|
| train | 54 | 0.270% | 0 | 0.000% |
| validation | 2285 | 22.850% | 1451 | 14.510% |
| test | 2289 | 22.890% | 1434 | 14.340% |

## Integrity checks

- Official split separation: **True**.
- Selected record overlap: **False**.
- Duplicate selected normalized records: **0**.
- Sequences crossing sentence boundaries: **0**.
- Sequences crossing record boundaries: **0**.
- Training OOV targets: **0** (generation stops with an error if this invariant fails).

## Test results

Final full suite: 57 passed, 13 Matplotlib/PyParsing deprecation warnings.

## Artifacts

- Tokenizer: `./models/demo_v3/tokenizer.pkl`
- Preprocessing metadata: `./models/demo_v3/preprocessing_metadata.json`
- Preprocessing configuration: `./models/demo_v3/preprocessing_config.json`
- Dataset statistics: `./results/demo_v3/dataset_statistics.json`
- Processed data: `./data/demo_v3/processed_train.npz`, `./data/demo_v3/processed_validation.npz`, `./data/demo_v3/processed_test.npz`

## Remaining limitations

- The 50,000 training-sequence budget is larger than Demo V2 but remains a subset of WikiText-2; rare held-out terms can still map to OOV.
- Sentence splitting is punctuation-based and can split after abbreviations; a linguistic sentence segmenter is not used.
- Punctuation and WikiText markers are vocabulary tokens, so Demo V3 requires models trained with this exact tokenizer and cannot use V2 checkpoints.
- `mask_zero=True` is configured as an opt-in experiment, not yet validated by training or evaluation.
- No Demo V3 models were trained and no V2 files or metrics were changed.
