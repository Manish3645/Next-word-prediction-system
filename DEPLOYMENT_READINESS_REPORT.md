# GitHub and Streamlit Community Cloud Readiness Audit

**Audit date:** 2026-11-11  
**Scope:** Demo V3 inference deployment, optional Demo V2 inference, GitHub packaging, and clean-environment dependency requirements.  
**Deployment/push status:** Nothing was pushed or deployed.

## Executive summary

The app's Demo V3 runtime artifacts are present and were loaded and used for inference. All four checkpoints match the V3 tokenizer, have the expected mask configuration and 9,593 output classes, and produce five valid next-word predictions. The Streamlit app starts successfully and its health endpoint responds with HTTP 200.

Deployment blockers found and addressed:

1. The original global `.gitignore` patterns excluded all `.keras` and `.pkl` files, including required inference artifacts. Narrow allow-rules now expose the four V3 checkpoints and V3 tokenizer. Optional V2 runtime files are also allow-listed because the app exposes a Demo V2 selector.
2. `predict.py` imports `preprocess.py`; preprocessing previously imported Hugging Face `datasets` unconditionally. The import is now delayed until dataset preparation is explicitly called, so model inference does not require the package or a local Hugging Face cache.
3. `requirements.txt` included development and data-preparation packages unused by inference. Runtime, training, and test dependencies are now separated.
4. Deployment-included metadata and metrics contained absolute Windows paths. Path-valued metadata was made project-relative without changing measured metric values, and the V3 preprocessing/report generators now write portable relative paths.

**Git status at final pre-push review:** An empty Git repository has now been initialized on branch `main`. There are no commits and nothing is staged; all non-ignored project files are untracked. No commit or push was performed.

## Project structure inspected

- Root Python modules: `app.py`, `config.py`, `predict.py`, `preprocess.py`, `demo_v3_preprocess.py`, `train.py`, `train_demo_v3_experiments.py`, and `evaluate.py`.
- Streamlit shared module: `ui_common.py`.
- Five UI pages: next-word prediction, sentence continuation, model comparison, performance dashboard, and About.
- Seven `tests/test_*.py` modules cover preprocessing, models, prediction, Streamlit pages, and Demo V3 integration/training helpers.
- Deployment/dependency files: `requirements.txt`, `requirements-training.txt`, `requirements-dev.txt`, `.gitignore`, `.streamlit/config.toml`, `README.md`, and `LICENSE`.
- Runtime artifact locations are constructed from `PROJECT_ROOT = Path(__file__).resolve().parent`; inference paths are not tied to the audited Windows path.
- `app.py` registers all five pages and contains no training or dataset-download call. Prediction and continuation load only the selected model. Model/tokenizer loading uses Streamlit resource caching and process-level caching.
- Suggestion callbacks store prompt state in Streamlit session state. Missing required model/tokenizer/metadata files are reported with expected paths.

## Checkpoint and tokenizer verification

The tokenizer's effective vocabulary size and metadata vocabulary size are both **9,593**. The calculated ordered tokenizer mapping SHA-256 matches `preprocessing_metadata.json`:

`f2b73d7ae243444d65804fcdc4b3ef7de507a3f81c44a5f674f5d1f209ab3790`

| ID | Architecture | Expected mask | File exists | Exact size (bytes) | Output classes | Inference |
|---|---|---:|---:|---:|---:|---|
| A | SimpleRNN | false | Yes | 22,395,652 | 9,593 | Loaded; returned five valid predictions |
| B | LSTM | false | Yes | 22,840,388 | 9,593 | Loaded; returned five valid predictions |
| C | SimpleRNN | true | Yes | 22,395,651 | 9,593 | Loaded; returned five valid predictions |
| D | LSTM | true | Yes | 22,840,387 | 9,593 | Loaded; returned five valid predictions |

For each model, the loaded embedding's `mask_zero` value matched its experiment, the tokenizer hash/vocabulary checks passed, and the next-word inference path validated that model probabilities were finite, non-negative, and summed to approximately one. The V3 tokenizer and preprocessing metadata exist at `models/demo_v3/tokenizer.pkl` and `models/demo_v3/preprocessing_metadata.json`.

The following supporting V3 artifacts are present: `results/demo_v3/metrics.json`, the model-comparison, training-curves, and duration charts, and all four per-experiment `training_history.json` files. The app reads results from these saved artifacts; it does not recreate or hardcode the evaluation.

Demo V2's tokenizer, RNN/LSTM checkpoints, metrics, history, and comparison chart were also found. The two inference pages keep Demo V2 selectable; include its three runtime artifacts if you want that selector to work in the Cloud deployment.

## Dependency and Python compatibility audit

The tested environment is Python **3.11.9**, TensorFlow **2.17.1**, NumPy **1.26.4**, Matplotlib **3.8.4**, and Streamlit **1.37.0**. These versions match the runtime pins. `pip check` reported no broken requirements.

Direct application imports require TensorFlow, NumPy, Matplotlib, and Streamlit. Pandas and PyArrow are not directly imported by the application; Streamlit and Hugging Face dependencies may install their own transitive dependencies. Plotly and scikit-learn are not imported or used. `datasets` is needed only for preprocessing/download workflows, and `pytest` only for tests.

Dependency files now have distinct purposes:

- `requirements.txt`: runtime dependencies for Streamlit Community Cloud.
- `requirements-training.txt`: runtime dependencies plus Hugging Face `datasets`.
- `requirements-dev.txt`: training dependencies plus `pytest`.

The inference import path was also checked in a subprocess with imports of `datasets` deliberately blocked; model loading and prediction still succeeded. Thus app startup/inference do not require Hugging Face network access or cached dataset files.

Use Python 3.11 in the Community Cloud app's runtime settings. The current pins were exercised locally; a new Cloud build should still be confirmed after you create the GitHub repository.

## Git ignore, secrets, and repository size

The updated ignore rules cover virtual environments, Python caches, pytest cache, temporary/editor files, logs/checkpoints, environment files, Streamlit secrets, raw/preprocessed datasets, and non-runtime model/checkpoint files. Specific allow-rules make V3 inference artifacts and optional V2 inference artifacts available to Git despite the generic model/tokenizer exclusions.

Ignore behavior was verified against a temporary Git repository for 33 representative paths, including required V3/V2 models and tokenizers, unrelated model files, data arrays/raw data, `.env`, `.env.example`, and `.streamlit/secrets.toml`. All expected files were either ignored or allowed.

The workspace has **125 files / 287,917,181 bytes (about 274.6 MiB)** excluding `.venv`, `.pytest_cache`, and `__pycache__`. This includes unnecessary-for-deployment legacy checkpoints and WikiText raw/processed data. After the final ignore rules, the actual untracked non-ignored list contains **66 files / 119,674,857 bytes (about 114.13 MiB)**, including optional Demo V2 runtime models. The V3-only runtime set is **14 files / 91,055,255 bytes (about 86.84 MiB)**.

The largest individual V3 checkpoint is 22,840,388 bytes (about 21.8 MiB); no required checkpoint approaches GitHub's 100 MB individual-file limit. Git LFS is not required for the current model sizes. It remains an option to reduce normal Git history growth. External model hosting would require adding and testing a download/authentication mechanism; the current app expects repository-relative artifacts and has no such mechanism.

No secret contents were inspected. The ignore file excludes `.env*` files other than an example template and `.streamlit/secrets.toml`. All files in the actual non-ignored status list were scanned for absolute Windows user paths after the metadata cleanup; none remain. Confirm there are no credentials staged before committing.

## Streamlit Community Cloud readiness

- Entry point: `app.py`.
- Paths: derived from the source file's project directory; no absolute Windows path is used by inference.
- V3 model mapping: A-D each route to their own `models/demo_v3/experiments/<ID>/best_model.keras` and use `models/demo_v3/tokenizer.pkl`.
- V2 model mapping: Demo V2 RNN/LSTM selection routes to the V2 tokenizer and checkpoints, never the V3 tokenizer.
- Caching: both Streamlit `cache_resource` and the prediction module's loader cache are in place.
- Training and Hugging Face downloads: not invoked on app startup.
- Navigation/session state: all five pages render in AppTest; the suggestion callback appends a word and re-runs prediction.
- Missing files: the prediction/continuation pages report absent tokenizer, metadata, or checkpoints. Missing metrics/histories/charts are handled by the corresponding report/dashboard UI.
- Streamlit configuration: dark theme, headless server, and usage-statistics opt-out are configured in `.streamlit/config.toml`.
- Clean startup: local server health endpoint returned **HTTP 200, `ok`**. Port 8501 was already occupied, so validation used port 8502 and stopped only the server started for this check.

## Test results

Final pre-push full test suite: **82 passed**, with 13 Matplotlib/PyParsing deprecation warnings. The suite includes actual loading/inference checks for all four Demo V3 checkpoints, all three continuation strategies, and portable deployment metadata paths.

## Deployment checklist for manual GitHub/Cloud setup

1. Initialize Git in the project root and review `git status` plus `git check-ignore -v` for the required V3 files.
2. Commit source and the required V3 tokenizer, metadata, checkpoints, metrics, charts, and histories.
3. To retain Demo V2 functionality in the deployed app, include its tokenizer and both checkpoints too.
4. Create the Streamlit Community Cloud app with `app.py` as the entry point and Python 3.11.
5. Confirm the Cloud build installs `requirements.txt`, then test both prediction pages and the comparison/dashboard pages.
6. Do not commit raw datasets, preprocessed arrays, local environments, caches, or credentials.
