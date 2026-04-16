# Source Code (`/src`)

End-to-end churn ML pipeline: data prep → training. All stages can be driven by YAML configs and log to MLflow.

Repo layout note:

- canonical lifecycle entrypoints remain at the repo root
- secondary operator helpers now live under `tools/`
- utility modules such as `hpo_utils.py` are separate from the operator entrypoint taxonomy
- `src/` stays the runtime/domain implementation layer rather than becoming a second CLI surface

## Ownership Map

- `src/azureml/`
  - shared Azure ML adapter layer for AML client creation, input builders, registry helpers, and deployment helpers used by orchestration entrypoints
- `src/config/`
  - central runtime ownership for Azure workspace settings, training defaults, and promotion thresholds
- `src/data/`
  - preprocessing logic plus data-domain config helpers used by `data_prep.py`
- `src/models/`
  - model definitions and model-factory behavior
- `src/training/`
  - training orchestration, hyperparameter parsing, and MLflow-facing training flow
- `src/promotion/`
  - candidate-versus-baseline evaluation logic
- `src/release/`
  - release gating, registry metadata helpers, and release-record construction
- `src/utils/`
  - small atomic helpers such as YAML loading, env loading, metrics, MLflow utility functions, and shared step-manifest writing

## Quickstart

### Azure ML Pipeline Execution

```bash
# 1) Ensure config.env is configured with Azure ML settings
# Note: Scripts now load config.env automatically, but you can also source it manually:
# set -a && source config.env && set +a

# 2) Submit the regular training pipeline (runs validation + train.yaml component)
python run_pipeline.py

# 2a) Optional low-cost fixed-training smoke profile
python run_pipeline.py --data-config configs/data_smoke.yaml --train-config configs/train_smoke.yaml

# 2b) Register and optionally deploy a promoted fixed-training job
python run_release.py --job-name <FIXED_TRAIN_JOB_NAME> --deploy

# 3) Submit the script-first HPO sweep against prepared data
python run_hpo.py --processed-data-uri azureml://jobs/<data-prep-job>/outputs/processed_data

# 4) After HPO completes, review the best trial in notebooks/main.ipynb
#    and write the chosen model configuration back into configs/train.yaml

# 5) Train the best model with optimized hyperparameters
python run_pipeline.py
```

> Tip: Run `mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --port 5000` to inspect runs if you mirror MLflow tracking locally.

---

## Models

The project supports three models:
- **Logistic Regression** (`logreg`) - Fast baseline model
- **Random Forest** (`rf`) - Tree-based ensemble model, hyperparameters optimized in HPO
- **XGBoost** (`xgboost`) - Gradient boosting model

**Model Selection:**
- **Regular mode**: Models to train are specified in `configs/train.yaml` under the `models` key (e.g., `models: [logreg, rf]`)
- **HPO mode**: Each trial trains one model type (specified via `--model-type`) with sampled hyperparameters
- The best model (highest F1 score) is selected and logged as `model_type` tag in MLflow
- After HPO, the notebook review surface is used to select the best configuration and write it back into `configs/train.yaml`

## Pipeline Overview

### Azure ML Pipeline

```
Azure data asset (e.g. churn-data:1)
  └─► run_pipeline.py (regular training)
         ├─► validate_data component → validation report + summary
         ├─► data_prep component → processed dataset (uri_folder output)
         ├─► train component (train.yaml) → trains models from config, logs to MLflow
         │      ├─► emits best-model pickle artifact
         │      ├─► emits deployable MLflow model bundle
         │      └─► emits best candidate metrics
         └─► promote_model component → candidate-vs-approved-baseline decision artifact

After fixed training:
  └─► run_release.py → downloads promotion outputs, registers approved MLflow model, optionally deploys endpoint

  └─► run_hpo.py (HPO sweep)
         ├─► Reuses a prepared dataset URI from the data-prep stage
         └─► run_sweep_trial.py → sweep with multiple trials
                ├─► Each trial trains one model type with sampled hyperparameters
                ├─► Logs all trials to MLflow
                └─► Picks best based on configs/hpo.yaml

After HPO:
  └─► notebooks/main.ipynb review flow → confirms the best configuration and updates configs/train.yaml
  └─► run_pipeline.py → trains best model with optimized hyperparameters
```

### Key Differences: Local vs Azure ML

| Feature | Local Execution | Azure ML Execution |
|---------|----------------|-------------------|
| MLflow Runs | Nested runs supported | Uses active run (no nesting) |
| Model Saving | `mlflow.sklearn.log_model()` | Pickle artifact plus MLflow model bundle |
| Model Loading | `mlflow.sklearn.load_model()` | `joblib.load()` from outputs |
| Artifact Logging | Full MLflow API support | Limited (uses outputs directory) |

---

## Scripts

### `data_prep.py` — prepare raw data

- Drops uninformative columns, encodes categoricals, splits train/test, scales numeric features (fit on train; transform on test), and writes artifacts.
- Reads defaults from `configs/data.yaml`; CLI flags override.

**Examples**

```bash
# Default config
python src/data_prep.py

# Custom output + seed
python src/data_prep.py --output data/processed_custom --random-state 1337
```

**Outputs (default: `data/processed/`)**

- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `encoders.pkl`, `scaler.pkl`
- `metadata.json` (feature names, dropped/encoded/scaled columns, target)

> Note: We use label encoding for categoricals. Tree models handle this well; for linear models consider one-hot encoding.

---

### `train.py` — train and log models

- Trains models specified in `configs/train.yaml` (regular mode) or a single model type (HPO mode), optional SMOTE on the **train** split, logs params/metrics/artifacts to MLflow.
- **Model Selection**: In regular mode, models are determined from `configs/train.yaml` → `training.models`. In HPO mode, use `--model-type` to specify a single model.
- **Azure ML Compatibility**: When running in Azure ML, nested runs are automatically disabled and models are saved as pickle files to the outputs directory (Azure ML automatically captures these as artifacts). In local execution, nested runs are used as before.
- Supports hyperparameter overrides via `--set model.param=value` (useful for manual tuning or integration with Azure ML HyperDrive sweeps).
- **HPO Mode**: When `--model-type` is provided, trains only the specified model. Hyperparameters are passed via `--set` flags (used by Azure ML HyperDrive sweeps).

**Examples**

```bash
# Use defaults from configs/train.yaml
# configs/mlflow.yaml is only a legacy fallback if train.yaml omits the experiment name
# Trains all models specified in configs/train.yaml → training.models
python src/train.py

# Train with custom experiment name
python src/train.py --experiment-name churn-experiments

# Enable SMOTE for imbalanced data
python src/train.py --use-smote

# Override training-level settings without editing YAML
python src/train.py --set use_smote=false --set class_weight=balanced --set random_state=7

# Override hyperparameters manually (applies to all models that have those params)
python src/train.py --set rf.n_estimators=200 --set rf.max_depth=15

# HPO mode: train single model with hyperparameters via --set flags
python src/train.py --model-type rf --set rf.n_estimators=200 --set rf.max_depth=15
```

> Implementation note: when `class_weight='balanced'` and SMOTE is **off**, XGBoost maps imbalance via `scale_pos_weight`.

> **Azure ML Note**: In Azure ML environments, the script automatically detects the Azure ML context and:
> - Uses the existing active MLflow run instead of creating nested runs
> - Saves models as pickle files to `AZUREML_ARTIFACTS_DIRECTORY` or `AZUREML_OUTPUT_DIRECTORY` (automatically captured by Azure ML)
> - Logs model paths as MLflow tags for reference
> - Handles MLflow API limitations gracefully (some artifact APIs have different signatures in Azure ML)

---

### Notebook Review Surface — apply the chosen HPO result

- `notebooks/main.ipynb` remains the live review surface after the sweep completes
- Review the completed sweep results there, choose the promoted configuration, and update `configs/train.yaml`
- The script-first operational path stops at `run_hpo.py`; the notebook owns the final human review/writeback step

---

## Configuration Reference

Configs live in `configs/`. CLI flags always override config values.

- **`configs/data.yaml`** — controls raw input path, output dir, test split, random seed, target column, columns to drop/encode, and validation thresholds.
- **`configs/data_smoke.yaml`** — local smoke data-prep/validation profile using `data/smoke/positive/churn_smoke.csv` and `data/processed_smoke`.
- **`configs/assets.yaml`** — lightweight Azure ML asset, component, model, environment, and endpoint name manifest used for lineage tags.
- **`configs/train.yaml`** — lists models to train, owns the canonical training experiment name, and carries promotion-threshold defaults for the promotion utility.
- **`configs/train_smoke.yaml`** — fixed-training smoke profile passed through `run_pipeline.py --data-config configs/data_smoke.yaml --train-config configs/train_smoke.yaml`.
- **`configs/hpo.yaml`** — defines the production search space, budget, and early stopping for sweeps.
- **`configs/hpo_smoke.yaml`** — tiny sweep profile for smoke validation.
- **`configs/mlflow.yaml`** — legacy fallback for MLflow experiment naming if `train.yaml` does not define one.

Runtime loading ownership is intentionally split:

- `src/azureml/` owns reusable Azure ML SDK integration shared by `run_pipeline.py`, `run_hpo.py`, and `run_release.py`
- `src/config/runtime.py` owns Azure workspace, release, training-default, and promotion-threshold resolution
- `src/config/assets.py` owns lightweight asset-manifest loading and lineage tag construction
- `src/data/config.py` owns data-prep config merging for `src/data_prep.py`
- `src/utils/config_loader.py` remains the low-level YAML loader used by both domains
- `src/utils/step_manifest.py` owns the shared `step_manifest.json` schema for validation, data prep, training, and local smoke preflight outputs

---

## Hyperparameter Optimization (Azure ML HyperDrive)

The project includes an optional sweep job powered by Azure ML HyperDrive. The sweep uses the search space defined in `configs/hpo.yaml` for the configured model(s).

**Submit the sweep job**

```bash
# Ensure config.env is configured with Azure ML settings
# Note: run_hpo.py now loads config.env automatically
python run_hpo.py --processed-data-uri azureml://jobs/<data-prep-job>/outputs/processed_data

# Use the smoke sweep profile explicitly
python run_hpo.py --config configs/hpo_smoke.yaml --processed-data-uri azureml://jobs/<data-prep-job>/outputs/processed_data
```

The sweep job:

1. Reuses a prepared dataset URI from the data-prep stage.
2. Launches a sweep over the specified hyperparameters (configured in `configs/hpo.yaml`), logging all trials to MLflow.
3. Each trial trains one model type with sampled hyperparameters.
4. Surfaces the best-performing configuration via MLflow metrics/tags (`model_type` tag).

**After HPO completes:**

1. Use `notebooks/main.ipynb` to review completed sweep results and write the selected configuration back into `configs/train.yaml`.
2. Train the best model:
   ```bash
   python run_pipeline.py
   ```

**Configuration**

The HPO behavior is controlled by the selected HPO YAML profile (`configs/hpo.yaml` by default, or `--config configs/hpo_smoke.yaml` for smoke runs):

- `metric`: Primary metric to optimize (e.g., `f1`, `roc_auc`)
- `mode`: Optimization direction (`max` or `min`)
- `sampling_algorithm`: Sampling method (e.g., `random`, `grid`, `bayesian`)
- `budget.max_trials`: Maximum number of trials to run
- `budget.max_concurrent`: Maximum parallel trials
- `early_stopping`: Configuration for early termination policies
- `search_space`: Hyperparameter ranges per model (supports `rf`, `xgboost`, `logreg`)

Results and individual trials can be inspected in Azure ML Studio using the URL printed after submission.

---

## Troubleshooting

### General Issues

- **Missing parent run ID:** When reviewing sweep results in the notebook, ensure you use the correct parent run ID from the sweep job. Check Azure ML Studio for the sweep job run ID.
- **No models trained:** Check training logs; if all models error, the run won't complete successfully. Verify `configs/train.yaml` has `models` specified under `training` section.
- **Model selection:** Models are determined from `configs/train.yaml` → `training.models` in regular mode. There is no `--models` CLI argument.
- **Hyperparameter override syntax:** Use `--set model.param=value` format. For boolean values, use `true`/`false` (lowercase). For `None`, use `none` (lowercase). Numeric values are parsed automatically.

### Azure ML Specific Issues

- **MLflow nested runs error:** The script automatically detects Azure ML and disables nested runs. If you see errors about active runs, ensure you're using the latest version of `train.py` that includes Azure ML detection.

- **Model artifact logging errors:** In Azure ML, `mlflow.sklearn.log_model()` and some artifact APIs have limitations. The script automatically falls back to saving models as pickle files to the outputs directory, which Azure ML captures automatically.

- **Model loading errors:** When loading models in Azure ML, the script looks for models in the outputs directory (`AZUREML_ARTIFACTS_DIRECTORY` or `AZUREML_OUTPUT_DIRECTORY`). Ensure models were saved during training.

- **Python version compatibility:** The project runtime, AML environment template, and CI smoke suite all target **Python 3.9**. Keep local development aligned with that baseline when reproducing pipeline behavior.

- **Data asset configuration:** `run_pipeline.py` resolves current and reference assets from `configs/assets.yaml`, with `DATA_*` env vars available only as local overrides. `run_hpo.py` expects a prepared data URI.
- **Release configuration:** `run_release.py` resolves model, endpoint, deployment, and sizing defaults from `configs/assets.yaml`, with `MODEL_NAME`, `ENDPOINT_NAME`, `DEPLOYMENT_NAME`, and `AML_ONLINE_*` env vars available only as local overrides.

For active release and monitoring troubleshooting, see
[`../docs/features/online-endpoint-deployment/release-validation.md`](../docs/features/online-endpoint-deployment/release-validation.md).
