# Configuration Files

This repo uses two config layers:

- `config.env.example`
  - infrastructure, workspace, data-asset, and operator-facing shell values
- `configs/*.yaml`
  - versioned policy and workflow defaults used by the codebase
- `configs/assets.yaml`
  - lightweight asset ownership and lineage metadata; this documents canonical names and does not replace `config.env`

The active YAML files are:

- `assets.yaml`
  - canonical Azure ML asset, component, model, environment, and endpoint names used for lineage tags
- `data.yaml`
  - preprocessing defaults plus validation thresholds
- `data_smoke.yaml`
  - positive-path smoke data-prep and validation defaults for local preflight checks
- `train.yaml`
  - fixed-training behavior, active models, hyperparameters, and the canonical training experiment name
- `train_smoke.yaml`
  - fixed-training smoke profile with cheap model settings and smoke-scoped promotion thresholds
- `hpo.yaml`
  - production sweep policy
- `hpo_smoke.yaml`
  - tiny sweep profile for quick validation and smoke runs
- `mlflow.yaml`
  - legacy compatibility fallback for MLflow experiment naming if `train.yaml` does not define one

## How the configs are used

| Flow | Configs referenced | Notes |
| --- | --- | --- |
| Local/AML data prep (`src/data_prep.py`, `aml/components/data_prep.yaml`) | `data.yaml`, `data_smoke.yaml` | `data.target_column` is the canonical target owner for both prep and validation; `run_pipeline.py --data-config ...` selects the AML policy file. |
| Validation (`src/validate_data.py`, `aml/components/validate_data.yaml`) | `data.yaml`, `data_smoke.yaml` | `validation.*` owns thresholds; target ownership stays under `data.target_column`; AML receives the selected data config as a component input. |
| Production training (`run_pipeline.py`, `src/train.py`, AML `train.yaml` component) | `train.yaml`, `train_smoke.yaml`, `mlflow.yaml` | `train.yaml` is the canonical owner for production experiment naming and training defaults; `train_smoke.yaml` is opt-in via `run_pipeline.py --train-config`; `mlflow.yaml` is a legacy fallback only. |
| Promotion utility (`src/promotion/promote_model.py`, AML `promote_model.yaml` component) | `train.yaml`, `train_smoke.yaml`, Azure ML approved-model tags | Reads `promotion.*` thresholds from the selected training config and compares candidate metrics against baseline metrics materialized from the latest approved Azure ML model metadata. |
| Script-first sweeps (`run_hpo.py`) | `hpo.yaml`, `hpo_smoke.yaml` | `run_hpo.py` uses `configs/hpo.yaml` by default and can target an alternate profile such as `configs/hpo_smoke.yaml` via `--config`; the notebook still owns best-config review and writeback. |
| Script-first release (`run_release.py`) | `config.env`, job outputs, Azure ML registry metadata | Downloads `promotion_decision`, blocks rejected candidates, registers the approved MLflow model, and can deploy it to the managed online endpoint. |

## Overriding values

- Point a script at an alternate file: `python src/train.py --config configs/train_smoke.yaml`.
- Run local data prep against the smoke fixture: `python src/data_prep.py --config configs/data_smoke.yaml`.
- Submit fixed training with smoke-scoped data, training, and promotion policy: `python run_pipeline.py --data-config configs/data_smoke.yaml --train-config configs/train_smoke.yaml`.
- Override training-level keys without editing YAML: `python src/train.py --set use_smote=false --set class_weight=balanced --set random_state=7`.
- Override model hyperparameters with `model.param=value`: `python src/train.py --set rf.max_depth=6`.
- Submit the smoke HPO profile without editing files: `python run_hpo.py --config configs/hpo_smoke.yaml --processed-data-uri azureml://jobs/<data-prep-job>/outputs/processed_data`.
- Azure ML sweeps automatically merge the selected HPO YAML profile with the CLI overrides that `run_sweep_trial.py` emits.

### Precedence (highest → lowest)

1. CLI / notebook overrides (`--config`, `--set`, or Azure ML sweep parameters)
2. Values stored in the selected YAML file
3. Legacy compatibility fallbacks such as `configs/mlflow.yaml`
4. Hard-coded defaults inside the scripts

Keep any experimental variations under `configs/` so they remain versioned and easy to diff. Remove unused files once a workflow is decommissioned to avoid stale settings.
