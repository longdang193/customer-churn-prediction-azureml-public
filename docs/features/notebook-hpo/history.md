# History

## 2026-04-10

- Seeded the initial feature contract from the live notebook implementation in `notebooks/main.ipynb`.
- Recorded `notebooks/main.ipynb` as the current code reference because the repo docs still drift between `main.ipynb` and `hpo_manual_trials.ipynb`.
- Normalized the current HPO notebook reference to `notebooks/main.ipynb` while recording the target shift toward a script-first Azure ML entrypoint.
- Added `run_hpo.py` as the canonical script-first HPO submission path while preserving `notebooks/main.ipynb` as the review surface.
- Split the smoke sweep profile out of `configs/hpo.yaml` into `configs/hpo_smoke.yaml` to keep the production sweep config clean.
- Added explicit `run_hpo.py --config ...` support so the smoke HPO profile is reachable from the script-first submission path instead of being doc-only.

## 2026-04-11

- Aligned the HPO smoke profile with the broader e2e smoke surface alongside smoke data, fixed-training smoke config, and endpoint smoke payload ownership.

## 2026-04-12

- Added `run_hpo_pipeline.py` as the canonical end-to-end HPO submission path from raw data while preserving `run_hpo.py` for processed-data reruns.
- Added Azure ML HPO trial and summary components so HPO can surface one parent graph plus summary artifacts instead of only disconnected sweep pages.
- Recorded the new HPO summary artifact contract: `hpo_summary.json`, `hpo_summary.md`, and `hpo_manifest/step_manifest.json`.
- Added per-family `*_hpo_manifest` and `*_train_manifest` outputs to the HPO parent pipeline so operators can inspect one branch without opening Azure-generated controller/trial jobs first.
- Added `inspect_hpo_run.py` as the repo-owned HPO inspection helper for downloaded or remote parent runs.
- Clarified the operating split between smoke HPO (`configs/hpo_smoke.yaml`) and heavier-weight ranking HPO (`configs/hpo.yaml` by default).
- Unified the HPO trial artifact contract with fixed training so direct and pipeline HPO now share `model_output`, `mlflow_model`, `candidate_metrics`, `train_manifest`, and `hpo_manifest`.
- Added per-family `*_model_output` and `*_mlflow_model` outputs plus canonical winner outputs to the HPO parent pipeline.
- Added a winner materialization step so the HPO parent now publishes one stable winner handoff surface instead of forcing operators to reconstruct it from branch artifacts.
