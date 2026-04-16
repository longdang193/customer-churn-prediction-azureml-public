# Stage Docs

This directory holds stage contracts for the project’s pipeline-heavy workflow boundaries.

The initial stage map for this repo is:

- `data_validate`
- `data_prep`
- `model_sweep`
- `fixed_train`
- `model_promote`
- `online_deploy`
- `monitor`

Stages explain where work happens in the workflow. Features explain what capability exists and how it evolves.

Release-oriented stages such as `model_promote`, `online_deploy`, and `monitor` should be read together. They define the portfolio-facing release story from approved model to deployed service to post-release follow-up.
