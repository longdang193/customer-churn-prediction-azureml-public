# Stage Docs

This directory holds stage contracts for the project’s pipeline-heavy workflow boundaries.

Stage contracts now follow the same source/generated split as feature
contracts:

- `*.source.yaml` files are human-owned stage intent.
- `*.yaml` files are generated current contracts.
- generated refs come from feature metadata, canonical code/test markers,
  doc frontmatter, and YAML `# @architecture` metadata in configs/components.

Stage participation is now modeled on both sides:

- stage sources declare `primary_features` and optional `supporting_features`
- feature sources declare `stage_participation`
- generated stage `feature_refs` appear only after those two sides agree
- generated stage `capability_refs` are narrowed to the feature capability ids
  explicitly declared for that stage

The initial stage map for this repo is:

- `data_validate`
- `data_prep`
- `model_sweep`
- `fixed_train`
- `model_promote`
- `online_deploy`
- `monitor`

Stages explain where work happens in the workflow. Features explain what capability exists and how it evolves.

Do not hand-edit generated stage refs such as `code_refs`, `doc_refs`,
`config_refs`, or `component_refs`. Patch the owning metadata and rerun:

```powershell
.\.venv\Scripts\python.exe scripts\sync_architecture_docs.py
```

Release-oriented stages such as `model_promote`, `online_deploy`, and `monitor` should be read together. They define the portfolio-facing release story from approved model to deployed service to post-release follow-up.
