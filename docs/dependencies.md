# Dependencies

Centralized project dependencies and workflow for reproducible environments. All commands assume you are inside the project root with a Python **3.9** virtual environment activated (see `docs/python_setup.md`).

## Files

- `requirements.in` — application/runtime dependencies (unpinned)
- `dev-requirements.in` — developer tooling (unpinned)
- `requirements.txt` — pinned, compiled from `requirements.in` (generated)
- `dev-requirements.txt` — pinned, compiled from `dev-requirements.in` (generated)

## Manage dependencies with pip-tools

Install pip-tools (inside the venv):

```bash
python -m pip install --upgrade pip
python -m pip install pip-tools
```

Pin runtime and dev requirements (only needed when `.in` files change):

```bash
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```

Install from the pinned files (or re-sync an existing venv):

```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional for non-dev use

# or keep an existing environment in sync
pip-sync requirements.txt dev-requirements.txt
```

## Important Notes

- Run `pip-compile` with Python 3.9 so the generated headers show `Python 3.9`; other versions may produce incompatible pins.
- All packages listed in `requirements.in` are runtime dependencies required for the project.
- If you change `requirements.in` or `dev-requirements.in`, re-run `pip-compile` and commit the updated `*.txt` files along with the `requirements.txt` / `dev-requirements.txt`.
