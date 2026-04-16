# Utils Package

This package now provides only low-level reusable helpers. It is no longer the owner of runtime config composition, promotion thresholds, release-record construction, or data-prep config merging.

## Structure

The package is organized into two categories:

### Core Utilities (Atomic, Reusable)

These are low-level utilities that can be used independently across the project:

- **`config_loader.py`** - YAML configuration file loading
- **`env_loader.py`** - Environment variable loading from `.env` files
- **`path_utils.py`** - Path resolution utilities
- **`type_utils.py`** - Type conversion and parsing utilities

### Focused Runtime Helpers

These helpers are still shared across multiple domains:

- **`mlflow_utils.py`** - MLflow run management and Azure ML detection
- **`metrics.py`** - Model evaluation metrics calculation

## Core Utilities

### `config_loader.py`

YAML configuration file loading utilities.

**Functions:**

- `load_config(config_path: str) -> Dict[str, Any]` - Load configuration from YAML file
- `get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any` - Get nested configuration value using dot notation

**Example:**

```python
from utils.config_loader import get_config_value, load_config

# Load YAML config
config = load_config("configs/data.yaml")

# Get nested value
test_size = get_config_value(config, "data.test_size", default=0.2)
```

### `env_loader.py`

Environment variable loading utilities.

**Functions:**

- `load_env_file(config_path: Optional[str] = None) -> None` - Load environment variables from config.env file
- `get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]` - Get environment variable with validation

**Example:**

```python
from utils.env_loader import get_env_var, load_env_file

# Load config.env
load_env_file()

# Get environment variable
subscription_id = get_env_var("AZURE_SUBSCRIPTION_ID", required=True)
workspace_name = get_env_var("AZURE_WORKSPACE_NAME", default="default-workspace")
```

### `path_utils.py`

Path resolution utilities.

**Functions:**

- `get_project_root() -> Path` - Get the project root directory
- `get_config_env_path(config_path: Optional[str] = None) -> Path` - Get the path to config.env file

**Example:**

```python
from utils.path_utils import get_config_env_path, get_project_root

# Get project root
root = get_project_root()

# Get config.env path
config_path = get_config_env_path()
```

### `type_utils.py`

Type conversion and parsing utilities.

**Functions:**

- `parse_bool(value: Any, *, default: bool) -> bool` - Parse loose truthy/falsey values

**Example:**

```python
from utils.type_utils import parse_bool

# Parse boolean from various formats
value1 = parse_bool("true", default=False)  # True
value2 = parse_bool("yes", default=False)   # True
value3 = parse_bool("0", default=True)      # False
value4 = parse_bool(None, default=True)     # True (default)
```

## Domain-Specific Utilities

### Runtime Ownership Moved Out

These responsibilities were intentionally moved to thicker domain owners:

- Azure workspace, training-default, promotion, and release config now live in `src/config/runtime.py`
- data-prep config merging now lives in `src/data/config.py`
- release-record construction now lives in `src/release/workflow.py`

### `mlflow_utils.py`

MLflow run management and Azure ML environment detection.

**Functions:**

- `is_azure_ml() -> bool` - Check if running in Azure ML environment
- `get_run_id(run_obj: Any) -> str` - Extract run ID from MLflow run object
- `get_active_run()` - Get the active MLflow run
- `start_parent_run(experiment_name: str, run_name: str = "Churn_Training_Pipeline")` - Start a parent MLflow run
- `start_nested_run(run_name: str)` - Start a nested MLflow run

**Example:**

```python
from utils.mlflow_utils import is_azure_ml, start_nested_run, start_parent_run

# Check environment
if is_azure_ml():
    print("Running in Azure ML")

# Start parent run
with start_parent_run("churn-prediction-experiment"):
    # Start nested run
    nested_run, run_id = start_nested_run("data_prep")
    # ... your code ...
```

### `metrics.py`

Model evaluation metrics calculation.

**Functions:**

- `calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]` - Calculate core evaluation metrics

**Example:**

```python
from utils.metrics import calculate_metrics
import numpy as np

y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2])

metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
# Returns: {
#     "accuracy": 0.75,
#     "precision": 1.0,
#     "recall": 0.5,
#     "f1": 0.67,
#     "roc_auc": 0.75
# }
```

## Usage

### Importing Utilities

Import utilities from their concrete module owners:

```python
from utils.config_loader import get_config_value, load_config
from utils.metrics import calculate_metrics
from utils.mlflow_utils import is_azure_ml, start_parent_run
```

### Module Dependencies

The utilities are organized to minimize dependencies:

```
Core Utilities:
├── config_loader.py
├── env_loader.py (depends on path_utils)
├── path_utils.py
└── type_utils.py

Focused Shared Helpers:
├── mlflow_utils.py
└── metrics.py

Higher-level domain owners live outside this package:
├── src/config/runtime.py
├── src/data/config.py
├── src/models/factory.py
└── src/release/workflow.py
```

## Design Principles

1. **Atomic Functions**: Each function has a single, well-defined responsibility
2. **Reusability**: Core utilities can be used independently across the project
3. **No Duplication**: Shared logic is extracted to common modules
4. **Type Hints**: All functions include type hints for better IDE support
5. **Documentation**: All functions include docstrings with examples
6. **Error Handling**: Functions provide clear error messages when validation fails

## Testing

Each utility module should have corresponding tests. The atomic nature of these utilities makes them easy to test in isolation.

## Contributing

When adding new utilities:

1. **Determine if it's core or domain-specific**: Core utilities should be generic and reusable. Domain-specific utilities can depend on core utilities.
2. **Follow naming conventions**: Use descriptive function names and follow existing patterns.
3. **Add type hints**: All functions should include type hints.
4. **Write docstrings**: Include docstrings with Args, Returns, Raises, and Examples.
5. **Avoid rebuilding a barrel**: import from concrete modules instead of reintroducing `from utils import ...`.
6. **Update this README**: Document new ownership only when the helper truly belongs in `utils/`.
