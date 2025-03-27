"""Configuration loader module.

Provides functions to locate the project root (by finding pyproject.toml) and to load
YAML configuration files from the config/ directory.
"""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Find pyproject.toml location and return project root."""
    current_path = Path(__file__).absolute()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    message = "pyproject.toml not found in directory hierarchy"
    raise FileNotFoundError(message)


def load_config(config_name: str) -> dict[str, Any]:
    """Load YAML config from config/ directory."""
    root_path = get_project_root()
    config_path = root_path / "config" / f"{config_name}.yaml"

    with config_path.open() as f:  # Use Path.open() instead of open()
        config = yaml.safe_load(f)

    # Resolve paths relative to project root.
    if "paths" in config:
        for key, value in config["paths"].items():
            if value and isinstance(value, str):
                config["paths"][key] = str(root_path / value)

    return config
