# src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Find pyproject.toml location and return project root"""
    current_path = Path(__file__).absolute()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("pyproject.toml not found in directory hierarchy")

def load_config(config_name: str) -> Dict[str, Any]:
    """Load YAML config from config/ directory"""
    root_path = get_project_root()
    config_path = root_path / "config" / f"{config_name}.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve paths relative to project root
    if "paths" in config:
        for key, value in config["paths"].items():
            if value and isinstance(value, str):
                config["paths"][key] = str(root_path / value)
                
    return config