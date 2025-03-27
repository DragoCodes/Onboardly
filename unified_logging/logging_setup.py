"""Module for setting up unified network logging.

This module provides a function to set up logging using a TOML configuration file.
It falls back to a default configuration if loading fails.
"""

from pathlib import Path

from loguru import logger

from unified_logging.config_types import LoggingConfigs
from unified_logging.logging_client import setup_network_logger_client

# Define project root relative to this file (unified_logging/)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "unified_logging" / "configs.toml"

def setup_logging(config_path: str = str(DEFAULT_CONFIG_PATH)) -> LoggingConfigs:
    """Set up unified network logging with a specified config file.

    Args:
        config_path (str): Path to the logging config file (default:
            project_root/unified_logging/configs.toml).

    Returns:
        LoggingConfigs: Loaded logging configuration.

    """
    try:
        config_file = Path(config_path)
        logging_configs = LoggingConfigs.load_from_path(config_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load logging config from {config_path}: {e}")
        logging_configs = LoggingConfigs()  # Fallback to defaults

    # Set up network logging.
    setup_network_logger_client(logging_configs, logger)

    logger.info(f"Unified logging initialized with config from {config_path}")
    return logging_configs

if __name__ == "__main__":
    setup_logging()
    logger.debug("Test debug message")
