"""Module defining configuration types for unified logging.

This module provides functions to load TOML configuration and a Pydantic model
for logging configuration.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


def load_toml(file_name: Path) -> dict:
    """Load TOML configuration from a file.

    Args:
        file_name (Path): The path to the TOML file.

    Returns:
        dict: The loaded configuration dictionary.

    """
    with file_name.open("rb") as file_obj:
        return tomllib.load(file_obj)


class LoggingConfigs(BaseModel):
    """Pydantic model for logging configuration.

    Attributes:
        min_log_level: Minimum log level.
        log_server_port: Port for the logging server.
        server_log_format: Log format for server logs.
        client_log_format: Log format for client logs.
        log_rotation: Log rotation time.
        log_file_name: File name for the log file.
        log_compression: Log file compression type.

    """

    model_config = ConfigDict(extra="forbid")
    min_log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL",
    ] = "DEBUG"
    log_server_port: int = 9999
    server_log_format: str = "[{level}] | {message}"
    client_log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {file}: {line} | {message}"
    log_rotation: str = "00:00"
    log_file_name: str = "logs/logs.txt"
    log_compression: str = "zip"

    @staticmethod
    def load_from_path(file_path: str) -> "LoggingConfigs":
        """Load logging configuration from a file path.

        Args:
            file_path (str): The path to the TOML configuration file.

        Returns:
            LoggingConfigs: The loaded logging configuration.

        """
        configs: LoggingConfigs = LoggingConfigs.model_validate(
            load_toml(Path(file_path)),
        )
        return configs
