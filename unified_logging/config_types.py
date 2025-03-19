# GiG

from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict


def load_toml(file_name: Path) -> dict:
    with file_name.open("rb") as file_obj:
        return tomllib.load(file_obj)


class LoggingConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "DEBUG"
    log_server_port: int = 9999
    server_log_format: str = "[{level}] | {message}"
    client_log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {file}: {line} | {message}"
    log_rotation: str = "00:00"
    log_file_name: str = "logs/logs.txt"
    log_compression: str = "zip"

    @staticmethod
    def load_from_path(file_path: str) -> "LoggingConfigs":
        configs: LoggingConfigs = LoggingConfigs.model_validate(
            load_toml(Path(file_path))
        )
        return configs
