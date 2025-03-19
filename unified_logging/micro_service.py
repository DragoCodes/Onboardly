# GiG

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from config_types import LoggingConfigs
from fastapi import FastAPI
from logging_client import setup_network_logger_client
from loguru import logger

# Segregate imports done solely for static typing
if TYPE_CHECKING:
    pass


def get_logging_configs() -> LoggingConfigs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path")
    args = parser.parse_args()

    config_file_name = Path(args.config_file_path)
    return LoggingConfigs.load_from_path(config_file_name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging_configs = get_logging_configs()
    setup_network_logger_client(logging_configs, logger)
    yield


#################### Do not change this

nlu_app = FastAPI(
    title="Intent Detection API",
    description="This microservice takes a string as input and determines the top-K relevant intents",
    version="1.0",
    lifespan=lifespan,
)


@nlu_app.get("/ping")
def ping() -> dict[str, str]:
    logger.info("I am logging from a microservice!")
    return {"pong": "ping"}


if __name__ == "__main__":
    uvicorn.run(
        "micro_service:nlu_app",
        host="127.0.0.1",
        port=12345,
        log_level="warning",
        workers=4,
    )
