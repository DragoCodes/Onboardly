"""Simple script to initialize network logging and log a message.

This script loads logging configuration from a file and sets up network logging.
"""

import argparse
from pathlib import Path

from config_types import LoggingConfigs
from logging_client import setup_network_logger_client
from loguru import logger


def main() -> None:
    """Initialize logging and log a message."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path")
    args = parser.parse_args()

    config_file_name = Path(args.config_file_path)

    logging_configs = LoggingConfigs.load_from_path(config_file_name)

    # Setup network logging.
    setup_network_logger_client(logging_configs, logger)
    logger.info("I am logging from a script")

if __name__ == "__main__":
    main()
