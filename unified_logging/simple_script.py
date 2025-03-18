# GiG
import argparse
from pathlib import Path

from config_types import LoggingConfigs
from logging_client import setup_network_logger_client
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path")
    args = parser.parse_args()

    config_file_name = Path(args.config_file_path)

    logging_configs = LoggingConfigs.load_from_path(config_file_name)

    # Setup logging
    setup_network_logger_client(logging_configs, logger)
    logger.info("I am logging from a script")


if __name__ == "__main__":
    main()
