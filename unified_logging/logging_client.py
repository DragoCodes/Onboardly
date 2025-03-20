"""Logging client module.

This module implements a logging client that sends log messages over the network
to a logging server, enabling multiple processes to consolidate logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import zmq
from zmq.log.handlers import PUBHandler

if TYPE_CHECKING:
    from config_types import LoggingConfigs
    from loguru import Logger


def setup_network_logger_client(
    logging_configs: LoggingConfigs, logger: Logger,
) -> None:
    """Set up a network logger client that sends log messages via ZMQ.

    Args:
        logging_configs (LoggingConfigs): The logging configuration.
        logger (Logger): The Loguru logger instance.

    """
    zmq_socket = zmq.Context().socket(zmq.PUB)
    zmq_socket.connect(f"tcp://127.0.0.1:{logging_configs.log_server_port}")
    handler = PUBHandler(zmq_socket)

    # Remove previous settings to prevent logging to stderr and log only to file.
    logger.remove()
    logger.add(
        handler,
        format=logging_configs.client_log_format,
        enqueue=True,
        level=logging_configs.min_log_level,
        backtrace=True,  # Detailed error traces.
        diagnose=True,   # Enable exception diagnostics.
    )
