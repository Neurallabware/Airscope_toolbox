"""Logging helpers for tqdm output redirection."""

import io
import logging


class TqdmToLogger(io.StringIO):
    """Output stream for tqdm that writes to logging."""

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)
