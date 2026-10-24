import logging
import os


def setup_logger(service_name):
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        # File
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/{service_name}.log")
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
