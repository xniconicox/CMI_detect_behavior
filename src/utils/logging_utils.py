import logging
from pathlib import Path


def setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with file and console handlers.

    Parameters
    ----------
    log_path : Path
        File to write log records.
    level : int, default logging.INFO
        Logging level for the root logger.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
