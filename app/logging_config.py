import logging
import sys


def configure_logging(level: str = "INFO", stream=None) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=stream or sys.stdout,
        force=True,
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3", "huggingface_hub", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
