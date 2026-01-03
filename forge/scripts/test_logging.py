#!/usr/bin/env python3
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
file_handler = logging.FileHandler('../logs/test_logging.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)
log.info("Test logging started - this should appear in console and ../logs/test_logging.log")

print("Check ../logs/test_logging.log for the log entry.")