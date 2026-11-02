#============================================
# forge/tests/test_logging.py
# Purpose: Diagnostic script to verify the project's logging configuration.
# Role: Environment Verification / Diagnostic Layer
# Used by: Troubleshooting and development
#============================================

import sys
import logging
from pathlib import Path

#============================================
# Function: main
# Purpose: Tests console and file logging connectivity.
# Args: None
# Returns: None
#============================================
def main() -> None:
    """
    Diagnostic check for logging functionality.
    """
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_logging.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Test logging session started - outputting to console and {log_file}")
    
    print(f"✅ Logging test complete. Verify entries in: {log_file}")


if __name__ == "__main__":
    main()