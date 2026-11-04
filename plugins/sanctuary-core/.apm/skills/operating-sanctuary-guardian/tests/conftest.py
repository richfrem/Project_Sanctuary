"""Test isolation: never run the real usage probes (claude/agy CLIs) from checkout/checkin during tests."""
import os

os.environ["SANCTUARY_NO_USAGE"] = "1"
