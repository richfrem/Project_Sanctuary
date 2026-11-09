#!/usr/bin/env python3
"""verify_health.py - host preflight probe for the Sanctuary supervisor daemon.

Prints the Python version, platform and free disk space on the current drive, and
exits 0 if Python is >= 3.10, otherwise 1. Stdlib only.

Usage: python3 verify_health.py
"""
import shutil
import sys

MIN_PYTHON = (3, 10)


def check_environment() -> dict:
    """Print and return the Python version, platform and free disk space (GB)."""
    free_gb = shutil.disk_usage(".").free / 1024 ** 3
    info = {"python": sys.version.split()[0], "platform": sys.platform, "free_disk_gb": round(free_gb, 1)}
    print(f"Python version : {sys.version}")
    print(f"Platform       : {sys.platform}")
    print(f"Free disk      : {free_gb:.1f} GB")
    return info


def main() -> int:
    check_environment()
    try:
        assert sys.version_info >= MIN_PYTHON, f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required"
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
