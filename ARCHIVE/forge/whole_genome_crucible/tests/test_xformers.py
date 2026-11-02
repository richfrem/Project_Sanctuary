#============================================
# forge/tests/test_xformers.py
# Purpose: Minimal smoke test to confirm xformers installation and version.
# Role: Diagnostic / Smoke Test
# Used by: Phase 1 of the Forge Pipeline (Memory Optimization)
#============================================

import sys

#============================================
# Function: main
# Purpose: Confirms xformers import and logs the version.
# Args: None
# Returns: None
#============================================
def main() -> None:
    """
    Diagnostic check for xformers.
    """
    try:
        import xformers
        version: str = getattr(xformers, '__version__', 'Unknown')
        print(f"✅ xformers import successful. Version: {version}")
    except ImportError:
        print("❌ Error: xformers not found. Install it for optimized attention if supported.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ xformers check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
