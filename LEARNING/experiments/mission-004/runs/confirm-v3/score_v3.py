"""Runs the byte-identical frozen evaluate.py. The frozen file defines load_jsonl after its __main__ block, so it cannot run
as a script; this wrapper imports it unchanged, sets HERE, and calls main(). No scoring logic is changed."""
import sys
from pathlib import Path
import evaluate
evaluate.HERE = Path(__file__).resolve().parent
sys.argv = ["evaluate.py", "--model", "claude"]
evaluate.main()
