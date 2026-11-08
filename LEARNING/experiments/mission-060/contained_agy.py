#!/usr/local/bin/python3
"""Executable selected by cli-agent for a contained AGY worker or smoke probe.

Removes the router's permission-bypass flag. The external sandbox profile is the
filesystem enforcement layer for the entire child process tree.
"""

import os
import pathlib
import shutil
import sys

from worker_containment import prepare


def main():
    root = pathlib.Path(os.environ["STUDY060_WORKER_ROOT"]).resolve()
    if not root.is_relative_to(pathlib.Path("/private/tmp/study060/work")):
        raise SystemExit("contained worker must use the designated temporary work area")
    if pathlib.Path.cwd().resolve() != root:
        raise SystemExit("contained worker cwd must equal the declared worker root")
    executable = shutil.which("agy")
    if not executable or pathlib.Path(executable).resolve() == pathlib.Path(__file__).resolve():
        raise SystemExit("real AGY executable unavailable")
    profile, runtime = prepare(root)
    os.environ["TMPDIR"] = str(runtime) + "/"
    args = [arg for arg in sys.argv[1:] if arg != "--dangerously-skip-permissions"]
    if "--version" not in args:
        args = ["--mode", "auto", *args]
    os.execv("/usr/bin/sandbox-exec", ["sandbox-exec", "-f", str(profile), executable, *args])


if __name__ == "__main__":
    main()
