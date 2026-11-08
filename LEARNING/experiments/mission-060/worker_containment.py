"""macOS filesystem write boundary for the Flash worker process tree.

Network and reads remain available. Client state is a documented write exception;
this is not a general-purpose hostile-agent security boundary.
"""

import json
import pathlib
import os
import tempfile


def profile_text(output, runtime):
    roots = [pathlib.Path(output).resolve(), pathlib.Path(runtime).resolve(),
             (pathlib.Path.home() / ".gemini/antigravity-cli").resolve()]
    return ('(version 1)\n(allow default)\n(deny file-write*)\n'
            + "\n".join('(allow file-write* (subpath ' + json.dumps(str(root)) + '))' for root in roots)
            + '\n(allow file-write-data (literal "/dev/null"))\n'
            + '(allow pseudo-tty)\n'
            + '(allow file-read* file-write* file-ioctl (literal "/dev/ptmx") (regex #"^/dev/ttys[0-9]+$"))\n')


def prepare(root):
    root = pathlib.Path(root).resolve()
    if not root.is_dir():
        raise ValueError("worker root must already exist")
    output = root / "deliverables"
    if output.is_symlink():
        raise ValueError("deliverables must not be a symlink")
    output.mkdir(exist_ok=True)
    runtime = root.parent / (".runtime-" + root.name)
    if runtime.is_symlink():
        raise ValueError("runtime must not be a symlink")
    runtime.mkdir(exist_ok=True, mode=0o700)
    profile = runtime / "worker.sb"
    # Atomic replacement never follows a worker-planted symlink or hard link.
    with tempfile.NamedTemporaryFile("w", dir=runtime, delete=False) as stream:
        stream.write(profile_text(output, runtime))
        temporary = stream.name
    os.replace(temporary, profile)
    return profile, runtime
