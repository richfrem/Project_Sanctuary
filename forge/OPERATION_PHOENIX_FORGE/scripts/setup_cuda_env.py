#!/usr/bin/env python3
"""
setup_cuda_env.py (moved)

This is a relocated copy of the repo's CUDA environment helper. It was moved
into `forge/OPERATION_PHOENIX_FORGE/scripts/` — the script now detects the
repository root by walking parent directories so it behaves correctly when
located inside the forge subfolder.
"""
from __future__ import annotations
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_repo_root(start: str | Path) -> str:
    p = Path(start).resolve()
    # Walk upwards looking for a .git directory or a marker file (requirements.txt)
    for parent in [p] + list(p.parents):
        if (parent / '.git').exists() or (parent / 'requirements.txt').exists():
            return str(parent)
    # Fallback: two levels up from this script
    return str(p.parents[2]) if len(p.parents) >= 3 else str(p.parents[-1])


# Compute ROOT relative to this file location
THIS_FILE = Path(__file__).resolve()
ROOT = find_repo_root(THIS_FILE)
ML_ENV_SCRIPT = os.path.join(ROOT, 'ML-Env-CUDA13', 'setup_ml_env_wsl.sh')
CUDA_MARKDOWN = os.path.join(ROOT, 'CUDA-ML-ENV-SETUP.md')
LOG_DIR = os.path.join(ROOT, 'ml_env_logs')


def run(cmd, capture=False):
    print(f"> {cmd}")
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    res = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.STDOUT if capture else None)
    return res


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def write_activate_helper(venv_path):
    # write activation helper into the forge scripts directory
    helper_dir = os.path.join(ROOT, 'forge', 'OPERATION_PHOENIX_FORGE', 'scripts')
    os.makedirs(helper_dir, exist_ok=True)
    helper = os.path.join(helper_dir, 'activate_ml_env.sh')
    with open(helper, 'w', newline='\n') as f:
        f.write(f"# Activation helper for ml_env venv\n")
        f.write(f"# Source this from your shell: source {os.path.relpath(helper, ROOT)}\n")
        f.write(f"source {os.path.join(venv_path, 'bin', 'activate')}\n")
    try:
        os.chmod(helper, 0o755)
    except Exception:
        pass
    print(f"Wrote activation helper: {helper}")


def find_torch_pin(req_path):
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('torch==') or line.startswith('torch '):
                    return line.split()[0]
    except Exception:
        return None
    return None


def find_torch_related_pins(req_path):
    pins = {}
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s.startswith('torch==') or s.startswith('torch '):
                    pins['torch'] = s.split()[0]
                elif s.startswith('torchvision==') or s.startswith('torchvision '):
                    pins['torchvision'] = s.split()[0]
                elif s.startswith('torchaudio==') or s.startswith('torchaudio '):
                    pins['torchaudio'] = s.split()[0]
    except Exception:
        return pins
    return pins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', default=os.path.expanduser('~/ml_env'), help='Path to venv')
    parser.add_argument('--requirements', default=os.path.join(ROOT, 'requirements.txt'))
    parser.add_argument('--staged', action='store_true', help='Run staged install (recommended)')
    parser.add_argument('--quick', action='store_true', help='Quick one-step install (pip install -r requirements.txt)')
    parser.add_argument('--regen-tests-only', action='store_true', help='Run ML-Env regen tests only and exit')
    parser.add_argument('--recreate', action='store_true', help='Remove existing venv and recreate')
    parser.add_argument('--pin-tensorflow', help='Pin TensorFlow to a specific version (e.g. 2.20.0)')
    args = parser.parse_args()

    ensure_dir(LOG_DIR)

    print('\nCUDA setup helper — references:')
    print(f" - {ML_ENV_SCRIPT}")
    print(f" - {CUDA_MARKDOWN}\n")

    if args.regen_tests_only:
        if os.path.exists(ML_ENV_SCRIPT):
            print('Running regen tests via ML-Env script...')
            res = run(['bash', ML_ENV_SCRIPT, '--regen-tests-only'])
            sys.exit(res.returncode)
        else:
            print('ML-Env regen script not found; nothing to do.')
            sys.exit(1)

    venv_path = os.path.expanduser(args.venv)

    if os.path.exists(venv_path) and args.recreate:
        print(f'Removing existing venv at {venv_path}')
        shutil.rmtree(venv_path)

    if not os.path.exists(venv_path):
        print(f'Creating venv at {venv_path}')
        run([sys.executable, '-m', 'venv', venv_path])
    else:
        print(f'Using existing venv at {venv_path}')

    venv_python = os.path.join(venv_path, 'bin', 'python')
    if not os.path.exists(venv_python):
        print('ERROR: venv python not found at', venv_python)
        sys.exit(1)

    write_activate_helper(venv_path)

    print('\nUpgrading pip/wheel/setuptools in venv...')
    run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'])

    if args.quick:
        print('\nQuick install: installing requirements.txt directly')
        run([venv_python, '-m', 'pip', 'install', '-r', args.requirements])
        print('\nQuick install finished; run activation with:')
        print(f'source {os.path.join(venv_path, "bin", "activate")}')
        return

    if args.staged:
        print('\nStaged install: installing CUDA PyTorch wheels first')
        pins = find_torch_related_pins(args.requirements)
        if pins.get('torch'):
            pkg_list = [pins.get('torch')]
            if pins.get('torchvision'):
                pkg_list.append(pins.get('torchvision'))
            if pins.get('torchaudio'):
                pkg_list.append(pins.get('torchaudio'))
            print(f"Found torch-related pins: {pkg_list} — installing via PyTorch index")
            run([venv_python, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu126'] + pkg_list)
        else:
            print('No torch pin found in requirements; installing latest torch (no pin)')
            run([venv_python, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu126', 'torch'])

        if args.pin_tensorflow:
            tf_pkg = f'tensorflow=={args.pin_tensorflow}'
        else:
            tf_pkg = 'tensorflow'
        print(f'Installing TensorFlow package: {tf_pkg}')
        run([venv_python, '-m', 'pip', 'install', '--upgrade', tf_pkg])

        core_log = os.path.join(LOG_DIR, 'test_torch_cuda.log')
        print('\nRunning core verification test (test_torch_cuda.py)')
        res = subprocess.run([venv_python, os.path.join(ROOT, 'ML-Env-CUDA13', 'test_torch_cuda.py')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(core_log, 'wb') as f:
            f.write(res.stdout)
        exit_file = os.path.join(LOG_DIR, 'test_torch_cuda.exit')
        with open(exit_file, 'w') as f:
            f.write(str(res.returncode))
        print(f'Core verification exit code: {res.returncode} (log: {core_log})')

        if res.returncode != 0:
            print('Core verification failed. Inspect the log and fix the environment before continuing.')
            print(f'cat {core_log}')
            sys.exit(res.returncode)

        ts = datetime.utcnow().strftime('%Y%m%d%H%M')
        pinned = os.path.join(ROOT, f'pinned-requirements-{ts}.txt')
        print('Core gate passed. Creating pinned requirements snapshot:', pinned)
        fr = subprocess.run([venv_python, '-m', 'pip', 'freeze'], stdout=subprocess.PIPE)
        with open(pinned, 'wb') as f:
            f.write(fr.stdout)

        print('\nInstalling remainder of requirements from', args.requirements)
        run([venv_python, '-m', 'pip', 'install', '-r', args.requirements])

        tests = ['test_pytorch.py', 'test_tensorflow.py', 'test_xformers.py', 'test_llama_cpp.py']
        for t in tests:
            tpath = os.path.join(ROOT, 'ML-Env-CUDA13', t)
            logp = os.path.join(LOG_DIR, t.replace('.py', '.log'))
            exitp = os.path.join(LOG_DIR, t.replace('.py', '.exit'))
            print(f'Running {t} -> {logp}')
            r = subprocess.run([venv_python, tpath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with open(logp, 'wb') as f:
                f.write(r.stdout)
            with open(exitp, 'w') as f:
                f.write(str(r.returncode))

        print('\nStaged install complete. Activate the venv with:')
        print(f'source {os.path.join(venv_path, "bin", "activate")}')
        print('Review logs in', LOG_DIR)
        return

    print('No mode selected. Run with --staged (recommended) or --quick. Use --help for options.')


if __name__ == '__main__':
    main()
