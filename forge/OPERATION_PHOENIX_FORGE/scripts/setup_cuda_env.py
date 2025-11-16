#!/usr/bin/env python3
"""
setup_cuda_env.py (v2.2 - All-in-One)

This script is the Foreman of the Forge. It is a single, unified command to
build the complete, CUDA-enabled ML environment (`~/ml_env`).

It now performs a prerequisite check and handles the installation of system
packages (like python3.11-venv) before creating the virtual environment.

*** IMPORTANT ***
This script must be run with `sudo` because it needs to install system packages
using 'apt'. It will intelligently drop privileges to create the user-owned venv.

Example from your project root:
sudo python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged --recreate
"""
from __future__ import annotations
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# --- Global Configuration ---
PYTHON_VERSION = "3.11"

def check_and_install_prerequisites():
    """Checks for and installs system-level dependencies using apt."""
    print("--- Phase 0: Checking System Prerequisites ---")
    
    try:
        subprocess.run(
            ['dpkg-query', '-W', f'python{PYTHON_VERSION}-venv'],
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[INFO] Prerequisite 'python{PYTHON_VERSION}-venv' is already installed. Skipping system setup.")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"[WARN] Prerequisite 'python{PYTHON_VERSION}-venv' not found. Attempting installation...")

    prereq_commands = [
        ['apt-get', 'update', '-y'],
        ['apt-get', 'install', 'software-properties-common', '-y'],
        ['add-apt-repository', 'ppa:deadsnakes/ppa', '-y'],
        ['apt-get', 'update', '-y'],
        ['apt-get', 'install', f'python{PYTHON_VERSION}', f'python{PYTHON_VERSION}-venv', '-y']
    ]
    
    for cmd in prereq_commands:
        print(f"> {' '.join(shlex.quote(c) for c in cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n[FATAL] Prerequisite installation failed: {e}", file=sys.stderr)
            print("Please try running the failed command manually to diagnose the issue.", file=sys.stderr)
            sys.exit(1)
    
    print("[INFO] System prerequisites installed successfully.")


def find_repo_root(start: str | Path) -> str:
    """Walks upwards from a starting path to find the git repository root."""
    p = Path(start).resolve()
    for parent in [p] + list(p.parents):
        if (parent / '.git').exists() or (parent / 'requirements.txt').exists():
            return str(parent)
    return str(Path.cwd())


# --- Global Paths ---
THIS_FILE = Path(__file__).resolve()
ROOT = find_repo_root(THIS_FILE)
LOG_DIR = os.path.join(ROOT, 'forge', 'OPERATION_PHOENIX_FORGE', 'ml_env_logs')


def run_as_user(cmd: list, user: str, venv_python: str | None = None) -> bool:
    """Executes a command as a specific user, dropping sudo privileges."""
    base_cmd = ['sudo', '-u', user]
    if venv_python:
        full_cmd = base_cmd + [venv_python, '-m'] + cmd
    else:
        full_cmd = base_cmd + cmd

    print(f"> {' '.join(shlex.quote(str(c)) for c in full_cmd)}")
    try:
        subprocess.run(full_cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  -> COMMAND FAILED: {e}", file=sys.stderr)
        return False
    return True


def ensure_dir(path: str):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)


def parse_requirements(req_path: str) -> tuple[dict, str | None]:
    """Parses requirements.txt to find PyTorch-related pins and the extra-index-url."""
    pins = {}
    extra_index_url = None
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('--extra-index-url'):
                    extra_index_url = s.split(maxsplit=1)[1]
                elif '==' in s:
                    pkg_name = s.split('==')[0].lower()
                    if pkg_name in ['torch', 'torchvision', 'torchaudio']:
                         pins[pkg_name] = s
    except FileNotFoundError:
        print(f"WARNING: requirements file not found at {req_path}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to parse requirements file: {e}", file=sys.stderr)
    return pins, extra_index_url


def main():
    if os.geteuid() != 0:
        print("[FATAL] This script needs to install system packages.", file=sys.stderr)
        print(f"Please run it with sudo: 'sudo {sys.executable} {' '.join(sys.argv)}'", file=sys.stderr)
        sys.exit(1)
        
    original_user = os.environ.get('SUDO_USER')
    if not original_user:
        print("[FATAL] Could not determine the original user.", file=sys.stderr)
        print("Please ensure you are running this with 'sudo', not as the root user directly.", file=sys.stderr)
        sys.exit(1)
    
    default_venv_path = os.path.join(os.path.expanduser(f'~{original_user}'), 'ml_env')
    
    parser = argparse.ArgumentParser(
        description="The Foreman: Builds the complete ML environment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--venv', default=default_venv_path, help='Path to the virtual environment.')
    parser.add_argument('--requirements', default=os.path.join(ROOT, 'requirements.txt'), help='Path to the requirements blueprint.')
    parser.add_argument('--staged', action='store_true', help='Run staged install (Highly Recommended).')
    parser.add_argument('--recreate', action='store_true', help='Force removal of the existing venv before starting.')
    args = parser.parse_args()

    check_and_install_prerequisites()

    ensure_dir(LOG_DIR)
    venv_path = os.path.expanduser(args.venv)

    if os.path.exists(venv_path):
        if args.recreate:
            print(f'[INFO] Purging existing venv at {venv_path}...')
            shutil.rmtree(venv_path, ignore_errors=True)
        else:
            print(f'[INFO] Using existing venv at {venv_path}. Use --recreate to force a rebuild.')

    if not os.path.exists(venv_path):
        print(f'Creating new venv at {venv_path} for user {original_user}...')
        venv_cmd = [f'python{PYTHON_VERSION}', '-m', 'venv', venv_path]
        if not run_as_user(venv_cmd, user=original_user):
             print("\n[FATAL] Failed to create virtual environment.", file=sys.stderr)
             sys.exit(1)
    
    venv_python = os.path.join(venv_path, 'bin', 'python')
    if not os.path.exists(venv_python):
        print(f'[FATAL] Python executable not found in venv at {venv_python}', file=sys.stderr)
        sys.exit(1)

    if args.staged:
        print('\n--- STAGED INSTALLATION INITIATED ---')

        print('\nStep 1: Upgrading core packaging tools...')
        run_as_user(['pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], user=original_user, venv_python=venv_python)

        pins, extra_index_url = parse_requirements(args.requirements)
        
        if extra_index_url and pins.get('torch'):
            print(f"\nStep 2: Installing pinned PyTorch, xformers, and CUDA packages from {extra_index_url}...")
            torch_packages = [v for k, v in pins.items() if k in ['torch', 'torchvision', 'torchaudio']]
            
            # MODIFICATION: Add xformers to the initial, crucial installation step.
            # This ensures its dependencies are resolved alongside PyTorch correctly.
            torch_packages.append('xformers')
            
            install_cmd = ['pip', 'install'] + torch_packages + ['--index-url', extra_index_url]
            if not run_as_user(install_cmd, user=original_user, venv_python=venv_python):
                print("\n[FATAL] Failed to install PyTorch/xformers packages. The Forge is misaligned.", file=sys.stderr)
                sys.exit(1)
        else:
            print("\n[WARN] Could not find PyTorch pins or --extra-index-url in requirements.txt.", file=sys.stderr)
            print("Skipping explicit Torch install. The subsequent step may fail.", file=sys.stderr)

        print('\nStep 3: Installing all remaining requirements from the blueprint...')
        if not run_as_user(['pip', 'install', '-r', args.requirements], user=original_user, venv_python=venv_python):
            print("\n[FATAL] Failed to install remaining requirements. Check requirements.txt for conflicts.", file=sys.stderr)
            sys.exit(1)
        
        print('\n--- STAGED INSTALLATION COMPLETE ---')
        print('The environment is forged and aligned.')
        print(f'\nTo activate, run: source {os.path.join(venv_path, "bin", "activate")}')

    else:
        print('\n[INFO] No installation mode selected. Run with --staged to build the environment.')


if __name__ == '__main__':
    main()