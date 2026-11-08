#!/usr/bin/env python3
"""
sync_soul_dataset.py — Hugging Face Hub Dataset Synchronizer CLI
==============================================================
Purpose:
    Coordinates synchronization of the local huggingface/ dataset workspace
    to the remote Hugging Face repository (richfrem/Project_Sanctuary_Soul).
Layer: Hub Transport / Remote Persistence (delegates to managing-huggingface-assets)

Key Inputs:
    Files: huggingface/data/soul_traces.jsonl, huggingface/metadata/manifest.json, README.md
    Credentials: HUGGING_FACE_TOKEN (~/.zshrc), HUGGING_FACE_USERNAME=richfrem
    Skill Dependency: managing-huggingface-assets (global APM skill)

Key Outputs:
    Hub Dataset: https://huggingface.co/datasets/richfrem/Project_Sanctuary_Soul
    Immutable Commit: Pushed training split, dataset card, and cryptographic seal

Usage Examples:
    python3 scripts/sync_soul_dataset.py --check-readiness  # Validate local files & HF skill
    python3 scripts/sync_soul_dataset.py --dry-run          # Preview upload without remote write
    python3 scripts/sync_soul_dataset.py --message "..."    # Execute upload to Hugging Face Hub

Key Functions:
    find_managing_hf_skill(), validate_local_dataset(), execute_sync(), main()
"""

import os
import sys
import json
import hashlib
import time
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Global APM Skill Discovery
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soul_coverage as SC  # noqa: E402  (coverage manifest, check and receipts)


# ---------------------------------------------------------------------------
# Runtime preflight: .env loading and interpreter selection
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[6]  # plugins/sanctuary-core/.apm/skills/<skill>/scripts


def load_repo_env(env_file: Path = REPO_ROOT / ".env") -> list:
    """Load KEY=VALUE lines from the repo .env into os.environ without overriding set values.

    Standard library only: the shared hf_config only reads .env when python-dotenv is
    installed, and silently skips it otherwise. Returns the keys it set.
    """
    loaded = []
    if not env_file.is_file():
        return loaded
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def reexec_in_repo_venv() -> None:
    """If huggingface_hub is missing here, re-run this script under the repo .venv (once)."""
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists() and not os.environ.get("SANCTUARY_SYNC_REEXEC"):
        if Path(sys.executable).resolve() != venv_python.resolve():
            sys.stderr.write(f"huggingface_hub not importable in {sys.executable}; re-running under {venv_python}\n")
            os.environ["SANCTUARY_SYNC_REEXEC"] = "1"
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.stderr.write("huggingface_hub is not installed. Install the dev requirements into .venv: "
                     ".venv/bin/pip install -r requirements-dev.txt\n")


# Discover the installed location of the global managing-huggingface-assets skill.
def find_managing_hf_skill() -> Optional[Path]:
    """Locate the global APM skill managing-huggingface-assets."""
    env_path = os.getenv("MANAGING_HF_SKILL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Standard global APM skill directories: Claude, Copilot/Codex, Antigravity, Gemini
    home = Path.home()
    for base in (home / ".claude" / "skills", home / ".agents" / "skills",
                 home / ".gemini" / "antigravity-cli" / "skills", home / ".gemini" / "skills"):
        scripts = base / "managing-huggingface-assets" / "scripts"
        if scripts.exists():
            return scripts

    return None


# ---------------------------------------------------------------------------
# Local Dataset Validation & Upload Execution
# ---------------------------------------------------------------------------

# Verify that the local dataset workspace contains valid data files and manifests.
def validate_local_dataset(dataset_dir: Path) -> Tuple[bool, str]:
    """Check that the local dataset workspace contains valid artifacts."""
    if not dataset_dir.exists():
        return False, f"Dataset directory {dataset_dir} does not exist"

    data_file = dataset_dir / "data" / "soul_traces.jsonl"
    manifest_file = dataset_dir / "metadata" / "manifest.json"
    readme_file = dataset_dir / "README.md"

    if not data_file.exists() or data_file.stat().st_size == 0:
        return False, f"Missing or empty dataset file: {data_file}"

    if not manifest_file.exists():
        return False, f"Missing manifest file: {manifest_file}"

    if not readme_file.exists():
        return False, f"Missing dataset card README: {readme_file}"

    return True, "Dataset structure valid"


def verify_remote(repo_id, local_dataset, local_manifest, revision_hint=None, fetch=None):
    """Confirm that what is on the Hub equals what was exported. `fetch(repo_id, filename, revision)` -> bytes is injectable for tests.
    Returns dict(verified, revision, remote_dataset_sha256, remote_manifest_sha256, problems)."""
    problems = []
    try:
        if fetch is None:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            revision = api.dataset_info(repo_id).sha
            def fetch(rid, filename, rev):
                return Path(hf_hub_download(rid, filename, repo_type="dataset", revision=rev, force_download=True)).read_bytes()
        else:
            revision = revision_hint
        remote_ds = hashlib.sha256(fetch(repo_id, "data/soul_traces.jsonl", revision)).hexdigest()
        remote_mf = hashlib.sha256(fetch(repo_id, "metadata/coverage_manifest.json", revision)).hexdigest()
    except Exception as e:  # noqa: BLE001
        return {"verified": False, "revision": revision_hint, "problems": [f"remote verification failed: {type(e).__name__}: {e}"]}
    lds = hashlib.sha256(Path(local_dataset).read_bytes()).hexdigest()
    lmf = hashlib.sha256(Path(local_manifest).read_bytes()).hexdigest()
    if remote_ds != lds:
        problems.append("remote soul_traces.jsonl differs from the local export")
    if remote_mf != lmf:
        problems.append("remote coverage_manifest.json differs from the local one")
    return {"verified": not problems, "revision": revision, "remote_dataset_sha256": remote_ds, "remote_manifest_sha256": remote_mf,
            "local_dataset_sha256": lds, "local_manifest_sha256": lmf, "problems": problems}


def write_receipt(repo_root, repo_id, result, coverage_ok):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repo": repo_id, "coverage_required_ok": coverage_ok, **result}
    p = Path(repo_root) / SC.RECEIPTS
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


# Delegate the folder upload to managing-huggingface-assets with exponential backoff.
async def execute_sync(dataset_dir: Path, hf_scripts_dir: Path, commit_msg: str, dry_run: bool = False) -> bool:
    """Execute upload via managing-huggingface-assets."""
    sys.path.insert(0, str(hf_scripts_dir))
    try:
        from hf_config import get_hf_config
        from hf_upload import upload_folder, ensure_dataset_structure, ensure_dataset_card
    except ImportError as e:
        sys.stderr.write(f"Failed to import from managing-huggingface-assets: {e}\n")
        return False

    config = get_hf_config()
    target_repo = config.dataset_repo_id

    print(f"Target Hugging Face Dataset: https://huggingface.co/datasets/{target_repo}")
    print(f"Local Dataset Root:         {dataset_dir}")
    print(f"Commit Message:             {commit_msg}")

    if dry_run:
        print("\n[DRY-RUN] Dataset folder is valid. Sync would delegate to managing-huggingface-assets upload_folder().")
        print(f"[DRY-RUN] Destination: repo_id='{target_repo}', folder='{dataset_dir}'")
        print("[DRY-RUN] Nothing was sent: remote persistence has NOT occurred.")
        return True

    # Real upload execution (requires explicit human approval per Human Gate Law)
    print("\nInitializing dataset card and folder structure on remote...")
    await ensure_dataset_structure(config)
    await ensure_dataset_card(config)

    print(f"Uploading {dataset_dir.name} to Hugging Face...")
    result = await upload_folder(
        folder=dataset_dir,
        remote_prefix="",
        config=config,
        commit_msg=commit_msg
    )

    if result.success:
        print(f"\nUPLOAD FINISHED to {result.repo_url} - verifying what the Hub actually holds...")
        repo_root = dataset_dir.parent
        v = verify_remote(target_repo, dataset_dir / "data" / "soul_traces.jsonl", dataset_dir / "metadata" / "coverage_manifest.json")
        cov_ok = SC.check(repo_root)["coverage"] == "complete"
        rc = write_receipt(repo_root, target_repo, v, cov_ok)
        if v["verified"]:
            print(f"REMOTE VERIFIED at revision {v['revision']}: dataset and coverage manifest hashes match the local export. Receipt appended to {SC.RECEIPTS} (commit it).")
        else:
            print("REMOTE NOT VERIFIED: " + "; ".join(v["problems"]) + f" (receipt appended to {SC.RECEIPTS} with verified=false)")
            return False
        return True
    else:
        print(f"\nERROR: Upload failed: {result.error}")
        return False


# ---------------------------------------------------------------------------
# CLI Command Entrypoint
# ---------------------------------------------------------------------------

# Command-line entrypoint: parse flags, validate readiness, and execute Hub sync.
def main():
    """CLI entrypoint for soul trace dataset readiness and synchronization."""
    reexec_in_repo_venv()
    load_repo_env()
    parser = argparse.ArgumentParser(description="Sanctuary Soul Trace Dataset Sync")
    parser.add_argument(
        "--dataset-dir",
        default="huggingface",
        help="Path to local Hugging Face dataset folder"
    )
    parser.add_argument(
        "--message",
        default="Persist sovereign soul traces and learning snapshot",
        help="Commit message for Hugging Face upload"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and preview sync without uploading"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Upload even though required source coverage is incomplete or stale (the omissions are printed and recorded; not a normal closure)"
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Validate local dataset and managing-huggingface-assets availability"
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    # 1. Validate local dataset
    valid, msg = validate_local_dataset(dataset_dir)
    if not valid:
        sys.stderr.write(f"Dataset Validation Error: {msg}\n")
        sys.exit(1)

    print(f"Dataset Check: OK ({dataset_dir.name})")

    # 1b. Coverage gate: schema validity, required-source coverage and export currency are checked BEFORE anything is sent.
    cov = SC.check(dataset_dir.parent)
    print("Coverage check (four separate facts):")
    rc_cov = SC.report(cov)
    bad = cov["schema"] != "valid" or cov["coverage"] != "complete" or cov["local_export"] != "current"
    if bad and not (args.dry_run or args.check_readiness):
        if not args.allow_partial:
            sys.stderr.write("REFUSING to upload: required coverage is incomplete, the export is stale, or the schema is invalid. Fix it, or pass --allow-partial to publish a declared partial export.\n")
            sys.exit(2)
        print("WARNING: --allow-partial given; publishing a declared PARTIAL export (omissions listed above).")

    # 2. Check managing-huggingface-assets integration
    hf_scripts = find_managing_hf_skill()
    if not hf_scripts:
        sys.stderr.write(
            "ERROR: Global APM skill 'managing-huggingface-assets' not found.\n"
            "Install it via APM or set MANAGING_HF_SKILL_PATH.\n"
        )
        sys.exit(1)

    print(f"HF Skill Integration: OK ({hf_scripts})")

    if args.check_readiness:
        print("Readiness check passed. Skill is ready to persist soul traces.")
        sys.exit(0)

    # 3. Execute sync
    success = asyncio.run(execute_sync(
        dataset_dir=dataset_dir,
        hf_scripts_dir=hf_scripts,
        commit_msg=args.message,
        dry_run=args.dry_run
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
