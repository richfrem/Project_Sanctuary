"""Isolation preflight for the confirm-v3 clean-context Haiku arm: refuses to proceed unless the CLI's own init
and hook events show the experimental context is clean. Supported diagnostics only (stream-json init event)."""
import json, subprocess, sys
from pathlib import Path
import run

HERE = Path(__file__).resolve().parent


def check(write=True):
    cmd = [c for c in run.CLAUDE_CMD]
    i = cmd.index("json"); cmd[i] = "stream-json"
    cmd += ["--verbose"]
    run.NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, input="Reply with exactly: OK", text=True, capture_output=True, timeout=120, cwd=run.NEUTRAL_CWD)
    events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    init = next(e for e in events if e.get("type") == "system" and e.get("subtype") == "init")
    hooks = [e for e in events if e.get("type") == "system" and "hook" in str(e.get("subtype"))]
    mem = init.get("memory_paths", {}).get("auto", "")
    problems = []
    if Path(init["cwd"]) != run.NEUTRAL_CWD.resolve() and Path(init["cwd"]) != run.NEUTRAL_CWD: problems.append(f"cwd {init['cwd']}")
    if hooks: problems.append(f"{len(hooks)} hook events")
    if init.get("slash_commands"): problems.append("slash commands loaded")
    if init.get("skills"): problems.append("skills loaded")
    if init.get("tools"): problems.append("tools enabled")
    if init.get("mcp_servers"): problems.append("mcp servers")
    if "Project_Sanctuary" in mem or (mem and (Path(mem) / "MEMORY.md").exists()): problems.append(f"project memory {mem}")
    if init.get("model") != "claude-haiku-5-5": problems.append(f"model {init.get('model')}")
    report = {"cwd": init["cwd"], "model": init.get("model"), "tools": init.get("tools"), "mcp_servers": init.get("mcp_servers"),
              "slash_commands": len(init.get("slash_commands", [])), "skills": len(init.get("skills", [])), "agents": init.get("agents"),
              "plugins": [p.get("name") for p in init.get("plugins", [])], "hook_events": len(hooks), "memory_path": mem,
              "permissionMode": init.get("permissionMode"), "claude_code_version": init.get("claude_code_version"), "problems": problems}
    if write:
        (HERE / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    if problems:
        raise SystemExit("ISOLATION PREFLIGHT FAILED: " + "; ".join(problems))
    return report


if __name__ == "__main__":
    print(json.dumps(check(), indent=1))
