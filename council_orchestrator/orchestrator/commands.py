# council_orchestrator/orchestrator/commands.py
# Command parsing and validation utilities

import json
from typing import Dict, Any, Optional
from datetime import datetime
from .memory.cache import CacheManager

def determine_command_type(command: Dict[str, Any]) -> str:
    """Determine the type of command based on its structure."""
    # Check for specific task_type values first
    task_type = command.get("task_type")
    if task_type == "cache_wakeup":
        return "CACHE_WAKEUP"
    elif task_type == "cache_request":
        return "CACHE_REQUEST"
    elif task_type == "query_and_synthesis":
        return "QUERY_AND_SYNTHESIS"
    elif task_type == "cognitive_task":
        return "COGNITIVE_TASK"
    
    # Then check for generic structure patterns
    if "entry_content" in command and "output_artifact_path" in command:
        return "MECHANICAL_WRITE"
    elif "git_operations" in command:
        return "MECHANICAL_GIT"
    elif "task_type" in command and "task_description" in command and "output_artifact_path" in command:
        return "CACHE_WAKEUP"  # Generic cache wakeup pattern
    elif "task_description" in command and not command.get("task_type"):
        return "COGNITIVE_TASK"
    elif "development_cycle" in command:
        return "DEVELOPMENT_CYCLE"
    else:
        return "UNKNOWN"

def validate_command(command: Dict[str, Any]) -> tuple[bool, str]:
    """Validate that a command has the required fields for its type."""
    command_type = determine_command_type(command)

    if command_type == "MECHANICAL_WRITE":
        required_fields = ["entry_content", "output_artifact_path"]
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field '{field}' for MECHANICAL_WRITE command"

    elif command_type == "MECHANICAL_GIT":
        if "git_operations" not in command:
            return False, "Missing 'git_operations' field for MECHANICAL_GIT command"

    elif command_type == "CACHE_WAKEUP":
        required_fields = ["task_type", "task_description", "output_artifact_path"]
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field '{field}' for CACHE_WAKEUP command"
        if command.get("task_type") != "cache_wakeup":
            return False, "task_type must be 'cache_wakeup' for CACHE_WAKEUP command"

    elif command_type == "CACHE_REQUEST":
        required_fields = ["task_type", "task_description", "output_artifact_path", "cache_request"]
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field '{field}' for CACHE_REQUEST command"
        if command.get("task_type") != "cache_request":
            return False, "task_type must be 'cache_request' for CACHE_REQUEST command"

    elif command_type == "QUERY_AND_SYNTHESIS":
        required_fields = ["task_type", "task_description", "output_artifact_path"]
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field '{field}' for QUERY_AND_SYNTHESIS command"
        if command.get("task_type") != "query_and_synthesis":
            return False, "task_type must be 'query_and_synthesis' for QUERY_AND_SYNTHESIS command"

    elif command_type == "COGNITIVE_TASK":
        if "task_description" not in command:
            return False, "Missing 'task_description' field for COGNITIVE_TASK command"

    elif command_type == "DEVELOPMENT_CYCLE":
        if "development_cycle" not in command:
            return False, "Missing 'development_cycle' field for DEVELOPMENT_CYCLE command"

    elif command_type == "UNKNOWN":
        return False, "Unknown or invalid command type"

    return True, "Command is valid"

def parse_command_from_json(json_content: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Parse a command from JSON string and validate it."""
    try:
        command = json.loads(json_content)
        is_valid, error_msg = validate_command(command)
        if is_valid:
            return command, determine_command_type(command)
        else:
            return None, f"INVALID_JSON: {error_msg}"
    except json.JSONDecodeError as e:
        return None, f"INVALID_JSON: {str(e)}"


def handle_cache_request(command: Dict[str, Any]) -> str:
    """Handle a cache_request command and return verification artifact markdown."""
    cache_request = command["cache_request"]
    policy = cache_request.get("policy", {"refresh_if_stale": True, "strict": False})

    # Refresh if requested
    if policy.get("refresh_if_stale", True):
        if "bundle" in cache_request and cache_request["bundle"] == "guardian_start_pack":
            CacheManager.prefill_guardian_start_pack()

    # Get cache entries
    entries = []
    if "bundle" in cache_request:
        if cache_request["bundle"] == "guardian_start_pack":
            entries = CacheManager.get_bundle("guardian_start_pack")
    elif "keys" in cache_request:
        entries = CacheManager.get_keys(cache_request["keys"])

    # Generate verification report
    timestamp = datetime.now().isoformat()
    bundle_name = cache_request.get("bundle", "custom")
    refresh_policy = "refresh_if_stale=true" if policy.get("refresh_if_stale", True) else "refresh_if_stale=false"
    strict_policy = "strict=true" if policy.get("strict", False) else "strict=false"

    # Calculate summary stats
    total_items = len(entries)
    missing = sum(1 for e in entries if e.get("missing", False))
    expired = sum(1 for e in entries if e.get("expired", False))
    refreshed = sum(1 for e in entries if e.get("refreshed", False))

    # Build markdown
    lines = [
        "# Guardian Wakeup Cache Check (v9.4)",
        "",
        f"**When:** {timestamp}",
        f"**Command:** cache_request → bundle={bundle_name}, {refresh_policy}, {strict_policy}",
        "",
        "## Summary",
        f"- Items: {total_items}",
        f"- Missing: {missing}",
        f"- Expired: {expired}",
        f"- Refreshed: {refreshed}",
        "- TTL Policy: docs=24h, configs=6h, logs=10m",
        "",
        "## Items",
        "| key | ttl_remaining | size | sha256[:10] | source | last_updated |",
        "|-----|---------------|------|-------------|--------|--------------|"
    ]

    for entry in entries:
        key = entry.get("key", "unknown")
        ttl_remaining = entry.get("ttl_remaining", "N/A")
        size = entry.get("size", "N/A")
        sha256_prefix = entry.get("sha256_prefix", "N/A")[:10]
        source = entry.get("source", "N/A")
        last_updated = entry.get("last_updated", "N/A")
        lines.append(f"| {key} | {ttl_remaining} | {size} | {sha256_prefix} | {source} | {last_updated} |")

    if missing > 0 or expired > 0:
        lines.extend([
            "",
            "## Notes",
            f"- Missing items: {missing}",
            f"- Expired items: {expired}"
        ])
        if policy.get("strict", False):
            lines.append("- Strict mode enabled: command will fail due to missing/expired items")

    return "\n".join(lines)


def handle_cache_wakeup(command: Dict[str, Any]) -> str:
    """Handle a cache_wakeup command and return Guardian boot digest."""
    from .memory.cache import CacheManager
    import time
    from datetime import datetime

    # Load config with defaults
    config = command.get("config", {})
    bundle_names = config.get("bundle_names", ["chronicles", "protocols", "roadmap"])
    max_items = int(config.get("max_items_per_bundle", 10))

    # Fetch from cache
    start_time = time.time()
    cm = CacheManager()
    result = cm.fetch_guardian_start_pack(bundles=bundle_names, limit=max_items)
    time_saved_ms = int((time.time() - start_time) * 1000)

    # Add timing info
    result["time_saved_ms"] = time_saved_ms
    result["generated_at"] = datetime.now().isoformat()

    # Render digest
    return render_guardian_boot_digest(result)


def render_guardian_boot_digest(result: Dict[str, Any]) -> str:
    """
    Render Guardian boot digest from cache result.

    result format:
    {
      "bundles": {
        "chronicles": [{"title": "...", "path": "...", "updated_at": "..."}],
        "protocols": [{...}],
        "roadmap": [{...}]
      },
      "generated_at": "ISO8601",
      "time_saved_ms": 1234
    }
    """
    lines = [
        "# Guardian Boot Digest (Cache)",
        f"_Generated: {result.get('generated_at','')}_",
        ""
    ]

    bundles = result.get("bundles", {})
    for bundle_name, items in bundles.items():
        lines.extend([
            f"## {bundle_name.capitalize()}",
            ""
        ])

        if not items:
            lines.append("_No cached items available_")
        else:
            for item in items:
                title = item.get('title', item.get('name', '(untitled)'))
                path = item.get('path', '')
                updated_at = item.get('updated_at', item.get('mtime', ''))
                if isinstance(updated_at, (int, float)):
                    # Convert timestamp to readable format
                    from datetime import datetime
                    updated_at = datetime.fromtimestamp(updated_at).strftime('%Y-%m-%d %H:%M')

                lines.append(f"- **{title}** — `{path}`  _(updated {updated_at})_")

        lines.append("")

    lines.append(f"\n_Time saved (cache): ~{result.get('time_saved_ms',0)}ms_")
    return "\n".join(lines)