#!/usr/bin/env python3
"""Normalize Sanctuary registrations after APM installation; preserve other hooks."""
import json
from pathlib import Path


def normalize(data: dict, client: str) -> dict:
    """Flatten only Sanctuary entries and normalize their native event names."""
    hooks = data.get('hooks', {})
    for old, new in [('SessionStart', 'sessionStart'), ('PreCompact', 'preCompact')]:
        if old not in hooks:
            continue
        retained = []
        converted = []
        for entry in hooks[old]:
            children = entry.get('hooks', [entry])
            if not all('sanctuary-core' in item.get('command', '') for item in children):
                retained.append(entry)
                continue
            converted.extend({'command': item['command']} if client == 'cursor' else {'type': 'command', 'command': item['command']} for item in children)
        if retained:
            hooks[old] = retained
        else:
            del hooks[old]
        if converted:
            existing = hooks.setdefault(new, [])
            for entry in converted:
                if entry not in existing:
                    existing.append(entry)
    data['version'] = 1
    return data


def main() -> None:
    """Repair installed registrations beneath the current project root."""
    root = Path.cwd()
    for client, name in [('cursor', '.cursor/hooks.json'), ('copilot', '.github/hooks/sanctuary-core-hooks.json')]:
        path = root / name
        if path.is_file():
            original = path.read_text()
            updated = json.dumps(normalize(json.loads(original), client), indent=2) + '\n'
            if updated != original:
                path.write_text(updated)
                print(f'Normalized {name}')


if __name__ == '__main__':
    main()
