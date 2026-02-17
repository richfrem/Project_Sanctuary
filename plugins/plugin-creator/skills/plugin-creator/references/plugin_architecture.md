# Claude Code Plugin Architecture

Plugins are the fundamental unit of extension for Claude Code. They package skills, agents, and hooks into distributable units.

## Directory Structure

A standard plugin looks like this:

```
my-plugin/
├── .claude-plugin/     # REQUIRED: Plugin metadata
│   └── plugin.json     # Manifest file
├── skills/             # OPTIONAL: Agent skills
│   └── my-skill/       # Individual skill directory
│       └── SKILL.md    # Skill definition
├── agents/             # OPTIONAL: Custom agents (future)
└── hooks/              # OPTIONAL: lifecycle hooks (future)
```

## The Manifest (`plugin.json`)

Located at `.claude-plugin/plugin.json`.

```json
{
  "name": "my-plugin",
  "description": "Description of what the plugin does",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

## Skills

Skills are placed in the `skills/` directory. Each skill must have its own subdirectory containing a `SKILL.md` file.

See the `skill-creator` plugin for details on how to write effective skills.
