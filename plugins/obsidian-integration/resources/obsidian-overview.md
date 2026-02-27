# Obsidian Vault Overview: A Developer's Guide

## What Is an Obsidian Vault?

An Obsidian Vault is simply **a folder on your hard drive**. There is no database, no server, no proprietary format. Obsidian reads `.md` files from that folder and builds its graph view, backlinks, and search index from the raw text.

When you designate a folder as a vault, Obsidian creates a hidden `.obsidian/` directory inside it. This is the only thing Obsidian adds to your project.

## The `.obsidian` Folder: The "Brain"

| What It Stores | Details |
|:---------------|:--------|
| Settings | `app.json`, `appearance.json` |
| Workspace layouts | Which panes are open, sidebar state |
| Themes | CSS overrides |
| Community plugins | Downloaded plugin code |

**Git advice**: Add `.obsidian/` to `.gitignore` unless you want your specific UI layout to follow you across devices. The vault content (your `.md` files) is tracked by Git normally.

## Integrating with a Dev Project

Dropping a vault into an existing project directory (like `Project_Sanctuary`) works seamlessly because Obsidian treats **every `.md` file** in the folder tree as a note.

### Documentation as Code
- Your existing `README.md`, ADRs, protocols, and specs are instantly browsable in Obsidian
- No migration needed — the files stay exactly where they are
- Edit in your IDE (VS Code, Cursor, Antigravity) or in Obsidian — same file, two views

### Bidirectional Linking
Obsidian uses `[[Internal Links]]` (wikilinks) to connect files:
- `[[Protocol 128]]` → links to that note
- `[[ADR 099#Rationale]]` → deep-links to a heading
- `![[diagram.png]]` → embeds an image inline

### The Exclusion Logic
Dev projects contain directories Obsidian should NOT index:

| Exclude | Why |
|:--------|:----|
| `node_modules/` | Library files — thousands of irrelevant matches |
| `.git/` | Git internals |
| `.worktrees/` | Spec-Kitty isolation workspaces |
| `venv/` | Python virtual environments |
| `__pycache__/` | Bytecode cache |
| `*.json`, `*.jsonl` | Data files, not knowledge |
| `*_packet.md`, `*_digest.md` | Machine-generated bundles that pollute backlinks |
| `learning_package_snapshot.md` | Giant concatenated bundle |

Configure in: **Settings → Files & Links → Excluded Files** or in `.obsidian/app.json` under `"userIgnoreFilters"`.

## Obsidian vs Standard File Explorer

| Feature | Standard Explorer | Obsidian Vault |
|:--------|:-----------------|:---------------|
| File View | Flat list / Folders | Graph View / Linked Network |
| Search | Filename / Text Match | Unlinked Mentions / Backlinks |
| Visualization | None | Canvas / Diagrams |
| Metadata | File size / Date | YAML Frontmatter / Tags |
| Relationships | Manual navigation | Automatic backlink discovery |

## Obsidian-Specific Markdown Syntax

### Links and References
| Syntax | Purpose |
|:-------|:--------|
| `[[Note Name]]` | Standard internal link |
| `[[Note#Heading]]` | Link to a specific heading |
| `[[Note#^block-id]]` | Link to a specific block |
| `[[Note\|Display Text]]` | Aliased link |
| `![[Note]]` | Transclusion (embed entire note) |
| `![[image.png]]` | Embed image |

### Callouts
```markdown
> [!info] Title
> Content appears in a styled box.
```
Supported types: `info`, `warning`, `error`, `success`, `note`, `tip`, `important`

### Frontmatter (YAML)
```yaml
---
title: My Note
status: active
tags: [obsidian, documentation]
---
```
Used by Obsidian Properties, Dataview plugin, and agent skills for metadata.

## Key Obsidian Features for Agent Integration

### Canvas (Visual Boards)
- `.canvas` files use JSON Canvas Spec 1.0
- Drag `.md` files onto a visual board
- Draw arrows between nodes to map logic flows
- Agents can generate these programmatically

### Bases (Database Views)
- `.base` files use YAML to define database views
- Render as tables, cards, or grids
- Agents can act as database administrators

### Graph View
- Visualizes all `[[wikilink]]` connections
- Shows relationship density and orphaned notes
- Useful for architecture visualization

## The Agent Workflow

```
IDE (VS Code / Antigravity)           Obsidian App
        │                                    │
        │   ┌─────────────────────┐          │
        ├──►│  Same .md files on  │◄─────────┤
        │   │  disk (Git tracked) │          │
        │   └─────────────────────┘          │
        │                                    │
   Code editing,                     Graph view,
   agent scripts,                    Canvas boards,
   CLI operations                    visual navigation
```

Both tools read/write the same files. Changes in one are instantly visible in the other.
