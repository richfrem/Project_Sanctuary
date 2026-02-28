---
name: obsidian-markdown-mastery
description: "Core markdown syntax skill for Sanctuary. Enforces the strict parsing and authoring of Obsidian's proprietary syntax (Wikilinks, Blocks, Headings, Aliases, Embeds, and Callouts) ensuring compatibility with the Vault graph."
---

# Obsidian Markdown Mastery (Protocol 129 COMPLIANT)

**Status:** Active
**Author:** Sanctuary Guardian
**Domain:** Obsidian Integration

## Core Mandate

The `obsidian-markdown-mastery` skill is responsible for the exact formatting, extraction, and validation of Obsidian-flavoured Markdown. It provides the low-level string manipulation that allows higher-order agents (like the Graph Traverser or JSON Canvas Architect) to safely interpret relational links without breaking the `.md` Vault.

> **CRITICAL ARCHITECTURAL RULE:**
> All Sanctuary data manipulation MUST occur through deterministic Python scripts rather than agent-prompted regex. This skill defines the `obsidian-parser` module that performs these deterministic actions.
> 
> *Agnosticism Enforcement*: This module knows NOTHING about Hugging Face, the RLM, or Project Sanctuary protocols. It only knows how to parse text into valid Obsidian links and block-quotes. The Guardian maintains the specific paths (like `protocol-manager` and `chronicle-manager` injections) via the `SANCTUARY_VAULT_PATH` environment variable.

## Available Commands

### Analyze Markdown Content
Extracts all Obsidian-specific metadata (links, embeds, blocks) from a given markdown file or string.
**Command**: `python plugins/obsidian-integration/obsidian-parser/parser.py analyze --file <path_to_md>`

### Inject Callout
Wraps a target text block in an Obsidian-flavored callout.
**Command**: `python plugins/obsidian-integration/obsidian-parser/parser.py callout --type <type> --title <title> --text <content>`

## The Parsed Syntax (Data Dictionary)

When manipulating strings via this module, the following formats are enforced:

### 1. Linking and Aliasing
*   **Standard Link**: `[[Note Name]]`
*   **Heading Link**: `[[Note Name#Heading Name]]`
*   **Block Link**: `[[Note Name#^block-id]]`
*   **Aliased Link**: `[[Note Name|Display Text]]`

### 2. Transclusion (Embeds)
*   **Standard Embed**: `![[Note Name]]` (Note the leading `!`)
*   *(The parser specifically categorizes these differently so graph mappers know they are transclusions, not semantic links).*

### 3. Callouts
*   **Syntax**:
    ```markdown
    > [!type] Title
    > Content block goes here.
    ```
*   **Supported Types**: `info`, `warning`, `error`, `success`, `note`.

## Configuration Environment Variable
Other tools (such as `protocol-manager` and `chronicle-manager`) rely on the unified `SANCTUARY_VAULT_PATH` environment variable to discover where the root of the Obsidian Vault resides. If missing, it defaults to the project root.
