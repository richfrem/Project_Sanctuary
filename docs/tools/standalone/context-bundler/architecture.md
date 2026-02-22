# Context Bundler Architecture & Workflow

**Purpose**: This document explains the architecture, components, and recursive workflow of the Context Bundler tool.

## 1. Core Architecture

The system follows a **Controller-Manager-Worker** pattern with a shared **State** file.

| Component | Role | File | Responsibility |
| :--- | :--- | :--- | :--- |
| **Interface** | Controller | `tools/cli.py` | Routes user commands (`manifest`, `init-context`) to the Manager. No logic here. |
| **Manager** | Orchestrator | `plugins/context-bundler/scripts/bundle.py` | Handles the *Workflow* & *Manifest CRUD*. Inits, Adds, Updates, Removes, and Lists/Queries files. |
| **Bundler** | Worker | `plugins/context-bundler/scripts/bundle.py` | Handles the *Action*. Reads state and compiles the Markdown bundle. |
| **State** | Data Store | `tools/context-bundler/file-manifest.json` | The JSON list of files currently "in scope" for the bundle. |

## 2. Intelligence Sources (Inputs)

The Manager queries these sources to populate the State (`file-manifest.json`):

*   **dependency_map.json**: *The Graph*. Tells the manager *why* file B is needed (e.g. "Form A calls Table B").
*   **Miners (XML/PLL)**: *The Source Truth*. Extract declarative dependencies from raw code.
*   **Base Manifests**: *The Templates*. Provide the mandatory starting point for each analysis type (Form, Lib, etc.).
    *   *Index*: `tools/standalone/context-bundler/base-manifests-index.json` maps Type -> Template.

## 3. The Recursive Context Loop

Bundling is not a one-time event. It is a **Recursive Discovery Process**.

![Recursive Context Loop](../../../diagrams/workflows/context-first-analysis.mmd)

**(See full workflow diagram: [`docs/diagrams/workflows/context-first-analysis.mmd`](../../../diagrams/workflows/context-first-analysis.mmd))**

### Workflow Steps:
1.  **Initialization**: `cli.py init-context` calls `manifest_manager.py init` using the target ID as the **Bundle Title**. It auto-resolves the Artifact Type (e.g., FORM) from the Inventory. It **strictly loads the Base Manifest template** and overwrites `file-manifest.json`. **No dependency analysis happens here.**
2.  **Review**: The Agent reads the generated bundle (Default Context).
3.  **Recursion (The Loop)**:
    *   The Agent analyzes logic.
    *   The Agent uses specific tools (e.g., `/retrieve-dependency-graph`) which query `dependency_map.json` to find missing context.
    *   If a gap is found, the Agent uses `/curate-manifest-add` to update the `file-manifest.json` (add specific files).
    *   The Agent runs `/retrieve-bundle` to regenerate the markdown.
4.  **Completion**: When the bundle contains all necessary context for the task.

## 5. Tool Distribution & Configuration

The tool is self-contained in `tools/standalone/context-bundler/`.

*   **Logic**: `manifest_manager.py`, `bundle.py`
*   **Configuration**:
    *   `base-manifests-index.json`: Maps analysis types (form, lib) to template files.
    *   `base-manifests/*.json`: The actual JSON templates used during `init`.
*   **Documentation**:
    *   `README.md`: Setup & Usage guide.
    *   `architecture.md`: This file (Internal Logic).

> **Note**: When distributing this tool to other agents, bundle the entire `context-bundler/` directory including these config files.
