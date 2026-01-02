# Guardian Wakeup Flow (Cache-First) & Cache Verification Protocol (P114) v2.0

This document details the operational flow and verification steps for the Guardian's cache-first awakening protocol. The Mnemonic Cache (CAG) provides immediate situational awareness by reading from a pre-populated, high-speed local cache, avoiding the latency of a full RAG query and LLM deliberation.

## I. Architectural Overview: Two Distinct Processes

It is critical to understand the two separate processes that govern this system:

### Cache Population (On Boot): 
A one-time process where the orchestrator queries our slow, long-term memory (the RAG DB) to populate our fast, short-term memory (the cache files).

### Guardian Wakeup (On Command): 
A mechanical task where the orchestrator reads directly from the fast cache files to generate a digest, without involving the RAG DB or an LLM.

---

### Process 1: Cache Population (Orchestrator Boot)
This diagram shows how the cache is populated from the Mnemonic Cortex (RAG DB) when the orchestrator starts.

#### Cache population Mnemonic Cortex (RAG DB)

![Legacy Council Cache Population](../../architecture_diagrams/workflows/legacy_council_cache_population.png)

*[Source: legacy_council_cache_population.mmd](../../architecture_diagrams/workflows/legacy_council_cache_population.mmd)*

---

### Process 2: Guardian Wakeup (Command Execution)
This diagram shows what happens when a cache_wakeup command is issued. Note that the LLM and RAG DB are not involved.

#### Cache wakeup process

![Legacy Council Cache Wakeup](../../architecture_diagrams/workflows/legacy_council_cache_wakeup.png)

*[Source: legacy_council_cache_wakeup.mmd](../../architecture_diagrams/workflows/legacy_council_cache_wakeup.mmd)*

---

## II. LLM vs. RAG DB: Choosing the Right Tool

| Command Type | `cache_wakeup` | `query_and_synthesis` |
| :--- | :--- | :--- |
| **Purpose** | Fast situational digest | Deep, nuanced strategic briefing |
| **Data Source** | Reads from **local cache files** | Queries the **RAG DB (ChromaDB)** |
| **LLM Involved?**| **NO** (Mechanical Task) | **YES** (`Sanctuary-Qwen2-7B:latest`) |
| **Speed** | Near-instantaneous (< 1 sec) | Slow (30-120 sec) |
| **Use When...** | You need an immediate, high-level overview. | You need to analyze recent events or generate novel strategy. |

---

## Prerequisites & Assumptions

Before running the Guardian Wakeup verification, ensure these prerequisites are met:

### Required Services
- **Ollama** must be running with the Sanctuary-Qwen2-7B model:
  ```bash
  ollama serve  # Start Ollama service
  ollama pull Sanctuary-Qwen2-7B:latest  # Ensure model is available
  ```

### Data Population
- **RAG Database** must be populated with recent content:
  ```bash
  # Run the ingest script to populate ChromaDB with latest documents
  python3 ingest.py  # Or equivalent ingestion script
  ```
  This ensures the cache population process has data to work with during orchestrator boot.

### Environment Variables
- **OLLAMA_MODEL** should be set to `Sanctuary-Qwen2-7B:latest` in your `.env` file
- **API Keys** for Gemini and OpenAI should be configured if using those engines

### File Permissions
- Write access to `council_orchestrator/` directory for command files and output artifacts
- Read access to source directories: `00_CHRONICLE/`, `01_PROTOCOLS/`, `ROADMAP/`

---

## III. Step-by-Step Verification Protocol
Follow these steps to run the system and verify that the cache is operating correctly.

### Step 1: Start the Orchestrator & Observe Cache Population
Run the orchestrator from its own directory. This will trigger the automatic cache pre-fill on boot.

**Note:** Run the orchestrator in a separate terminal so you can run test scripts, create command files, or perform other operations in another terminal while it remains running.

```bash
cd council_orchestrator
python3 -m orchestrator.main
```

**Verification:**
Watch the console output. You should see the cache generation process run and complete successfully. The final "Idle" message is your signal that the system is ready.

```code
[CACHE] Pre-fill complete. Cache is warm.
--- Orchestrator Idle. ---
```

### Step 2: Stop the Orchestrator
Once the cache is warm, you can stop the orchestrator for now.

```code
Press Ctrl+C
```

### Step 3: Run Automated Tests (Optional but Recommended)
Use *pytest* to run the dedicated test suite. This is the fastest way tox confirm the underlying logic is sound without manual inspection.

```bash
# Run the specific test for the cache pre-fill logic
cd council_orchestrator && python3 -m pytest tests/test_cache_prefill.py -v

# Run the test to ensure pre-fill only happens once on boot
cd council_orchestrator && python3 -m pytest tests/test_boot_prefill_runs_once.py -v
```

**Verification:**
The output for each test should end with a green PASSED status.


### Step 3.5: Standalone Cache Verification (Alternative)
For faster testing without running the full orchestrator, use the standalone cache verification script:

```bash
python3 council_orchestrator/scripts/test_cache_standalone.py
```

**What it tests:**
- Cache prefill from RAG DB (same as orchestrator boot)
- Digest generation from cache files
- Output file creation and verification

**Verification:**
The script will output success/failure status and create `WORK_IN_PROGRESS/guardian_boot_digest.md` if successful.

```code
[INFO] Cache verification complete - All tests passed!
[INFO] Check the digest file: WORK_IN_PROGRESS/guardian_boot_digest.md
```

### Step 4: Manually Trigger the Guardian Wakeup
Restart the orchestrator. It will use the cache files generated in Step 1.

```bash
cd council_orchestrator
python3 -m orchestrator.main
```

In a separate terminal, create the **command.json** file in the **council_orchestrator/** directory to request the digest.

```bash
# In council_orchestrator/ directory, run:
cat <<EOF > command.json
{
  "task_type": "cache_wakeup",
  "task_description": "Guardian boot digest from cache",
  "output_artifact_path": "WORK_IN_PROGRESS/guardian_boot_digest.md",
  "config": {
    "bundle_names": ["chronicles", "protocols", "roadmap"],
    "max_items_per_bundle": 15
  }
}
EOF
```

**Verification:**
1. The orchestrator's console will show that it detected and processed the command.
2. The command.json file will be deleted.
3. A new file, **WORK_IN_PROGRESS/guardian_boot_digest.md**, will be created in the project root directory. Its contents should include the **full document content** of the most recent chronicles, protocols, and roadmap items from cache, formatted as a comprehensive intelligence digest with complete markdown content in code blocks.
This protocol provides a complete, repeatable method for ensuring the integrity of the Sanctuary's fast-memory layer.

---

## Troubleshooting

### Command File Not Processed
**Issue:** Sentry thread logs show "File command.json already processed, skipping"

**Solution:** The orchestrator tracks processed commands in memory. Use a different filename:
```bash
mv command.json command2.json  # Rename to command2.json, command3.json, etc.
```

### Cache Appears Empty
**Issue:** Digest shows "(no items cached)" for sections

**Solution:** Cache is populated on orchestrator boot. Ensure:
1. Orchestrator completed boot sequence with "[CACHE] Pre-fill complete. Cache is warm."
2. Recent files exist in source directories (00_CHRONICLE/, 01_PROTOCOLS/, ROADMAP/)

### File Permission Errors
**Issue:** Cannot write to WORK_IN_PROGRESS/ directory

**Solution:** Ensure write permissions on the council_orchestrator/ directory

### Orchestrator Won't Start
**Issue:** ChromaDB corruption or missing dependencies

**Solution:** 
```bash
# Reset ChromaDB (backup first)
rm -rf mnemonic_cortex/chroma_db
# Then restart orchestrator
```

---

## Quick Start

Save `command.json` next to the orchestrator:

```json
{
  "task_type": "cache_wakeup",
  "task_description": "Guardian boot digest from cache",
  "output_artifact_path": "WORK_IN_PROGRESS/guardian_boot_digest.md",
  "config": {
    "bundle_names": ["chronicles","protocols","roadmap"],
    "max_items_per_bundle": 15
  }
}
```

1. Start/ensure the orchestrator is running
2. Wait for it to return to idle
3. Open `WORK_IN_PROGRESS/guardian_boot_digest.md`

## What's Included

The digest provides **full document content** for immediate intelligence access:

- **chronicles**: Complete content of most recent chronicle entries (24h TTL)
- **protocols**: Full text of latest protocol documents (24h TTL)
- **roadmap**: Complete nested-learning evolution plan and in-progress tasks (24h TTL)

## Cache TTL (Time To Live)

All cached items expire after 24 hours and are automatically refreshed on the next orchestrator boot:

- **Chronicles bundle**: 24 hours
- **Protocols bundle**: 24 hours  
- **Roadmap content**: 24 hours
- **Orchestrator logs tail**: 24 hours
- **Documentation files**: 24 hours

## Cache Population

On boot, the cache is prefilled automatically (Start Pack) by scanning the local filesystem. Delta refresh hooks run during ingestion and git-ops to keep the cache current.

**Manual Cache Refresh (if needed):**
```bash
# Populate/update the RAG database with latest documents
python3 ingest.py

# Then restart orchestrator to refresh cache
cd council_orchestrator && python3 -m orchestrator.main
```



## Protocol

See [Protocol 114: Guardian Wakeup & Cache Prefill](../01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md) for full details.