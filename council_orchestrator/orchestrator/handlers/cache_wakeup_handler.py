# council_orchestrator/orchestrator/handlers/cache_wakeup_handler.py

import json
from pathlib import Path
from datetime import datetime

# NOTE: This is a synchronous, mechanical function. It will be run in an executor by the main async loop.
def handle_cache_wakeup(command: dict, orchestrator_instance):
    """
    Handles the 'cache_wakeup' mechanical task.
    Reads pre-populated JSON bundles from the cache and renders a markdown digest.
    This function DOES NOT invoke any LLM or the RAG DB. It is a pure file I/O operation.
    """
    project_root = orchestrator_instance.project_root
    logger = orchestrator_instance.logger
    
    try:
        output_path_str = command["output_artifact_path"]
        output_path = project_root / output_path_str
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = command.get("config", {})
        bundle_names = config.get("bundle_names", ["chronicles", "protocols", "roadmap"])
        max_items = config.get("max_items_per_bundle", 15)
        
        cache_dir = project_root / "mnemonic_cortex" / "cache"
        digest_content = [f"# Guardian Boot Digest\n\nGenerated On: {datetime.utcnow().isoformat()}Z\n"]

        logger.info(f"[CACHE WAKEUP] Reading bundles from: {cache_dir}")

        for bundle_name in bundle_names:
            bundle_file = cache_dir / f"{bundle_name}_bundle.json"
            digest_content.append(f"\n---\n\n## CACHE BUNDLE: {bundle_name.upper()}\n\n")
            
            if not bundle_file.exists():
                digest_content.append("`(no items cached)`\n")
                logger.warning(f"[CACHE WAKEUP] Cache bundle not found: {bundle_file}")
                continue

            try:
                with open(bundle_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data:
                    digest_content.append("`(no items cached)`\n")
                    continue

                for i, item in enumerate(data[:max_items]):
                    source = item.get("metadata", {}).get("source_file", "Unknown Source")
                    content = item.get("page_content", "No content available.")
                    digest_content.append(f"### Item {i+1}: `{source}`\n\n```markdown\n{content}\n```\n\n")
                
                logger.info(f"[CACHE WAKEUP] Successfully processed {len(data)} items from '{bundle_name}' bundle.")

            except json.JSONDecodeError:
                digest_content.append("`(error decoding cache file)`\n")
                logger.error(f"[CACHE WAKEUP] Failed to decode JSON from {bundle_file}")
            except Exception as e:
                digest_content.append(f"`(error processing bundle: {e})`\n")
                logger.error(f"[CACHE WAKEUP] Unexpected error processing bundle {bundle_file}: {e}")

        final_digest = "".join(digest_content)
        output_path.write_text(final_digest, encoding='utf-8')
        logger.info(f"[CACHE WAKEUP] Mechanical Success: Digest written to {output_path}")

    except Exception as e:
        logger.error(f"[CACHE WAKEUP] Mechanical Failure: The cache wakeup operation failed critically: {e}")