# council_orchestrator/orchestrator/sentry.py
# Command file watcher thread

import os
import sys
import time
import json
import logging
from pathlib import Path
from queue import Queue
from .commands import determine_command_type, parse_command_from_json

class CommandSentry:
    """Watches for command*.json files and queues them for processing."""

    def __init__(self, command_queue: Queue, logger: logging.Logger):
        self.command_queue = command_queue
        self.logger = logger

    def watch_for_commands_thread(self):
        """This function runs in a separate thread and watches for command*.json files only."""
        command_dir = Path(__file__).parent.parent  # council_orchestrator directory
        processed_commands = set()  # Track processed command files

        print(f"[SENTRY THREAD] Started monitoring directory: {command_dir}")
        print(f"[SENTRY THREAD] Directory exists: {command_dir.exists()}")
        print(f"[SENTRY THREAD] Directory is readable: {os.access(command_dir, os.R_OK)}")
        print(f"[SENTRY THREAD] DEBUG: Entering main monitoring loop")
        while True:
            try:
                # V5.0 MANDATE 1: Only process files explicitly named command*.json
                # This prevents the rogue sentry from ingesting config files, state files, etc.
                # Updated to match any .json file containing "command" in the name
                found_files = [f for f in command_dir.glob("*.json") if "command" in f.name.lower()]
                print(f"[SENTRY THREAD] DEBUG: Scanning for command*.json files in {command_dir}")
                print(f"[SENTRY THREAD] DEBUG: All .json files in directory: {list(command_dir.glob('*.json'))}")
                if found_files:
                    print(f"[SENTRY THREAD] Found {len(found_files)} command file(s): {[f.name for f in found_files]}")
                else:
                    print(f"[SENTRY THREAD] DEBUG: No command*.json files found this scan")

                for json_file in found_files:
                    print(f"[SENTRY THREAD] DEBUG: Processing file: {json_file.name}")
                    print(f"[SENTRY THREAD] DEBUG: File path: {json_file.absolute()}")
                    print(f"[SENTRY THREAD] DEBUG: File exists: {json_file.exists()}")
                    print(f"[SENTRY THREAD] DEBUG: File size: {json_file.stat().st_size if json_file.exists() else 'N/A'} bytes")
                    print(f"[SENTRY THREAD] DEBUG: File is readable: {os.access(json_file, os.R_OK) if json_file.exists() else 'N/A'}")

                    if json_file.name in processed_commands:
                        print(f"[SENTRY THREAD] DEBUG: File {json_file.name} already processed, skipping")
                        continue

                    processing_start = time.time()
                    print(f"[SENTRY THREAD] DEBUG: Starting processing of {json_file.name} at {time.strftime('%H:%M:%S', time.localtime(processing_start))}")
                    # Determine command type for logging
                    command_type = "UNKNOWN"
                    try:
                        command, parsed_type = parse_command_from_json(json_file.read_text())
                        command_type = parsed_type
                    except:
                        command_type = "INVALID_JSON"

                    print(f"[SENTRY THREAD] Processing command file: {json_file.name} (path: {json_file.absolute()})")
                    self.logger.info(f"COMMAND_PROCESSING_START - File: {json_file.name}, Path: {json_file.absolute()}, Type: {command_type}, Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_start))}")

                    try:
                        # Wait for file to be fully written (check size is stable)
                        initial_size = json_file.stat().st_size
                        print(f"[SENTRY THREAD] DEBUG: Initial file size: {initial_size} bytes")
                        time.sleep(0.1)  # Brief pause to allow writing to complete
                        current_size = json_file.stat().st_size
                        print(f"[SENTRY THREAD] DEBUG: Current file size after pause: {current_size} bytes")
                        if json_file.stat().st_size == initial_size and initial_size > 0:
                            print(f"[SENTRY THREAD] DEBUG: File size stable and > 0, attempting to read JSON")
                            command = json.loads(json_file.read_text())
                            print(f"[SENTRY THREAD] DEBUG: JSON parsed successfully")
                            task_desc = command.get('task_description', 'No description')
                            print(f"[SENTRY THREAD] Loaded command: {task_desc[:50]}...")
                            self.logger.info(f"COMMAND_LOADED - File: {json_file.name}, Task: {task_desc[:100]}..., Config: {command.get('config', {})}")

                            # Put the command onto the thread queue for the main loop to process
                            self.command_queue.put(command)
                            processed_commands.add(json_file.name)
                            json_file.unlink() # Consume the file

                            processing_end = time.time()
                            processing_duration = processing_end - processing_start
                            print(f"[SENTRY THREAD] Command processed and file deleted: {json_file.name} (duration: {processing_duration:.2f}s)")
                            self.logger.info(f"COMMAND_PROCESSING_COMPLETE - File: {json_file.name}, Duration: {processing_duration:.2f}s, End_Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_end))}")
                        else:
                            print(f"[SENTRY THREAD] File appears incomplete (initial: {initial_size}, current: {current_size}), will retry...")
                    except Exception as e:
                        processing_end = time.time()
                        processing_duration = processing_end - processing_start
                        print(f"[SENTRY THREAD ERROR] Could not process command file {json_file.name}: {e}", file=sys.stderr)
                        print(f"[SENTRY THREAD ERROR] Exception type: {type(e).__name__}", file=sys.stderr)
                        import traceback
                        print(f"[SENTRY THREAD ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
                        self.logger.error(f"COMMAND_PROCESSING_FAILED - File: {json_file.name}, Error: {str(e)}, Duration: {processing_duration:.2f}s, End_Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_end))}")
                print(f"[SENTRY THREAD] DEBUG: Sleeping for 1 second before next scan...")
                time.sleep(1) # Check every second
            except Exception as e:
                print(f"[SENTRY THREAD ERROR] Critical error in monitoring loop: {e}", file=sys.stderr)
                self.logger.error(f"SENTRY_THREAD_CRITICAL_ERROR - Error: {str(e)}, Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(1)  # Continue monitoring despite errors