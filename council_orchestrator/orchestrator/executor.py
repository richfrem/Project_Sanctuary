import subprocess
import logging
from typing import List, Union, Optional
from pathlib import Path

# Define ProtocolViolationError
class ProtocolViolationError(Exception):
    """Raised when a command violates Protocol 101 mandates."""
    pass

PROHIBITED_COMMANDS = [
    "git pull",
    "git reset",
    "git checkout", 
    "git clean",
    "git revert",
    "git fetch",
    "git merge" # Adding 'git merge' to force use of the dedicated MCP tool
]

def execute_shell_command(
    command: Union[str, List[str]], 
    cwd: Optional[Path] = None, 
    check: bool = True,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Executes a shell command with Protocol 101 Whitelist Enforcement.
    
    Args:
        command: The command string or list of arguments.
        cwd: Current working directory.
        check: Whether to raise CalledProcessError on failure.
        capture_output: Whether to capture stdout/stderr.
        
    Returns:
        subprocess.CompletedProcess object.
        
    Raises:
        ProtocolViolationError: If the command is prohibited.
        subprocess.CalledProcessError: If the command fails and check is True.
    """
    # Normalize command to string for checking
    if isinstance(command, list):
        command_str = " ".join(command)
    else:
        command_str = command
        
    # Ensure the command is in lowercase for case-insensitive filtering
    command_to_execute = command_str.lower().strip()

    # --- Enforce Protocol 101 Whitelist ---
    for prohibited_op in PROHIBITED_COMMANDS:
        if command_to_execute.startswith(prohibited_op):
            # Violation of the Mandate of the Whitelist
            error_msg = f"PROTOCOL VIOLATION: Unauthorized command '{command_to_execute[:20]}...'. Use designated MCP tool."
            
            # STOP and REPORT, as mandated by Protocol 101
            print(f"[CRITICAL] {error_msg}")
            raise ProtocolViolationError(error_msg)

    # If allowed, execute
    try:
        # Use shell=True if it's a string, False if list (standard subprocess behavior)
        shell = isinstance(command, str)
        
        return subprocess.run(
            command,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=True,
            shell=shell
        )
    except subprocess.CalledProcessError as e:
        # Log failure
        print(f"[!] Command failed: {command_str}")
        print(f"[!] Stderr: {e.stderr}")
        raise
