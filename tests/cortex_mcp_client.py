import subprocess
import json
import sys
import os
import time
import threading
import argparse

def read_stream(stream, prefix, output_list):
    """Read stream line by line and store in list."""
    try:
        for line in iter(stream.readline, ''):
            if not line: break
            clean_line = line.strip()
            if clean_line:
                # Filter out the specific known noise if any remains
                if "checking chromadb" in clean_line.lower(): continue
                output_list.append(clean_line)
    except Exception as e:
        pass

def run_mcp_tool(tool_name, tool_args, timeout=30):
    """Run a specific MCP tool against a fresh server instance."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Start server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_servers.rag_cortex.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=os.getcwd()
    )

    stdout_lines = []
    stderr_lines = []
    
    # Start reader threads
    t_out = threading.Thread(target=read_stream, args=(proc.stdout, "STDOUT", stdout_lines))
    t_err = threading.Thread(target=read_stream, args=(proc.stderr, "STDERR", stderr_lines))
    t_out.daemon = True
    t_err.daemon = True
    t_out.start()
    t_err.start()

    # Wait for startup
    time.sleep(2)
    
    # --- Handshake ---
    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "manual-cli", "version": "1.0"}
        }
    }
    
    try:
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"ERROR: Failed to write initialize: {e}")
        return

    # Wait for initialize response
    time.sleep(1)
    
    # 2. Initialized notification
    init_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    try:
        proc.stdin.write(json.dumps(init_notif) + "\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"ERROR: Failed to write initialized notification: {e}")
        return

    # Wait a moment
    time.sleep(1)

    # --- Tool Call ---
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": tool_args
        }
    }
    
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"ERROR: Failed to write to stdin: {e}")
        return

    # Wait for response (up to timeout)
    start_time = time.time()
    response_found = False
    
    while time.time() - start_time < timeout:
        if stdout_lines:
            for line in stdout_lines:
                try:
                    data = json.loads(line)
                    if data.get("id") == 1:
                        print(json.dumps(data, indent=2))
                        response_found = True
                        break
                except:
                    pass
        if response_found: break
        time.sleep(0.5)

    if not response_found:
        print("TIMEOUT: No valid JSON-RPC response received.")
        if stderr_lines:
            print("\nSTDERR Output:")
            for line in stderr_lines:
                print(line)

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="Tool name")
    parser.add_argument("args", help="JSON arguments string", default="{}")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    args = parser.parse_args()
    
    try:
        tool_args = json.loads(args.args)
    except:
        tool_args = {}
        
    run_mcp_tool(args.tool, tool_args, args.timeout)
