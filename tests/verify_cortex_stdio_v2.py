import subprocess
import json
import sys
import os
import time
import threading

def read_stream(stream, prefix, output_list):
    """Read stream line by line and store in list."""
    try:
        for line in iter(stream.readline, ''):
            if not line: break
            clean_line = line.strip()
            if clean_line:
                print(f"{prefix}: {clean_line}")
                output_list.append(clean_line)
    except Exception as e:
        print(f"Error reading {prefix}: {e}")

def verify_server_stdio():
    print("Starting server process verification...")
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
        cwd="/Users/richardfremmerlid/Projects/Project_Sanctuary"
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

    # Wait a bit for startup noise
    time.sleep(2)
    
    # Send request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    print("Sending JSON-RPC request...")
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"Failed to write to stdin: {e}")
        return

    # Wait for response
    time.sleep(15)
    
    print("\n--- Analysis ---")
    
    # Analyze STDOUT (Should only contain JSON)
    json_count = 0
    garbage_count = 0
    
    for line in stdout_lines:
        try:
            json.loads(line)
            json_count += 1
        except json.JSONDecodeError:
            garbage_count += 1
            print(f"GARBAGE DETECTED in STDOUT: {line}")
            
    if garbage_count == 0 and json_count > 0:
        print("✅ SUCCESS: STDOUT contains only valid JSON.")
    elif garbage_count > 0:
        print(f"❌ FAILURE: STDOUT contains {garbage_count} garbage lines.")
    else:
        print("⚠️ WARNING: No JSON response received (timeout or startup delay).")

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

if __name__ == "__main__":
    verify_server_stdio()
