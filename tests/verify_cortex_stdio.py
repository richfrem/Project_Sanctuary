import subprocess
import json
import sys
import os

def verify_server_stdio():
    print("Starting server process...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
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

    # Request to list tools
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    try:
        # Send request
        input_str = json.dumps(request) + "\n"
        stdout, stderr = proc.communicate(input=input_str, timeout=10)
        
        print(f"--- STDERR ---\n{stderr}\n----------------")
        
        # Check for non-JSON lines in stdout
        lines = stdout.strip().split('\n')
        json_lines = 0
        garbage_lines = 0
        
        print(f"--- STDOUT ({len(lines)} lines) ---")
        for line in lines:
            if not line.strip(): continue
            try:
                json.loads(line)
                print(f"[JSON] {line[:100]}...")
                json_lines += 1
            except json.JSONDecodeError:
                print(f"[GARBAGE] {line}")
                garbage_lines += 1
        print("----------------")
                
        if garbage_lines == 0 and json_lines > 0:
            print("SUCCESS: Output is clean JSON")
        elif garbage_lines > 0:
            print(f"FAILURE: Found {garbage_lines} lines of garbage output")
        else:
            print("FAILURE: No output received")
            
    except subprocess.TimeoutExpired:
        proc.kill()
        print("TIMEOUT: Server did not respond in time")
    except Exception as e:
        proc.kill()
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_server_stdio()
