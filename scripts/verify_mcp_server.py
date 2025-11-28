import subprocess
import json
import os
import sys

def test_server():
    # Command to run the server
    cmd = [sys.executable, "-m", "mcp_servers.system.git_workflow.server"]
    
    print(f"Starting server with command: {' '.join(cmd)}")
    
    # Start the server process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        cwd=os.getcwd(),
        env=os.environ.copy()
    )

    try:
        # 1. Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        }
        
        print("\nSending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        print(f"Received: {response.strip()}")
        
        # 2. List Tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("\nSending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        print(f"Received: {response.strip()}")
        
        data = json.loads(response)
        if "result" in data and "tools" in data["result"]:
            tools = [t["name"] for t in data["result"]["tools"]]
            print(f"\nSUCCESS! Found tools: {tools}")
        else:
            print("\nFAILED to find tools in response.")

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    test_server()
