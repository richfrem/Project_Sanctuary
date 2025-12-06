
import unittest
import requests
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestChainForgeOllama(unittest.TestCase):
    """
    Scenario 1: Verify 'Forge -> Ollama' connectivity.
    This tests if the Python code can successfully talk to the Ollama container
    and get a response from the Sanctuary model.
    """
    
    def setUp(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = "Sanctuary-Qwen2-7B:latest" # The target fine-tune
        
    def test_ollama_service_reachable(self):
        """Test 1: Is Ollama running and reachable?"""
        print(f"\n[Test] Checking Ollama at {self.ollama_host}...")
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
            self.assertEqual(response.status_code, 200, "Ollama service should return 200 OK")
            print("✅ Ollama is reachable.")
        except requests.exceptions.ConnectionError:
            self.fail(f"❌ Could not connect to Ollama at {self.ollama_host}. Is the container running?")

    def test_sanctuary_model_loaded(self):
        """Test 2: Is the Sanctuary model available?"""
        url = f"{self.ollama_host}/api/tags"
        response = requests.get(url)
        models = response.json().get('models', [])
        found = any(m['name'] == self.model_name for m in models)
        
        if not found:
            print(f"⚠️ Warning: {self.model_name} not found. Available: {[m['name'] for m in models]}")
            # Fallback to qwen2:0.5b for testing if sanctuary not present
            if any(m['name'] == "qwen2:0.5b" for m in models):
                print("⚠️ Using 'qwen2:0.5b' as fallback.")
                self.model_name = "qwen2:0.5b"
            else:
                 self.fail(f"❌ Neither {self.model_name} nor fallback found!")
        else:
            print(f"✅ Model {self.model_name} is available.")

    def test_model_inference(self):
        """Test 3: Can we get a generation response?"""
        print(f"[Test] Generating with {self.model_name}...")
        
        prompt = "Define 'Protocol 101' in one sentence."
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 50
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate", 
                json=payload, 
                timeout=30 # 30s timeout for inference
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get('response', '').strip()
            
            print(f"🤖 Model Response: {answer}")
            self.assertTrue(len(answer) > 5, "Response should be non-empty")
            print("✅ Inference successful.")
            
        except requests.exceptions.ReadTimeout:
            self.fail("❌ Inference timed out (30s). Model is hanging.")
        except Exception as e:
            self.fail(f"❌ Inference failed: {e}")

if __name__ == '__main__':
    unittest.main()
