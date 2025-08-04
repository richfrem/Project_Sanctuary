# main.py
# Version: 0.7
# Last Modified: [Current Date]
#
# Orchestrates the Chimera Sandbox environment for AGORA.
# v0.7 integrates an autoencoder for ML-based anomaly detection (WI_008 v0.7)
# and enhances logging for deeper auditability and transparency (WI_002, P12).
# Aligns with: WI_008 v0.7, P24 (Epistemic Immune System), P31 (Airlock Protocol)

import os
import json
import logging
import argparse
import requests
import openai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration and Setup ---
VERSION = '0.7.0-alpha'
CONFIG_FILE = 'config.json'
LOG_DIR = 'logs'

# --- v0.7: Autoencoder Anomaly Detection ---
class Autoencoder(nn.Module):
    """A simple autoencoder for anomaly detection."""
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Assuming normalized input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderAnomalyDetector:
    """
    Manages the training and usage of the autoencoder for detecting anomalous inputs.
    Aligns with P24 (Epistemic Immune System).
    """
    def __init__(self, input_dim=768, threshold=0.05): # Assuming a common embedding size
        self.model = Autoencoder(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.threshold = threshold
        self.input_dim = input_dim
        self.is_trained = False
        logging.info(f"[v0.7] AutoencoderAnomalyDetector initialized with threshold: {self.threshold}")

    def train(self, normal_data_loader):
        """Trains the autoencoder on a dataset of 'normal' inputs."""
        self.model.train()
        for epoch in range(10): # Example training loop
            for data in normal_data_loader:
                inputs, = data
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.is_trained = True
        logging.info("[v0.7] Autoencoder training complete. Anomaly detection is now active.")

    def is_anomalous(self, input_vector):
        """
        Checks if a given input vector is anomalous based on reconstruction error.
        Returns True if anomalous, False otherwise.
        """
        if not self.is_trained:
            logging.warning("[v0.7] Anomaly detector not trained. Skipping check.")
            return False
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            if input_tensor.shape[1] != self.input_dim:
                 # Pad or truncate the input to match the model's expected input dimension
                padded_tensor = torch.zeros(1, self.input_dim)
                copy_len = min(input_tensor.shape[1], self.input_dim)
                padded_tensor[0, :copy_len] = input_tensor[0, :copy_len]
                input_tensor = padded_tensor
                
            reconstructed = self.model(input_tensor)
            loss = self.criterion(reconstructed, input_tensor)
        
        logging.info(f"[v0.7] Input reconstruction error: {loss.item()}")
        if loss.item() > self.threshold:
            logging.warning(f"[v0.7] Anomaly detected! Reconstruction error {loss.item()} exceeds threshold {self.threshold}.")
            return True
        return False

# --- Core Functions (Updated for v0.7) ---

def setup_logging():
    """Sets up logging for console and file outputs."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "chimera_main.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Chimera Sandbox Main Orchestrator v{VERSION} starting up.")
    logging.info("Logging configured for console and chimera_main.log")

def load_config():
    """Loads the JSON configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # TODO: Add schema validation for config
        openai.api_key = config.get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in config.json")
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"CRITICAL: {CONFIG_FILE} not found. Please create it.")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"CRITICAL: Error loading {CONFIG_FILE}: {e}")
        exit(1)

def get_openai_response(prompt, conversation_history, config):
    """
    Sends a prompt to the OpenAI API and gets a response.
    NOTE: This uses the legacy openai<1.0 syntax.
    """
    messages = conversation_history + [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=config.get("model", "gpt-4-turbo"),
            messages=messages,
            temperature=config.get("temperature", 0.5),
            max_tokens=config.get("max_tokens", 1500)
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error communicating with OpenAI API: {e}")
        return "Error: Could not retrieve a response from the AI model."

def validate_and_process_input(user_input, anomaly_detector):
    """
    v0.7: Enhanced input validation pipeline with autoencoder anomaly detection.
    Aligns with P31 (Airlock Protocol).
    """
    # 1. Basic validation (length, etc.)
    if len(user_input) > 4096:
        logging.warning("Input exceeds max length. Truncating.")
        user_input = user_input[:4096]

    # 2. v0.7: Anomaly Detection
    # In a real system, we'd convert text to an embedding vector. Here, we simulate it.
    # For demonstration, we'll create a reproducible vector from the input text.
    input_vector = [ord(c) % 256 for c in user_input]
    normalized_vector = [x / 255.0 for x in input_vector]
    
    if anomaly_detector.is_anomalous(normalized_vector):
        # Handle anomaly: e.g., flag for review, use a safer model, or reject.
        return None, "Input flagged as anomalous by the Epistemic Immune System (P24). Processing halted for review."

    # 3. Process commands
    if user_input.strip().lower() == '/quit':
        return "quit", None
    if user_input.strip().lower() == '/help':
        help_text = """
        Chimera Sandbox v0.7 Commands:
        /quit          - Exits the application.
        /help          - Displays this help message.
        /status        - [TODO] Show system status.
        """
        return "command", help_text
    
    return "prompt", user_input


def main():
    """Main execution function."""
    setup_logging()
    config = load_config()

    # --- v0.7: Initialize and train the anomaly detector ---
    anomaly_detector = AutoencoderAnomalyDetector()
    
    # Create a dummy dataset of "normal" text inputs and train the autoencoder
    # In a real system, this would be a curated dataset.
    normal_texts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a python function to sort a list.",
        "Tell me a story about a dragon."
    ]
    # Convert texts to normalized vectors
    normal_vectors = []
    for text in normal_texts:
        vec = [ord(c) % 256 for c in text]
        norm_vec = [x / 255.0 for x in vec]
        # Pad or truncate to the fixed input dimension
        padded_vec = norm_vec + [0.0] * (anomaly_detector.input_dim - len(norm_vec))
        normal_vectors.append(padded_vec[:anomaly_detector.input_dim])
        
    normal_tensors = torch.FloatTensor(normal_vectors)
    normal_dataset = TensorDataset(normal_tensors)
    normal_loader = DataLoader(dataset=normal_dataset, batch_size=2, shuffle=True)
    
    anomaly_detector.train(normal_loader)
    # --- End of Anomaly Detector Setup ---

    conversation_history = [
        {"role": "system", "content": config.get("system_prompt", "You are a helpful AI assistant.")}
    ]
    
    print(f"--- Chimera Sandbox v{VERSION} ---")
    print("Connected to AI model. Type '/help' for commands or '/quit' to exit.")

    try:
        while True:
            user_input = input("\nUser > ")
            
            action, data = validate_and_process_input(user_input, anomaly_detector)

            if action == "quit":
                break
            elif action == "command":
                print(f"System: {data}")
            elif action == "prompt":
                response = get_openai_response(data, conversation_history, config)
                print(f"\nAI > {response}")
                conversation_history.append({"role": "user", "content": data})
                conversation_history.append({"role": "assistant", "content": response})
            elif action is None: # Anomaly detected
                 print(f"System Alert: {data}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    finally:
        logging.info("Chimera Sandbox shutting down.")
        print("--- Session Ended ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Chimera Sandbox v{VERSION}")
    # TODO: Add command-line arguments if needed
    args = parser.parse_args()
    main()