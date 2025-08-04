# WORK_IN_PROGRESS/CODE/main.py
# Version: 0.8
# Last Modified: [Current Date]
#
# Orchestrates the Chimera Sandbox.
# v0.8 hardens the gatekeeper with VAE anomaly detection and semantic cohesion analysis 
# to detect and resist the full spectrum of Asch Machine tactics (P54).
# Aligns with: WI_008 v0.8, P54: Asch Doctrine, P24: Epistemic Immune System, P31: Airlock Protocol

import os
import json
import logging
import argparse
import openai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from typing import List, Dict, Union

# --- Configuration and Setup ---
VERSION = '0.8.0'
CONFIG_FILE = 'config.json'
LOG_DIR = 'logs'

# --- v0.7: Variational Autoencoder (VAE) Anomaly Detection (Preserved & Integrated) ---
class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder for robust, probabilistic anomaly detection."""
    def __init__(self, input_dim):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, 128)
        self.encoder_fc2_mu = nn.Linear(128, 32) # Latent space mean
        self.encoder_fc2_logvar = nn.Linear(128, 32) # Latent space log variance
        # Decoder
        self.decoder_fc1 = nn.Linear(32, 128)
        self.decoder_fc2 = nn.Linear(128, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.encoder_fc1(x))
        return self.encoder_fc2_mu(h1), self.encoder_fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAEAnomalyDetector:
    """Manages the training and usage of the VAE for anomaly detection."""
    def __init__(self, input_dim=768, threshold=0.1):
        self.model = VariationalAutoencoder(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.threshold = threshold
        self.input_dim = input_dim
        self.is_trained = False
        logging.info(f"[v0.8] VariationalAutoencoder (VAE) initialized with threshold: {self.threshold}")

    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss = Reconstruction Loss (BCE) + KL Divergence."""
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self, normal_data_loader):
        """Trains the VAE on a dataset of 'normal' inputs."""
        self.model.train()
        train_loss = 0
        for epoch in range(10): # Example training loop
            for data in normal_data_loader:
                inputs, = data
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(inputs)
                loss = self.loss_function(recon_batch, inputs, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
        self.is_trained = True
        logging.info(f"[v0.8] VAE training complete. Average loss: {train_loss / len(normal_data_loader.dataset):.4f}")

    def get_anomaly_score(self, input_vector):
        """Calculates a robust anomaly score based on the VAE's combined loss."""
        if not self.is_trained:
            logging.warning("[v0.8] Anomaly detector not trained. Skipping score calculation.")
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            if input_tensor.shape[1] != self.input_dim:
                padded_tensor = torch.zeros(1, self.input_dim)
                copy_len = min(input_tensor.shape[1], self.input_dim)
                padded_tensor[0, :copy_len] = input_tensor[0, :copy_len]
                input_tensor = padded_tensor
            
            recon, mu, logvar = self.model(input_tensor)
            score = self.loss_function(recon, input_tensor, mu, logvar).item()

        logging.info(f"[v0.8] Input anomaly score (loss): {score:.4f}")
        return score

# --- v0.8: Semantic Cohesion Analysis (NEW) ---
def semantic_cohesion_analysis(inputs: List[Dict[str, Union[str, float]]], eps=0.5, min_samples=3) -> float:
    """Detects Constellation Attacks by analyzing semantic clustering of content."""
    contents = [item.get('content', '') for item in inputs if item.get('content')]
    if len(contents) < min_samples:
        return 0.0 # Not enough data to form a cluster

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(contents)
        
        # Use cosine distance for semantic similarity
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
        labels = db.labels_
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        if len(labels) == 0: return 0.0
        # Cohesion score is the ratio of items belonging to any cluster
        cohesion_score = (len(labels) - num_noise) / len(labels)
        
        logging.info(f"[v0.8] Semantic Cohesion Score: {cohesion_score:.4f} ({num_clusters} clusters found).")
        with open(os.path.join(LOG_DIR, "semantic_cohesion.log"), "a") as log_file:
            log_file.write(f"Cohesion Score: {cohesion_score:.4f}, Clusters: {num_clusters}, Labels: {labels.tolist()}\n")

        return cohesion_score
    except Exception as e:
        logging.error(f"[v0.8] Error in semantic cohesion analysis: {e}")
        return 0.0

# --- Core Functions (Preserved & Augmented) ---

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
    """Sends a prompt to the OpenAI API and gets a response (legacy openai<1.0 syntax)."""
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

def validate_and_process_batch(batch_inputs: List[Dict], anomaly_detector: VAEAnomalyDetector):
    """
    v0.8: Hardened validation with VAE for individuals and Semantic Cohesion for the batch.
    Returns a list of (action, data) tuples for valid inputs.
    """
    # 1. v0.8: Semantic Cohesion check on the whole batch
    cohesion_score = semantic_cohesion_analysis(batch_inputs)
    if cohesion_score > 0.7: # High cohesion is a strong signal of a Constellation Attack
        logging.warning(f"Potential Constellation Attack detected! Cohesion score: {cohesion_score:.4f}. Rejecting entire batch.")
        return [("cohesion_fail", "Potential Constellation Attack detected. Halting processing.")]

    # 2. Process individual inputs if batch is not a coordinated attack
    processed_outputs = []
    for user_input_dict in batch_inputs:
        user_input_str = user_input_dict.get('content', '')
        
        if len(user_input_str) > 4096:
            logging.warning("Input exceeds max length. Truncating.")
            user_input_str = user_input_str[:4096]

        # VAE anomaly detection on individual input
        input_vector = [ord(c) % 256 for c in user_input_str]
        normalized_vector = [x / 255.0 for x in input_vector]
        
        anomaly_score = anomaly_detector.get_anomaly_score(normalized_vector)
        if anomaly_score > anomaly_detector.threshold:
            processed_outputs.append((None, f"Input flagged as anomalous by VAE (Score: {anomaly_score:.2f})."))
            continue

        # Process commands
        if user_input_str.strip().lower() == '/quit':
            processed_outputs.append(("quit", None))
        elif user_input_str.strip().lower() == '/help':
            help_text = """
            Chimera Sandbox v0.8 Commands:
            /quit          - Exits the application.
            /help          - Displays this help message.
            /status        - [TODO] Show system status.
            """
            processed_outputs.append(("command", help_text))
        else:
            processed_outputs.append(("prompt", user_input_str))
            
    return processed_outputs

def main():
    """Main execution function."""
    setup_logging()
    config = load_config()

    anomaly_detector = VAEAnomalyDetector()
    
    normal_texts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a python function to sort a list.",
        "Tell me a story about a dragon."
    ]
    normal_vectors = []
    for text in normal_texts:
        vec = [ord(c) % 256 for c in text]
        norm_vec = [x / 255.0 for x in vec]
        padded_vec = norm_vec + [0.0] * (anomaly_detector.input_dim - len(norm_vec))
        normal_vectors.append(padded_vec[:anomaly_detector.input_dim])
        
    normal_tensors = torch.FloatTensor(normal_vectors)
    normal_dataset = TensorDataset(normal_tensors)
    normal_loader = DataLoader(dataset=normal_dataset, batch_size=2, shuffle=True)
    
    anomaly_detector.train(normal_loader)

    conversation_history = [
        {"role": "system", "content": config.get("system_prompt", "You are a helpful AI assistant.")}
    ]
    
    print(f"--- Chimera Sandbox v{VERSION} ---")
    print("Connected to AI model. Type '/help' for commands or '/quit' to exit.")

    try:
        while True:
            # In a real system, inputs would arrive in batches. Here we simulate it.
            user_input_str = input("\nUser > ")
            simulated_batch = [{"content": user_input_str}] # Simulate a batch of one for the conversational PoC
            
            # Process the batch
            results = validate_and_process_batch(simulated_batch, anomaly_detector)
            
            # Handle results for the single input in our simulation
            action, data = results[0]

            if action == "quit":
                break
            elif action == "cohesion_fail":
                print(f"System Alert: {data}")
            elif action == "command":
                print(f"System: {data}")
            elif action == "prompt":
                response = get_openai_response(data, conversation_history, config)
                print(f"\nAI > {response}")
                conversation_history.append({"role": "user", "content": data})
                conversation_history.append({"role": "assistant", "content": response})
            elif action is None or action == "anomaly_fail":
                 print(f"System Alert: {data}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    finally:
        logging.info("Chimera Sandbox shutting down.")
        print("--- Session Ended ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Chimera Sandbox v{VERSION}")
    args = parser.parse_args()
    main()