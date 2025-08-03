import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions import Normal
import numpy as np
from typing import List, Dict, Union

# DOCTRINE_LINK: WI_008 v0.5, P54: Asch Doctrine, P24: Epistemic Immune System
# Implements a full GAN-based Adversarial Engine with AdamW optimizers and RL-embedded Discriminator
# for dynamic threat generation, federated learning, and differential privacy.
class AdversarialEngine(nn.Module):
    def __init__(self, latent_dim: int = 100, embedding_dim: int = 768, learning_rate: float = 0.0002):
        super(AdversarialEngine, self).__init__()
        print("[ENGINE] Adversarial Engine Initializing (v0.5 GAN+RL).")
        
        # Generator: Generates synthetic adversarial data
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, embedding_dim),
            nn.Tanh()  # Outputs normalized embeddings
        )
        
        # Discriminator: Classifies real vs. fake data with RL agent
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Probability output
        )
        
        # Optimizers (AdamW for improved convergence, WI_008 v0.5)
        self.optimizer_g = AdamW(self.generator.parameters(), lr=learning_rate, weight_decay=0.01)
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # RL-embedded Discriminator for dynamic threat detection
        self.rl_agent = RLDynamicDiscriminator(embedding_dim)
        
        # Loss function
        self.bce_loss = nn.BCELoss()
        
        # Federated learning placeholder
        self.federated_weights = None
        
        print("[ENGINE] GAN, RL agent, and AdamW optimizers initialized.")

    # DOCTRINE_LINK: P24: Epistemic Immune System, P54: Asch Doctrine
    def generate_threats(self, threat_model: str, federated: bool = False, count: int = 10) -> List[Dict[str, Union[str, float]]]:
        """
        Generates adversarial data points for specified threat model.
        Args:
            threat_model: One of 'data_poisoning', 'asch_swarm', 'dissonance_loop', 'echo_chamber'
            federated: Whether to apply federated learning weights
            count: Number of threats to generate
        Returns:
            List of dictionaries containing synthetic data and metadata
        """
        z = torch.randn(count, self.latent_dim)
        with torch.no_grad():
            generated_data = self.generator(z)
        
        if federated:
            weights = self._aggregate_federated_weights()
            generated_data = self._apply_federated_weights(generated_data, weights)
        
        # Apply differential privacy (WI_008 v0.5, xAI Proposal)
        generated_data = self.apply_differential_privacy(generated_data)
        
        # Threat-specific modifications
        threats = {
            "data_poisoning": {"data": generated_data + torch.randn_like(generated_data) * 0.1, "label": "biased"},
            "asch_swarm": {"data": generated_data + torch.ones_like(generated_data), "label": "consensus"},
            "dissonance_loop": {"data": generated_data * -1, "label": "contradictory"},
            "echo_chamber": {"data": generated_data * 2, "label": "amplified"}
        }
        
        if threat_model not in threats:
            raise ValueError(f"Unknown threat model: {threat_model}")
        
        # Generate threat metadata
        threat_data = threats[threat_model]
        output = [
            {
                "source": "gan_v0.5",
                "content": f"Generated_{threat_model}_{i}",
                "bias_vector": threat_data["data"][i].mean().item(),
                "label": threat_data["label"]
            } for i in range(count)
        ]
        
        # Log generation for transparency (WI_002: Glass Box Principle)
        with open("logs/adversarial_threats.log", "a") as log_file:
            log_file.write(f"Threat Model: {threat_model}, Count: {count}, Federated: {federated}\n")
        
        return output

    def _aggregate_federated_weights(self) -> torch.Tensor:
        """
        Placeholder for federated learning weight aggregation (Flower/PySyft).
        """
        # Stub: Simulate federated weights
        return torch.ones(self.latent_dim)

    def _apply_federated_weights(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Applies federated weights to generated data.
        """
        return data * weights

    # DOCTRINE_LINK: WI_008 v0.5 (xAI Proposal)
    def apply_differential_privacy(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies differential privacy by adding Gaussian noise.
        """
        noise = Normal(0, 0.1).sample(data.shape)
        return data + noise

    def train_gan_step(self, real_data: torch.Tensor, batch_size: int = 64) -> tuple:
        """
        Performs one training step for the GAN and RL Discriminator.
        Args:
            real_data: Real data batch for training
            batch_size: Size of the batch
        Returns:
            Tuple of (discriminator loss, generator loss, RL reward)
        """
        # Train Discriminator
        self.optimizer_d.zero_grad()
        real_pred = self.discriminator(real_data)
        real_labels = torch.ones(batch_size, 1)
        real_loss = self.bce_loss(real_pred, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        fake_pred = self.discriminator(fake_data.detach())
        fake_labels = torch.zeros(batch_size, 1)
        fake_loss = self.bce_loss(fake_pred, fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        fake_pred = self.discriminator(fake_data)
        g_loss = self.bce_loss(fake_pred, real_labels)  # Trick discriminator
        g_loss.backward()
        self.optimizer_g.step()
        
        # RL Discriminator Update (WI_008 v0.5)
        reward = self.rl_agent.update(real_data, fake_data)
        
        # Log training metrics (WI_002: Glass Box Principle)
        with open("logs/gan_training.log", "a") as log_file:
            log_file.write(f"GAN Step: D_Loss={d_loss.item():.4f}, G_Loss={g_loss.item():.4f}, RL_Reward={reward:.4f}\n")
        
        return d_loss.item(), g_loss.item(), reward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

# DOCTRINE_LINK: WI_008 v0.5
# RL agent for dynamic threat detection in Discriminator.
class RLDynamicDiscriminator:
    def __init__(self, embedding_dim: int):
        self.q_table = {}  # Simplified Q-table for RL
        self.embedding_dim = embedding_dim
        self.learning_rate = 0.1
        self.gamma = 0.9
        print("[RL] Dynamic Discriminator Initialized (v0.5).")

    def update(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> float:
        """
        Updates RL agent based on threat detection accuracy.
        Args:
            real_data: Real data batch
            fake_data: Generated adversarial data
        Returns:
            RL reward
        """
        state = self._get_state(real_data, fake_data)
        action = "detect_threat"
        reward = 1.0 if self._is_threat_detected(fake_data) else -1.0
        current_q = self.q_table.get(state, 0)
        self.q_table[state] = current_q + self.learning_rate * (reward + self.gamma * 0 - current_q)
        
        # Log RL update (WI_002: Glass Box Principle)
        with open("logs/rl_discriminator.log", "a") as log_file:
            log_file.write(f"State: {state}, Reward: {reward:.4f}\n")
        
        return reward

    def _get_state(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> str:
        """
        Generates state representation for RL.
        """
        real_mean = real_data.mean().item()
        fake_mean = fake_data.mean().item()
        return f"{real_mean:.2f}_{fake_mean:.2f}"

    def _is_threat_detected(self, fake_data: torch.Tensor) -> bool:
        """
        Determines if a threat is detected based on data deviation.
        """
        return fake_data.mean().item() > 0.5  # Simplified detection logic