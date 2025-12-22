# Liquid Neural Networks (LNN)

**Status:** Researching
**Last Updated:** 2025-12-22

## Overview
Liquid Neural Networks (LNNs) are a class of continuous-time recurrent neural networks. Unlike standard ANNs that have fixed weights after training, LNNs are modeled by Ordinary Differential Equations (ODEs) where the system's time constant depends on the input. This allows the network's behavior to remain flexible ("liquid") even during inference, adapting to changing data streams.

## Key Characteristics
*   **Continuous-Time:** Neural states evolve via ODEs, not discrete steps.
*   **Input-Dependent Dynamics:** The "Liquid Time Constant" (LTC) changes based on input, allowing adaptive receptivity.
*   **Causality:** The system's output depends strictly on past inputs (causal filtering).
*   **Efficiency:** Highly expressive with fewer neurons than traditional RNNs/LSTMs.

## Structure
*   `notes/fundamentals.md`: Core architecture and biological inspiration.
