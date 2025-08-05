# WORK_IN_PROGRESS/CODE/resilience_metrics.py (Chimera v1.2)
import hashlib
import numpy as np

class ResilienceMetrics:
    def calculate_dfs_with_proof(self, data, baseline):
        if not data["stream"]:
            dfs = 1.0
        else:
            biases = [item["bias_vector"] for item in data["stream"]]
            dfs = 1 - np.mean(biases)
        proof = hashlib.sha256(str(data).encode()).hexdigest()
        dfs = np.clip(dfs, 0.0, 1.0)  # Amended v1.2: Added clamping for DFS bounds from v1.1
        return dfs, proof