# council_orchestrator/orchestrator/regulator.py
# Token Flow Regulator for TPM-aware rate limiting

import time
from typing import Dict

class TokenFlowRegulator:
    """
    Manages token throughput to respect per-minute token limits (TPM).
    Prevents rate limit violations by tracking cumulative usage and pausing execution when needed.
    """
    def __init__(self, limits: dict):
        """
        Initialize the regulator with TPM limits for each engine type.

        Args:
            limits: Dictionary mapping engine types to their TPM limits
                   e.g., {'openai': 30000, 'gemini': 60000, 'ollama': 999999}
        """
        self.tpm_limits = limits
        self.usage_log = []  # List of (timestamp, token_count) tuples

    def log_usage(self, token_count: int):
        """
        Log a token usage event with current timestamp.

        Args:
            token_count: Number of tokens used in this request
        """
        self.usage_log.append((time.time(), token_count))
        self._prune_old_usage()

    def _prune_old_usage(self):
        """Remove usage entries older than 60 seconds from the log."""
        current_time = time.time()
        cutoff_time = current_time - 60.0
        self.usage_log = [(ts, count) for ts, count in self.usage_log if ts > cutoff_time]

    def wait_if_needed(self, estimated_tokens: int, engine_type: str):
        """
        Check if adding estimated_tokens would exceed TPM limit.
        If so, calculate required sleep duration and pause execution.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request
            engine_type: The engine type to check limits for
        """
        self._prune_old_usage()

        # Get TPM limit for this engine type
        tpm_limit = self.tpm_limits.get(engine_type, 999999) # Default to very high limit

        # Calculate current usage in the last 60 seconds
        current_usage = sum(count for _, count in self.usage_log)

        # Check if we would exceed the limit
        if current_usage + estimated_tokens > tpm_limit:
            # Find the oldest entry that needs to expire
            if self.usage_log:
                oldest_timestamp = self.usage_log[0][0]
                current_time = time.time()
                time_since_oldest = current_time - oldest_timestamp
                sleep_duration = 60.0 - time_since_oldest + 1.0 # Add 1 second buffer

                if sleep_duration > 0:
                    print(f"[TOKEN REGULATOR] TPM limit approaching ({current_usage + estimated_tokens}/{tpm_limit})")
                    print(f"[TOKEN REGULATOR] Pausing execution for {sleep_duration:.1f} seconds to respect rate limits...")
                    time.sleep(sleep_duration)
                    self._prune_old_usage()  # Clean up after sleep