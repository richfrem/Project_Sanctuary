# council_orchestrator/cognitive_engines/base.py
from abc import ABC, abstractmethod

class BaseCognitiveEngine(ABC):
    """
    Abstract base class for all cognitive engines.
    Establishes the common interface for executing conversational turns,
    checking substrate health, and running functional tests.
    """
    @abstractmethod
    def execute_turn(self, prompt: str, history: list) -> str: pass
    @abstractmethod
    def check_health(self) -> dict: pass
    @abstractmethod
    def run_functional_test(self) -> dict: pass