"""What the domain needs from outside: the LLM and the decision model. Everything else the stages use is data
handed in."""
from abc import ABC, abstractmethod


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached or answered something unusable."""


class LLMPort(ABC):
    """The LLM as the LLM stage uses it: route to a tool, and judge whether speech was addressed."""
    @abstractmethod
    def route(self, utterance: str, tools: list) -> dict:
        """Pick exactly one of `tools` (tool_choice=required). Returns {"tool": name, "args": {...}, "ms": float}.
        Raises LLMUnavailable."""

    @abstractmethod
    def p_addressed(self, utterance: str) -> dict:
        """P(the utterance is directed at the assistant). Returns {"p_yes": float, "ms": float}.
        Raises LLMUnavailable."""


class DeciderPort(ABC):
    """A typed-decision model: picks one of several described options, with a probability for each."""
    @abstractmethod
    def choose(self, state: str, question: str, options: dict) -> dict:
        """`options` is {key: description}. Returns {"choice": key, "probabilities": {key: p}, "ms": float}."""
