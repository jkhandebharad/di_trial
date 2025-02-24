# llms.py

from abc import ABC, abstractmethod
from config import LLMConfig

class LLM(ABC):
    """
    Abstract Base Class (ABC) for all LLM implementations.
    Requires an `LLMConfig` instance.
    """
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def call_llm(self, prompts: list) -> dict:
        pass

class OpenAILLM(LLM):
    def call_llm(self, prompts: list) -> dict:
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "responses": prompts
        }

class GroqLLM(LLM):
    def call_llm(self, prompts: list) -> dict:
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "responses": [prompt.upper() for prompt in prompts]
        }

class HuggingFaceLLM(LLM):
    def call_llm(self, prompts: list) -> dict:
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "responses": [prompt.lower() for prompt in prompts]
        }
