# llms.py

from abc import ABC, abstractmethod

# Base LLM class (Interface)
class LLM(ABC):
    @abstractmethod
    def call_llm(self, prompts: list) -> dict:
        pass

# OpenAI Implementation
class OpenAILLM(LLM):
    def generate_text(self, prompts: list) -> dict:
        return f"OpenAI Responses: {prompts}"  # Dummy example (reverses text)

# Groq Implementation
class GroqLLM(LLM):
    def generate_text(self, prompt: list) -> dict:
        return f"Groq Response: {prompts}"

# Hugging Face Implementation
class HuggingFaceLLM(LLM):
    def generate_text(self, prompt: list) -> dict:
        return f"HuggingFace Response: {prompt.lower()}"
