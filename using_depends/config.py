# config.py

class LLMConfig:
    """
    Configuration class for LLMs.
    Stores API keys, model names, and other parameters.
    """
    def __init__(self, provider: str, api_key: str, model_name: str, max_tokens: int = 256):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

    def to_dict(self):
        """Returns config details as a dictionary"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens
        }
