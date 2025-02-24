# main.py

import os
import logging
from fastapi import FastAPI, Depends, Query
from llms import LLM, OpenAILLM, GroqLLM, HuggingFaceLLM
from config import LLMConfig

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_llm_config() -> LLMConfig:
    """Dynamically create `LLMConfig` from environment variables."""
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("LLM_API_KEY", "dummy_api_key"),
        model_name=os.getenv("LLM_MODEL", "gpt-4")
    )

def get_llm(config: LLMConfig = Depends(get_llm_config)) -> LLM:
    """Select the correct LLM based on the config."""
    provider = config.provider.lower()
    logging.info(f"hey we are creating instance on each call : {provider}")

    if provider == "groq":
        return GroqLLM(config)
    elif provider == "huggingface":
        return HuggingFaceLLM(config)
    return OpenAILLM(config)  # Default

@app.get("/generate")
def generate_text(
    prompts: str,
    llm: LLM = Depends(get_llm),
    config: LLMConfig = Depends(get_llm_config)
):
    """
    API endpoint to generate text using the injected LLM.
    Also returns LLM configuration.
    """
    return {
        "config": config.to_dict(),
        "response": llm.call_llm([prompts])
    }
