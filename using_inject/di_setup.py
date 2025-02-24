# di_setup.py

import inject
import os
from llms import LLM, OpenAILLM, GroqLLM, HuggingFaceLLM
from llm_config import LLMConfig
import logging

def llm_provider_injector(binder: inject.Binder):
    """
    This function binds the LLM class and LLMConfig to specific instances.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai")  
    api_key = os.getenv("LLM_API_KEY", "dummy_api_key") 
    model_name = os.getenv("LLM_MODEL", "gpt-4")  
    logging.info(f'provider is {provider}')
    llm_config = LLMConfig(provider=provider, api_key=api_key, model_name=model_name)
    binder.bind(LLMConfig, llm_config)

    # Choose which LLM to bind based on provider
    if provider == "groq":
        binder.bind(LLM, GroqLLM(llm_config))
    elif provider == "huggingface":
        binder.bind(LLM, HuggingFaceLLM(llm_config))
    else:
        binder.bind(LLM, OpenAILLM(llm_config))  # Default



def configure_injection():
    """
    Call this function at application startup to configure dependency injection.
    """
    inject.configure(llm_provider_injector)
