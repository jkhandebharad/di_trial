# main.py

import logging
import inject
from fastapi import FastAPI, Depends, Query
from di_setup import configure_injection
from llms import LLM
from llm_config import LLMConfig

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def startup():
    configure_injection()

# Dependency: Get the pre-configured LLM instance
def get_llm() -> LLM:
    llm_instance = inject.instance(LLM)
    logging.info(f"Using LLM instance: {id(llm_instance)}")
    return llm_instance

# Dependency: Get the injected LLMConfig instance
def get_llm_config() -> LLMConfig:
    return inject.instance(LLMConfig)

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
