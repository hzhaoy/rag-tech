from typing import Optional

from llama_index.llms.openai import OpenAI


def create_chat_model(
    model: str, api_key: Optional[str] = None, api_base: Optional[str] = None
):
    return OpenAI(model=model, api_key=api_key, api_base=api_base, is_chat_model=True)
