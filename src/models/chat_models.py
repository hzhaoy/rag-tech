from typing import Optional
from llama_index.llms.openai_like import OpenAILike


def create_chat_model(
    model: str, api_key: Optional[str] = None, api_base: Optional[str] = None
) -> OpenAILike:
    return OpenAILike(
        model=model, api_key=api_key, api_base=api_base, is_chat_model=True
    )
