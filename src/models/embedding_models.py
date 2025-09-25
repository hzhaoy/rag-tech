from typing import Optional

from llama_index.embeddings.openai import OpenAIEmbedding


def create_embedding_model(
    model: str, api_key: Optional[str] = None, api_base: Optional[str] = None
):
    return OpenAIEmbedding(model=model, api_key=api_key, api_base=api_base)
