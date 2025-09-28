from typing import List

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.workflow import Event


class DocumentEvent(Event):
    document: Document


class SentenceEvent(Event):
    sentences: List[str]


class LLMDecisionEvent(Event):
    sentence_groups: List[List[str]]


class ChunkEvent(Event):
    """Event for generic chunks"""

    chunks: List[TextNode]
    metadata: dict


class SemanticChunkEvent(ChunkEvent):
    """Event for semantic chunks"""

    pass


class ContextualChunkEvent(ChunkEvent):
    """Event for contextual chunks"""

    pass
