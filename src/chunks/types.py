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


class SemanticChunkEvent(Event):
    """Event for semantic chunks"""

    chunks: List[TextNode]
    metadata: dict
