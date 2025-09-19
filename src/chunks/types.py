from typing import List
from llama_index.core.workflow import Event
from llama_index.core import Document


class DocumentEvent(Event):
    document: Document


class SentenceEvent(Event):
    sentences: List[str]


class LLMDecisionEvent(Event):
    sentence_groups: List[List[str]]
