import os
from dataclasses import dataclass
from typing import Dict, Optional

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from src.config import PROJECT_ROOT, get_logger
from src.models.chat_models import create_chat_model
from src.models.embedding_models import create_embedding_model

from .types import DocumentEvent, SemanticChunkEvent

logger = get_logger(__name__)


@dataclass
class SemanticChunkConfig:
    """Configuration for semantic chunking parameters"""

    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    max_chunk_size: int = 500
    min_chunk_size: int = 100
    chunk_overlap: int = 50
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class SemanticChunkingWorkflow(Workflow):
    """
    Semantic chunking workflow using LlamaIndex's SemanticSplitterNodeParser.

    This workflow implements semantic chunking by:
    1. Creating embeddings for text segments
    2. Calculating semantic similarity between adjacent segments
    3. Splitting at semantic breakpoints where similarity drops significantly
    4. Ensuring chunks meet size constraints
    """

    def __init__(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        config: Optional[SemanticChunkConfig] = None,
    ):
        super().__init__()
        self.llm = llm
        self.embed_model = embed_model
        self.config = config or SemanticChunkConfig()

        # Initialize semantic splitter using from_defaults with config
        self.semantic_splitter = SemanticSplitterNodeParser.from_defaults(
            buffer_size=self.config.buffer_size,
            breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
            embed_model=self.embed_model,
            include_metadata=True,
            include_prev_next_rel=self.config.chunk_overlap > 0,
        )

        # Fallback splitters for size constraints
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self.token_splitter = TokenTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    @step
    async def input_node(self, ev: StartEvent) -> DocumentEvent:
        """Input node: receives raw text and creates a Document object"""
        return DocumentEvent(
            document=Document(text=ev.input_text, metadata=ev.get("metadata", {}))
        )

    @step
    async def semantic_chunking_node(self, ev: DocumentEvent) -> SemanticChunkEvent:
        """Semantic chunking node: split document based on semantic similarity"""
        document = ev.document

        logger.info(
            f"Starting semantic chunking for document of length: {len(document.text)}"
        )

        try:
            # First attempt: semantic splitting
            semantic_nodes = await self.semantic_splitter.aget_nodes_from_documents(
                [document]
            )

            # Validate chunk sizes and apply constraints
            valid_chunks = []
            oversized_chunks = []

            for node in semantic_nodes:
                chunk_size = len(node.text)
                if (
                    self.config.min_chunk_size
                    <= chunk_size
                    <= self.config.max_chunk_size
                ):
                    valid_chunks.append(node)
                elif chunk_size > self.config.max_chunk_size:
                    oversized_chunks.append(node)
                # Skip chunks that are too small

            logger.info(
                f"Semantic splitting produced {len(valid_chunks)} valid chunks, {len(oversized_chunks)} oversized chunks"
            )

            # Handle oversized chunks with sentence splitting
            for oversized_node in oversized_chunks:
                sentence_nodes = await self.sentence_splitter.aget_nodes_from_documents(
                    [
                        Document(
                            text=oversized_node.text, metadata=oversized_node.metadata
                        )
                    ]
                )
                valid_chunks.extend(sentence_nodes)

            # If no valid chunks were produced, fall back to sentence splitting
            if not valid_chunks:
                logger.warning(
                    "No valid semantic chunks produced, falling back to sentence splitting"
                )
                valid_chunks = await self.sentence_splitter.aget_nodes_from_documents(
                    [document]
                )

            # Add chunk metadata
            for i, chunk in enumerate(valid_chunks):
                chunk.metadata.update(
                    {
                        "chunk_id": f"semantic_chunk_{i+1}",
                        "chunk_size": len(chunk.text),
                        "chunking_method": "semantic",
                        "total_chunks": len(valid_chunks),
                    }
                )

            logger.info(f"Final chunking produced {len(valid_chunks)} chunks")

            return SemanticChunkEvent(
                chunks=valid_chunks,
                metadata={
                    "total_chunks": len(valid_chunks),
                    "avg_chunk_size": (
                        sum(len(c.text) for c in valid_chunks) / len(valid_chunks)
                        if valid_chunks
                        else 0
                    ),
                    "chunking_method": "semantic",
                    "embedding_model": self.config.embedding_model,
                },
            )

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            logger.info("Falling back to sentence splitting")

            # Fallback to sentence splitting
            sentence_nodes = await self.sentence_splitter.aget_nodes_from_documents(
                [document]
            )

            for i, chunk in enumerate(sentence_nodes):
                chunk.metadata.update(
                    {
                        "chunk_id": f"fallback_chunk_{i+1}",
                        "chunk_size": len(chunk.text),
                        "chunking_method": "sentence_fallback",
                        "total_chunks": len(sentence_nodes),
                    }
                )

            return SemanticChunkEvent(
                chunks=sentence_nodes,
                metadata={
                    "total_chunks": len(sentence_nodes),
                    "avg_chunk_size": (
                        sum(len(c.text) for c in sentence_nodes) / len(sentence_nodes)
                        if sentence_nodes
                        else 0
                    ),
                    "chunking_method": "sentence_fallback",
                    "embedding_model": self.config.embedding_model,
                    "fallback_reason": str(e),
                },
            )

    @step
    async def post_process_node(self, ev: SemanticChunkEvent) -> StopEvent:
        """Post-processing node: organize and format the final chunks"""
        chunks = ev.chunks
        metadata = ev.metadata

        # Calculate additional statistics
        chunk_sizes = [len(chunk.text) for chunk in chunks]

        result = {
            "chunks": [
                {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "text": chunk.text,
                    "size": len(chunk.text),
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            "metadata": {
                **metadata,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "median_chunk_size": (
                    sorted(chunk_sizes)[len(chunk_sizes) // 2] if chunk_sizes else 0
                ),
            },
        }

        return StopEvent(result=result)


class AdvancedSemanticChunker:
    """
    Advanced semantic chunking with multiple strategies and optimizations.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        config: Optional[SemanticChunkConfig] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.config = config or SemanticChunkConfig()
        self.logger = get_logger(__name__)

    async def chunk_with_topic_detection(
        self, text: str, detect_topics: bool = True
    ) -> Dict:
        """
        Advanced chunking with optional topic detection and hierarchical structuring.
        """
        workflow = SemanticChunkingWorkflow(self.llm, self.embed_model, self.config)

        # Run semantic chunking
        result = await workflow.run(input_text=text)

        if detect_topics:
            # Add topic detection for each chunk
            for chunk in result["chunks"]:
                chunk["topic"] = await self._detect_topic(chunk["text"])

        return result

    async def _detect_topic(self, text: str) -> str:
        """Detect the main topic of a text chunk"""
        prompt = f"""
        Analyze the following text and identify the main topic in 2-3 words:

        Text: {text[:500]}

        Main topic:"""

        try:
            response = await self.llm.achat(
                messages=[
                    ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=prompt)]),
                ]
            )
            topic = (
                response.message.content.strip()
                if response.message.content
                else "Unknown"
            )
            return topic[:50]  # Limit topic length
        except Exception as e:
            self.logger.error(f"Topic detection failed: {e}")
            return "Unknown"


async def main():
    """Example usage of semantic chunking"""

    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    api_base = os.getenv("OPENAI_BASE_URL")

    # Initialize models
    llm = create_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    embed_model = create_embedding_model(
        model=embedding_model,
        api_key=api_key,
        api_base=api_base,
    )

    # Load example text
    with open(os.path.join(PROJECT_ROOT, "data", "chunk_demo.txt"), "r") as f:
        document_text = f.read()

    # Create semantic chunker
    config = SemanticChunkConfig()

    workflow = SemanticChunkingWorkflow(llm, embed_model, config)

    # Run semantic chunking
    result = await workflow.run(input_text=document_text)

    # Display results
    print("Semantic Chunking Results:")
    print(f"Total chunks: {result['metadata']['total_chunks']}")
    print(f"Average chunk size: {result['metadata']['avg_chunk_size']:.1f} characters")
    print(f"Chunking method: {result['metadata']['chunking_method']}")
    print(f"Embedding model: {result['metadata']['embedding_model']}")
    print("=" * 80)

    for i, chunk in enumerate(result["chunks"]):
        print(f"\nChunk {i+1} (ID: {chunk['chunk_id']}):")
        print(f"Size: {chunk['size']} characters")
        print(f"Text: {chunk['text'][:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
