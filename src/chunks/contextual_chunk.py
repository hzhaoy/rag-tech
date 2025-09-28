import os
from dataclasses import dataclass
from typing import Optional

from llama_index.core import Document
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from tqdm import tqdm

from src.config import PROJECT_ROOT, get_logger
from src.models.chat_models import create_chat_model

from .types import ChunkEvent, ContextualChunkEvent, DocumentEvent

logger = get_logger(__name__)


@dataclass
class ContextualChunkConfig:
    """Configuration for contextual retrieval chunking parameters"""

    chunk_size: int = 100
    chunk_overlap: int = 50
    context_length: int = 100  # Target length for contextual explanations
    context_model: str = os.getenv("MODEL", "gpt-4o-mini")


class ContextualChunkingWorkflow(Workflow):
    """
    Contextual retrieval chunking workflow that adds contextual information to chunks.

    This workflow implements contextual retrieval by:
    1. Splitting documents into chunks using sentence-based splitting
    2. Generating contextual explanations for each chunk
    3. Prepending context to chunk text before embedding
    4. Ensuring final chunks meet size constraints
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[ContextualChunkConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.config = config or ContextualChunkConfig()

        # Initialize sentence splitter for initial chunking
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    @step
    async def input_node(self, ctx: Context, ev: StartEvent) -> DocumentEvent:
        """Input node: receives raw text and creates a Document object"""

        await ctx.store.set("document", ev.input_text)
        return DocumentEvent(
            document=Document(text=ev.input_text, metadata=ev.get("metadata", {}))
        )

    @step
    async def initial_chunking_node(
        self, ctx: Context, ev: DocumentEvent
    ) -> ChunkEvent:
        """Initial chunking node: split document into base chunks"""
        document = ev.document

        logger.info(
            f"Starting contextual chunking for document of length: {len(document.text)}"
        )

        try:
            # Initial sentence-based splitting
            base_chunks = await self.sentence_splitter.aget_nodes_from_documents(
                [document]
            )

            logger.info(f"Initial splitting produced {len(base_chunks)} chunks")

            return ChunkEvent(
                chunks=base_chunks,
                metadata={
                    "initial_chunks": len(base_chunks),
                    "document_length": len(document.text),
                },
            )

        except Exception as e:
            logger.error(f"Initial chunking failed: {e}")
            raise

    @step
    async def contextual_enrichment_node(
        self, ctx: Context, ev: ChunkEvent
    ) -> ContextualChunkEvent:
        """Contextual enrichment node: add context to each chunk"""
        base_chunks = ev.chunks
        document_text = await ctx.store.get("document")

        logger.info(f"Adding contextual information to {len(base_chunks)} chunks")

        enriched_chunks = []

        for i, chunk in tqdm(
            enumerate(base_chunks), total=len(base_chunks), desc="Enriching chunks"
        ):
            try:
                # Generate contextual explanation
                context = await self._generate_chunk_context(
                    chunk.text, document_text, i, len(base_chunks)
                )

                # Create enriched chunk with context prepended
                enriched_text = f"{context}\n\n{chunk.text}"

                # Update chunk with enriched text
                chunk.text = enriched_text
                chunk.metadata.update(
                    {
                        "has_context": True,
                        "context_length": len(context),
                        "original_length": len(chunk.text)
                        - len(context)
                        - 2,  # Account for separators
                        "chunk_position": i + 1,
                        "total_chunks": len(base_chunks),
                    }
                )

                enriched_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Failed to enrich chunk {i}: {e}")
                # Use original chunk if context generation fails
                chunk.metadata.update(
                    {
                        "has_context": False,
                        "context_error": str(e),
                        "chunk_position": i + 1,
                        "total_chunks": len(base_chunks),
                    }
                )
                enriched_chunks.append(chunk)

        logger.info(f"Successfully enriched {len(enriched_chunks)} chunks with context")

        return ContextualChunkEvent(
            chunks=enriched_chunks,
            metadata={
                **ev.metadata,
                "enriched_chunks": len(enriched_chunks),
                "context_generation_failures": len(base_chunks)
                - len([c for c in enriched_chunks if c.metadata.get("has_context")]),
            },
        )

    async def _generate_chunk_context(
        self, chunk_text: str, document_text: str, chunk_index: int, total_chunks: int
    ) -> str:
        """Generate contextual explanation for a chunk"""

        # Create prompt for context generation
        prompt = f"""You are analyzing a document chunk to provide contextual information for retrieval.

Document: {document_text}

This is chunk {chunk_index + 1} of {total_chunks} total chunks.

Chunk content: {chunk_text}

Provide a short, succinct context (50-100 tokens) that situates this chunk within the overall document.
This context will be prepended to the chunk before embedding to improve retrieval accuracy.

Context should explain:
1. What this chunk is about
2. How it relates to the overall document
3. Any key entities, topics, or concepts it contains

Keep the context concise and informative:"""

        try:
            response = await self.llm.achat(
                messages=[
                    ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=prompt)]),
                ]
            )

            context = (
                response.message.content.strip() if response.message.content else ""
            )

            # Ensure context length is reasonable
            if (
                len(context) > self.config.context_length * 4
            ):  # Rough token to char conversion
                context = context[: self.config.context_length * 4]

            return context

        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            # Return a simple fallback context
            return f"Chunk {chunk_index + 1} from document"

    @step
    async def post_process_node(self, ev: ContextualChunkEvent) -> StopEvent:
        """Post-processing node: organize and format the final chunks"""
        chunks = ev.chunks
        metadata = ev.metadata

        # Calculate statistics
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        original_sizes = [
            chunk.metadata.get("original_length", len(chunk.text)) for chunk in chunks
        ]
        context_sizes = [chunk.metadata.get("context_length", 0) for chunk in chunks]

        result = {
            "chunks": [
                {
                    "chunk_id": f"contextual_chunk_{i+1}",
                    "text": chunk.text,
                    "size": len(chunk.text),
                    "original_size": chunk.metadata.get(
                        "original_length", len(chunk.text)
                    ),
                    "context_size": chunk.metadata.get("context_length", 0),
                    "has_context": chunk.metadata.get("has_context", False),
                    "metadata": chunk.metadata,
                }
                for i, chunk in enumerate(chunks)
            ],
            "metadata": {
                **metadata,
                "total_chunks": len(chunks),
                "avg_chunk_size": (
                    sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                ),
                "avg_original_size": (
                    sum(original_sizes) / len(original_sizes) if original_sizes else 0
                ),
                "avg_context_size": (
                    sum(context_sizes) / len(context_sizes) if context_sizes else 0
                ),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "chunks_with_context": len(
                    [c for c in chunks if c.metadata.get("has_context")]
                ),
                "chunking_method": "contextual_retrieval",
                "context_model": self.config.context_model,
            },
        }

        return StopEvent(result=result)


async def main():
    """Example usage of contextual chunking"""

    # Initialize models
    llm = create_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    # Load example text
    with open(os.path.join(PROJECT_ROOT, "data", "chunk_demo.txt"), "r") as f:
        document_text = f.read()

    # Create contextual chunker
    config = ContextualChunkConfig()

    workflow = ContextualChunkingWorkflow(llm, config, timeout=None)

    # Run contextual chunking
    result = await workflow.run(input_text=document_text)

    # Display results
    print("Contextual Chunking Results:")
    print(f"Total chunks: {result['metadata']['total_chunks']}")
    print(f"Average chunk size: {result['metadata']['avg_chunk_size']:.1f} characters")
    print(
        f"Average original size: {result['metadata']['avg_original_size']:.1f} characters"
    )
    print(
        f"Average context size: {result['metadata']['avg_context_size']:.1f} characters"
    )
    print(f"Chunks with context: {result['metadata']['chunks_with_context']}")
    print(f"Chunking method: {result['metadata']['chunking_method']}")
    print("=" * 80)

    for i, chunk in enumerate(result["chunks"]):
        print(f"\nChunk {i+1} (ID: {chunk['chunk_id']}):")
        print(f"Size: {chunk['size']} characters (original: {chunk['original_size']})")
        print(f"Has context: {chunk['has_context']}")
        if chunk["has_context"]:
            # Show first part including context
            print(f"Text (with context): {chunk['text'][:300]}...")
        else:
            print(f"Text: {chunk['text'][:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
