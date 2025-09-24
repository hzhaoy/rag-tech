# rag-tech

This repository contains implementations of Retrieval-Augmented Generation (RAG) technologies using [LlamaIndex](https://github.com/run-llama/llama_index). The project provides modular workflows and utilities for document chunking, semantic search, and LLM-based decision making, enabling efficient integration of external knowledge into generative AI applications.

The project is actively maintained and new features and modules will be added over time.

## Features

- **Semantic Chunking**: Intelligent document splitting based on semantic similarity using embeddings
- **Agentic Chunking**: LLM-driven semantic grouping of text with decision-making capabilities
- **Document Processing**: Support for various document types and formats
- **Configurable Parameters**: Flexible chunking strategies with customizable settings
- **Easy Integration**: Simple integration with custom LLM and embedding endpoints

## Getting Started

1. **Install dependencies:**

    First, install [uv](https://docs.astral.sh/uv/).

    Then run the following command:
   ```sh
   uv sync
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root with your API keys and model settings.

3. **Run the example workflows:**
   ```sh
   # Run agentic chunking
   python -m src.chunks.agentic_chunk

   # Run semantic chunking
   python -m src.chunks.semantic_chunk
   ```

## Semantic Chunking

Semantic chunking uses embeddings to identify natural breakpoints in text based on semantic similarity. This approach creates more coherent chunks compared to traditional rule-based methods.

### Key Features:
- **Embedding-based similarity**: Uses embeddings to measure semantic similarity between text segments
- **Configurable breakpoints**: Adjust percentile threshold for chunk boundaries
- **Size constraints**: Ensures chunks meet minimum/maximum size requirements
- **Fallback strategies**: Automatically falls back to sentence or token-based chunking if needed
- **Topic detection**: Optional topic identification for each chunk

### Usage Example:
```python
from src.chunks.semantic_chunk import SemanticChunkingWorkflow, SemanticChunkConfig

# Configure semantic chunking
config = SemanticChunkConfig(
    buffer_size=2,
    breakpoint_percentile_threshold=90,
    max_chunk_size=800,
    min_chunk_size=150,
    chunk_overlap=30,
)

# Create workflow
workflow = SemanticChunkingWorkflow(llm, embed_model, config)

# Run chunking
result = await workflow.run(input_text="Your text here...")
```

## Project Structure

- `src/chunks/`: Document chunking and workflow logic
  - `agentic_chunk.py`: LLM-driven chunking with decision making
  - `semantic_chunk.py`: Embedding-based semantic chunking
  - `types.py`: Event types for workflow processing
- `src/models/`: LLM model configuration and utilities
- `src/config.py`: Project configuration and environment setup

## License

MIT

## Acknowledgements

Built with [LlamaIndex](https://github.com/run-llama/llama_index).