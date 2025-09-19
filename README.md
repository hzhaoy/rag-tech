# rag-tech

This repository contains implementations of Retrieval-Augmented Generation (RAG) technologies using [LlamaIndex](https://github.com/run-llama/llama_index). The project provides modular workflows and utilities for document chunking, semantic search, and LLM-based decision making, enabling efficient integration of external knowledge into generative AI applications.

The project is actively maintained and new features and modules will be added over time.

## Features

- Document chunking and sentence splitting
- LLM-driven semantic grouping of text
- Easy integration with custom LLM endpoints

## Getting Started

1. **Install dependencies:**

    First, install [uv](https://docs.astral.sh/uv/).

    Then run the following command:
   ```sh
   uv sync
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root with your API keys and model settings.

3. **Run the example workflow:**
   ```sh
   python -m src.chunks.agentic_chunk
   ```

## Project Structure

- `src/chunks/`: Document chunking and workflow logic
- `src/models/`: LLM model configuration and utilities
- `src/config.py`: Project configuration and environment setup

## License

MIT

## Acknowledgements

Built with [LlamaIndex](https://github.com/run-llama/llama_index).