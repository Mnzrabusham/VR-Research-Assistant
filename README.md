# VR Research Assistant

A RAG-based question-answering system for VR/XR research papers using LangChain and ChromaDB.

## Features
- Semantic search across VR research papers
- Retrieval-Augmented Generation (RAG) for accurate answers
- ChromaDB vector database for efficient retrieval
- Supports PDF document ingestion

## Installation
```bash
pip install -r requirements.txt
```

## Setup

1. Add your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

2. Add PDF research papers to the `papers/` directory

3. Run the assistant:
```bash
python vr_rag_simple.py
```

## Usage

Ask questions about VR research. Below are examples:
- "What are the main challenges in VR attention guidance?"
- "How does eye tracking work in VR headsets?"
- "What is gaze-contingent rendering?"

## Architecture

- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## License

MIT