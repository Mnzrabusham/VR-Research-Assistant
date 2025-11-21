# VR Research Assistant

A RAG-based question-answering system with OpenAI fine-tuning for research papers using LangChain and ChromaDB.

## Features
- Semantic search across VR research papers
- Retrieval-Augmented Generation (RAG) for accurate answers
- OpenAI fine-tuning pipeline
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

## Generate training data
python generate_training_data.py

## Fine-tune the model
python finetune_openai.py

## Test fine-tuned model
python test_finetuned.py

## Run RAG with fine-tuning
python vr_rag_finetuned.py

## Usage

Ask questions related to the field/papers.

### Cost:
- Training: ~$10-20 depending on your train data size (one-time)
- Inference: Same as GPT-3.5 base model

## Architecture

- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## License
MIT