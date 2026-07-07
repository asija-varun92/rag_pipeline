# RAG Pipeline

A Retrieval-Augmented Generation (RAG) project for ingesting documents, building a vector store, and exploring retrieval workflows with LangChain, ChromaDB, and FAISS.

## Overview

This repository contains a simple RAG pipeline for:
- ingesting PDFs and text files
- embedding content using sentence transformers
- storing vectors in a ChromaDB/FAISS database
- exploring retrieval and generation workflows in notebooks

## Features

- Document ingestion examples in `src/` and `learning/`
- ChromaDB vector store stored under `data/chroma_db/`
- Example source documents under `data/text_files/`
- Python package requirements in `requirements.txt` and `pyproject.toml`

## Requirements

- Python 3.12+
- `chromadb`
- `faiss-cpu`
- `langchain`, `langchain-core`, `langchain-community`
- `pypdf`, `pymupdf`
- `sentence-transformers`
- `ipykernel`

## Installation

Create and activate a virtual environment, then install dependencies with `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
uv install
```

If you want to install the local package from `pyproject.toml`:

```bash
uv install --develop
```

## Usage

- `main.py` is currently a placeholder entry point.
- Use the notebooks to explore ingestion and retrieval flows:
  - `learning/document.ipynb`
  - `learning/learning_rag.ipynb`
  - `src/ingestion_pipeline.ipynb`
  - `src/retrieval_pipeline.ipynb`

## Project Structure

- `data/`
  - `chroma_db/` - saved ChromaDB database files
  - `pdf/` - source PDFs for ingestion
  - `text_files/` - sample text documents
- `learning/` - notebook experiments for RAG workflows
- `src/` - pipeline notebooks and supporting code
- `main.py` - placeholder script
- `pyproject.toml` and `requirements.txt` - dependencies

## Notes

- If you want to rebuild the vector store, clear or recreate `data/chroma_db/` before rerunning ingestion.
- Update the notebooks or add scripts if you need a reusable CLI pipeline.

## License

Add a license file if you want to publish or share this repository publicly.
