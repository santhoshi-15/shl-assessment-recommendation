# SHL Assessment Recommendation System (GenAI)

## Overview
This project builds a Retrieval-Augmented Generation (RAG) based system to recommend SHL assessments using natural-language queries. It uses sentence embeddings, FAISS similarity search, and a FastAPI backend.

## Architecture
- Data: SHL public product catalog
- Embeddings: Sentence-Transformers (MiniLM)
- Vector Store: FAISS
- Backend: FastAPI
- Evaluation: Mean Recall@10

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
