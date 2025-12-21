# 🚀 Quick Start Guide

## Prerequisites
- Docker & Docker Compose
- OpenAI API Key (or Ollama for local LLM)

## Setup (5 minutes)

```bash
# 1. Set environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start services
docker-compose up -d

# 3. Access application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

## What to Implement

### ✅ Core Services (in `backend/app/services/`)

1. **document_processor.py** - Extract text, images, tables from PDF using Docling
2. **vector_store.py** - Store embeddings in pgvector, perform similarity search
3. **chat_engine.py** - RAG implementation with multimodal responses

### ✅ Frontend Features

- Document upload with progress
- Chat interface with image/table display
- Multi-turn conversation support

## Testing

Upload a PDF → View extracted content → Chat about it

Good luck! 🎯
