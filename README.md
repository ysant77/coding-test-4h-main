# Multimodal RAG — PDF Chat with Text, Images & Tables

A full‑stack coding-test submission that lets you:

- upload one or more PDF documents
- parse **text**, extract **images**, extract **tables**
- index content in **PostgreSQL + pgvector**
- chat with a **RAG** engine that returns answers plus relevant **images/tables**
- continue a **multi‑turn** conversation (conversation history is stored)
- (Bonus) use **WebSocket** for real‑time chat
- (Bonus) run **OCR** (Tesseract) on extracted images to improve recall on scanned PDFs
- (Bonus) run **multi‑document search** across all uploaded docs

> **Screenshots:** see the `testing_ss/` folder (placeholder for UI + API testing screenshots).

---

## Architecture

**Backend:** FastAPI + SQLAlchemy + PostgreSQL/pgvector + Docling  
**Frontend:** Next.js (App Router) + Tailwind CSS

High-level flow:

1. **Upload PDF** → backend stores it under `backend/uploads/documents/`
2. **Background processing** (Docling) extracts:
   - text → chunked and embedded
   - images → saved under `backend/uploads/images/`
   - tables → saved under `backend/uploads/tables/` (CSV/JSON)
3. **Vector search** (pgvector cosine distance) retrieves the most relevant text chunks
4. **Chat engine** builds a RAG prompt, attaches the most relevant media, and generates an answer

---

## Features mapped to evaluation criteria

### Document Processing (15 pts)
- **Docling integration & PDF parsing:** `backend/app/services/document_processor.py`
- **Image extraction & storage:** `DocumentImage` rows + files in `backend/uploads/images/`
- **Table extraction & storage:** `DocumentTable` rows + files in `backend/uploads/tables/`

### Vector Store (10 pts)
- **Embeddings:** `backend/app/services/vector_store.py`
  - Uses OpenAI embeddings if `OPENAI_API_KEY` exists
  - Falls back to a deterministic, dependency‑free hashing embedding when OpenAI is not configured
- **Similarity search:** pgvector cosine distance search with optional document filter

### Chat Engine (15 pts)
- **RAG quality:** combines conversation history + retrieved chunks
- **Multimodal response:** includes relevant images/tables for matched pages
- **Multi‑turn:** conversation + messages stored in DB

### UX/UI (15 pts)
- Intuitive chat view with:
  - message history
  - sources (text + images + tables)
- Document upload/management views with status feedback

### Documentation (10 pts)
- This README provides install/run/feature usage
- Complex logic is documented via docstrings in `services/`
- Swagger/OpenAPI available at `/docs`

### Testing (10 pts)
- Unit tests for core logic in `backend/tests/`
- Coverage config included (`backend/.coveragerc`)
- See **Testing** section for running with coverage

---

## Quickstart (Docker Compose — recommended)

### 1) Prerequisites
- Docker + Docker Compose

### 2) Configure environment
Copy example env files (optional):
```bash
cp .env.example .env
```

Key env vars:
- `DATABASE_URL` (already set for docker compose)
- `OPENAI_API_KEY` (optional) — enables OpenAI embeddings + chat
- `OLLAMA_BASE_URL` (optional) — enables Ollama chat fallback if OpenAI is not set
- `OCR_ENABLED` (optional, default: `true`) — OCR for extracted images

### 3) Run
```bash
docker compose up --build
```

Services:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

---

## Local Development (without Docker)

### Backend
```bash
cd backend
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> If you want OCR locally, you must also have **Tesseract** installed on your OS.

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## API Usage

### Upload a PDF
`POST /api/documents/upload` (multipart/form-data)

### List documents
`GET /api/documents`

### Start a chat (HTTP)
`POST /api/chat`
```json
{
  "message": "Summarize the document",
  "document_id": 1,
  "conversation_id": null
}
```

### Multi-document search (Bonus)
`GET /api/search?q=<query>&k=5`  
Searches across **all** documents (ignores document_id).

### Real-time chat via WebSocket (Bonus)
`WS /api/ws/chat/{conversation_id}`

Client sends:
```json
{ "message": "What does the table say about revenue?", "document_id": 1 }
```

Server replies:
```json
{ "type": "assistant", "data": { "...same schema as POST /api/chat..." } }
```

---

## OCR (Bonus: Advanced PDF processing)

OCR is used to improve recall on PDFs with scanned images / weak text extraction.

- Controlled by env var: `OCR_ENABLED=true|false`
- Runs on extracted images (Tesseract)
- OCR text is appended as extra chunks (tagged `metadata.type="ocr"`)

> In Docker, Tesseract is already installed (see `backend/Dockerfile`).

---

## Project Structure

```text
backend/
  app/
    api/                 # FastAPI endpoints
    core/                # settings/config
    db/                  # session/engine
    models/              # SQLAlchemy models
    services/            # 🔥 critical services (Docling, RAG, VectorStore)
  uploads/
    documents/           # uploaded PDFs
    images/              # extracted images
    tables/              # extracted tables
frontend/
  app/                   # Next.js pages (upload, documents, chat)
testing_ss/              # placeholder for screenshots
```

---

## Testing

### Unit tests
```bash
cd backend
pytest
```

### Coverage
```bash
cd backend
./scripts/run_tests.sh
```

This generates:
- `backend/htmlcov/...` (HTML report)
- `backend/coverage.xml` (CI-friendly)

> Target coverage: **60%+ overall app** (run uses `--cov=app`).

---

## Deployment (Bonus)

This repo is containerized:
- `backend/Dockerfile`
- `frontend/Dockerfile`
- `docker-compose.yml`

Deployment options:
- **Railway**: deploy Postgres + backend container; deploy frontend container separately
- **Render/Fly.io**: similar container-first setup
- **Vercel**: deploy frontend, and deploy backend elsewhere (Railway/Render) with `NEXT_PUBLIC_API_BASE`

---

## Notes & Design Decisions

- **DB session in background tasks:** background document processing uses its own `SessionLocal()` to avoid using a request-scoped session.
- **Local embedding fallback:** deterministic hashing embedding keeps the system functional without OpenAI keys.
- **Multimodal sources:** media is retrieved using page-number metadata from top-matching chunks.
- **Error handling:** failures mark documents as `error` with message; API returns clear HTTP errors.

---

## License
MIT (coding test submission)



## Local setup (Ollama-first)

This project is designed to run fully **locally** using **Ollama** for LLM inference.

### 1) Install Ollama and pull a model
- Install Ollama for your OS (macOS/Windows/Linux).
- Pull a chat-capable model, for example:

```bash
ollama pull llama3.1
```

By default Ollama listens on `http://localhost:11434`.

### 2) Backend environment variables

Create `backend/.env`:

```bash
# --- App ---
DATABASE_URL=sqlite:///./app.db
UPLOAD_DIR=./uploads

# --- Ollama (default path & model) ---
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# --- Optional: OCR (bonus) ---
OCR_ENABLED=true
# If you installed tesseract, pytesseract will use it automatically.
```

> If `OPENAI_API_KEY` is not set, the backend automatically uses Ollama.

### 3) Run the backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open Swagger: `http://localhost:8000/docs`

### 4) Run the frontend

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

Then:

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:3000`

---

## Real-time chat (WebSocket) with REST fallback (Bonus)

The **frontend uses WebSocket by default** for chat for low-latency, real-time interaction:

- WebSocket endpoint: `ws://localhost:8000/api/ws/chat/{conversation_id}`
- REST fallback: `POST http://localhost:8000/api/chat`

Behavior:
1. The frontend creates a conversation (`POST /api/conversations`) on page load.
2. It opens a WebSocket session for that conversation.
3. If the WebSocket fails (blocked network/proxy/etc.), it automatically falls back to REST.

