# Requirements and Setup Guide

## System Requirements

- **OS**: macOS, Linux, or Windows
- **Docker**: 20.10+ with Docker Compose 2.0+
- **Node.js**: 18.0+ (for local frontend development)
- **Python**: 3.11+
- **Git**: 2.0+

## Backend Dependencies

See `backend/requirements.txt` for complete Python package list.

### Key Backend Packages

```
FastAPI 0.109.0          - Web framework
SQLAlchemy 2.0.25        - ORM
PostgreSQL + pgvector    - Vector database
Docling 1.0.0            - PDF processing
OpenAI 1.10.0            - LLM API
Sentence-transformers    - Embeddings
```

## Frontend Dependencies

See `frontend/package.json` for complete Node.js package list.

### Key Frontend Packages

```
Next.js 14.1.0           - React framework
React 18                 - UI library
TailwindCSS 3.4.0        - Styling
Lucide React 0.309.0     - Icons
TypeScript 5             - Type safety
```

## Installation

### Backend Setup

```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# If you encounter lint errors, ensure node_modules is properly installed
npm install --legacy-peer-deps  # If needed for compatibility
```

### Docker Setup

```bash
# From project root
docker-compose up -d

# This will start:
# - PostgreSQL with pgvector
# - Redis
# - Backend API (FastAPI)
# - Frontend (Next.js)
```

## Environment Configuration

### Backend (.env)

```
DATABASE_URL=postgresql://docuser:docpass@localhost:5432/docdb
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800
EMBEDDING_DIMENSION=1536
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### Frontend

No environment file needed for basic setup. API calls default to `http://localhost:8000`.

## Troubleshooting

### Frontend Lint Errors

If you see TypeScript/ESLint errors in the IDE:

1. Ensure `node_modules` is installed:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

2. Restart IDE/TypeScript server

3. These errors are normal before `npm install` completes

### Backend Import Errors

If Python imports fail:

1. Ensure virtual environment is activated
2. Reinstall requirements:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Docker Issues

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart services
docker-compose restart
```

## Verification

### Backend Health Check

```bash
curl http://localhost:8000/docs
# Should show FastAPI Swagger UI
```

### Frontend Health Check

```bash
# Frontend should be accessible at
http://localhost:3000
```

### Database Connection

```bash
# Connect to PostgreSQL
psql postgresql://docuser:docpass@localhost:5432/docdb

# Check pgvector extension
\dx vector
```

## Development Workflow

### Backend Development

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
# Opens at http://localhost:3000
```

### Running Tests

```bash
cd backend
pytest
```

## Production Deployment

For production deployment, refer to:
- Backend: `backend/Dockerfile`
- Frontend: `frontend/Dockerfile`
- Orchestration: `docker-compose.yml`

Additional deployment guides can be added for specific platforms (Railway, Vercel, AWS, etc.).
