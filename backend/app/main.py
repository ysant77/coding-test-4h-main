"""
Main FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import documents, chat
from app.core.config import settings
from app.db.session import engine
from app.models import document, conversation
import os

# Create database tables
document.Base.metadata.create_all(bind=engine)
conversation.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Multimodal Document Chat System",
    description="PDF document processing and multimodal chat API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{settings.UPLOAD_DIR}/documents", exist_ok=True)
os.makedirs(f"{settings.UPLOAD_DIR}/images", exist_ok=True)
os.makedirs(f"{settings.UPLOAD_DIR}/tables", exist_ok=True)

# Mount static files for serving uploaded images and tables
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])


@app.get("/")
async def root():
    return {
        "message": "Multimodal Document Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
