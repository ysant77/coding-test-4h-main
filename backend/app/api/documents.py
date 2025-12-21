"""
Document management API endpoints
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.models.document import Document
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
from app.db.session import SessionLocal
import os
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for processing
    
    This endpoint:
    1. Saves the uploaded file
    2. Creates a document record
    3. Triggers background processing (Docling extraction)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {settings.MAX_FILE_SIZE / 1024 / 1024}MB limit")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, "documents", unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create document record
    document = Document(
        filename=file.filename,
        file_path=file_path,
        processing_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Trigger background processing using FastAPI BackgroundTasks.
    # IMPORTANT: do NOT reuse the request-scoped DB session in a background task.
    # We create a new SessionLocal inside the task.
    async def _process_in_background(doc_id: int, path: str):
        task_db = SessionLocal()
        try:
            processor = DocumentProcessor(task_db)
            await processor.process_document(file_path=path, document_id=doc_id)
        finally:
            task_db.close()

    if background_tasks is not None:
        background_tasks.add_task(_process_in_background, document.id, file_path)
    else:
        # Fallback (shouldn't happen in normal FastAPI usage)
        processor = DocumentProcessor(db)
        await processor.process_document(file_path=file_path, document_id=document.id)
    
    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.processing_status,
        "message": "Document uploaded successfully. Processing will begin shortly."
    }


@router.get("")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get list of all documents
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date,
                "status": doc.processing_status,
                "total_pages": doc.total_pages,
                "text_chunks": doc.text_chunks_count,
                "images": doc.images_count,
                "tables": doc.tables_count
            }
            for doc in documents
        ],
        "total": db.query(Document).count()
    }


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document details including extracted images and tables
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "upload_date": document.upload_date,
        "status": document.processing_status,
        "error_message": document.error_message,
        "total_pages": document.total_pages,
        "text_chunks": document.text_chunks_count,
        "images": [
            {
                "id": img.id,
                "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                "page": img.page_number,
                "caption": img.caption,
                "width": img.width,
                "height": img.height
            }
            for img in document.images
        ],
        "tables": [
            {
                "id": tbl.id,
                "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                "page": tbl.page_number,
                "caption": tbl.caption,
                "rows": tbl.rows,
                "columns": tbl.columns,
                "data": tbl.data
            }
            for tbl in document.tables
        ]
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all associated data
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete physical files
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    for img in document.images:
        if os.path.exists(img.file_path):
            os.remove(img.file_path)
    
    for tbl in document.tables:
        if os.path.exists(tbl.image_path):
            os.remove(tbl.image_path)
    
    # Delete database record (cascade will handle related records)
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}
