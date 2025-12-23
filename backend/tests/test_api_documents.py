import io

import pytest


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(client):
    file = ("note.txt", b"hello", "text/plain")
    r = client.post("/api/documents/upload", files={"file": file})
    assert r.status_code == 400
    assert "Only PDF" in r.json()["detail"]


@pytest.mark.asyncio
async def test_upload_pdf_queues_background_processing(client, monkeypatch):
    """We monkeypatch DocumentProcessor.process_document so Docling isn't required."""

    from app.services.document_processor import DocumentProcessor

    async def _noop(self, file_path: str, document_id: int):
        return None

    monkeypatch.setattr(DocumentProcessor, "process_document", _noop)

    # Minimal PDF header bytes (good enough for upload validation)
    fake_pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    file = ("paper.pdf", fake_pdf, "application/pdf")
    r = client.post("/api/documents/upload", files={"file": file})
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["filename"] == "paper.pdf"

    # list documents
    r = client.get("/api/documents")
    assert r.status_code == 200
    out = r.json()
    assert out["total"] >= 1

    # get document
    doc_id = data["id"]
    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    det = r.json()
    assert det["id"] == doc_id
    assert det["filename"] == "paper.pdf"

    # delete document (should succeed even if media lists are empty)
    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert "deleted" in r.json()["message"].lower()
