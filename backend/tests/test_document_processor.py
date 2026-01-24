import pytest
from app.services.document_processor import DocumentProcessor
from app.core.config import settings

class DummyDB:
    pass

@pytest.mark.asyncio
async def test_chunk_text_uses_overlap(monkeypatch):
    # force small chunks for test speed/predictability
    monkeypatch.setattr(settings, "CHUNK_SIZE", 50)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 10)

    proc = DocumentProcessor(DummyDB())
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    chunks = proc._chunk_text(text, page_number=1)
    # Implementation should produce multiple chunks with overlap.
    assert len(chunks) >= 2

    c0 = chunks[0]["content"]
    c1 = chunks[1]["content"]
    # with overlap=10, the last 10 chars of c0 should be the first 10 chars of c1 (approximately)
    assert c0[-10:] == c1[:10]
