import pytest


@pytest.mark.asyncio
async def test_vector_store_hash_embedding_deterministic():
    from app.db.session import SessionLocal, get_engine
    from app.services.vector_store import _hashing_embedding

    db = SessionLocal(bind=get_engine())
    try:
        e1 = _hashing_embedding("hello", 1536)
        e2 = _hashing_embedding("hello", 1536)
        e3 = _hashing_embedding("world", 1536)
        assert len(e1) == len(e2)
        assert (e1 == e2).all()
        assert (e1 != e3).any()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_vector_store_store_text_chunks_inserts_rows():
    from app.db.session import SessionLocal, get_engine
    from app.services.vector_store import VectorStore
    from app.models.document import Document, DocumentChunk

    db = SessionLocal(bind=get_engine())
    try:
        doc = Document(filename="a.pdf", file_path="/tmp/a.pdf", processing_status="completed")
        db.add(doc)
        db.commit()
        db.refresh(doc)

        vs = VectorStore(db)
        chunks = [
            {"content": "first chunk", "page_number": 1, "chunk_index": 0, "metadata": {"k": "v"}},
            {"content": "second chunk", "page_number": 2, "chunk_index": 1, "metadata": {}},
        ]
        stored = await vs.store_text_chunks(chunks, document_id=doc.id)
        assert stored == 2

        db_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
        assert len(db_chunks) == 2
        assert db_chunks[0].content in {"first chunk", "second chunk"}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_vector_store_generate_embedding_uses_hashing_when_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.db.session import SessionLocal, get_engine
    from app.services.vector_store import VectorStore, _hashing_embedding

    db = SessionLocal(bind=get_engine())
    try:
        vs = VectorStore(db)
        emb = await vs.generate_embedding("hello")
        ref = _hashing_embedding("hello", 1536)
        assert (emb == ref).all()
    finally:
        db.close()
