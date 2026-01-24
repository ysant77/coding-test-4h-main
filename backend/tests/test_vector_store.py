import os
import numpy as np
import pytest

from app.services.vector_store import VectorStore, _hashing_embedding


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed += 1


@pytest.mark.asyncio
async def test_hashing_embedding_is_deterministic_and_normalized():
    v1 = _hashing_embedding("Hello world", dim=64)
    v2 = _hashing_embedding("Hello world", dim=64)
    assert v1.shape == (64,)
    assert np.allclose(v1, v2)
    # L2 norm ~ 1 (or 0 if empty)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_generate_embedding_uses_local_fallback_when_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = VectorStore(FakeSession())
    vec = await store.generate_embedding("Some text")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] > 0
    # should be float32
    assert vec.dtype == np.float32


@pytest.mark.asyncio
async def test_store_text_chunks_adds_rows_and_commits(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db = FakeSession()
    store = VectorStore(db)

    chunks = [
        {"content": "A", "page_number": 1, "chunk_index": 0, "metadata": {"type": "text"}},
        {"content": "B", "page_number": None, "chunk_index": 1, "metadata": {"type": "text"}},
        {"content": "   ", "page_number": 2, "chunk_index": 2, "metadata": {}},  # ignored
    ]
    n = await store.store_text_chunks(chunks, document_id=123)
    assert n == 2
    assert len(db.added) == 2
    assert db.committed == 1
