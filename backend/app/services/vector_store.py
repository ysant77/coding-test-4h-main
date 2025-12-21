"""app.services.vector_store

Vector store service using PostgreSQL + pgvector.

This module provides:
- Embedding generation (OpenAI if configured, otherwise local sentence-transformers)
- Storage of document chunks with embeddings
- Similarity search (cosine distance)

Design goals:
- Keep the API small and testable
- Be resilient when OpenAI isn't configured
- Keep embeddings dimension fixed to the DB schema (Vector(1536))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import DocumentChunk


_ST_MODEL = None  # process-level cache

def _to_pgvector_literal(vec: np.ndarray) -> str:
    """Convert a vector to pgvector literal string, e.g. [0.1,0.2,...]."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    # keep it compact but deterministic
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"

def _pad_or_truncate(vec: np.ndarray, dim: int) -> np.ndarray:
    """Force embeddings to a fixed dimension.

    The DB schema uses Vector(1536). Some local embedding models produce
    different dimensions; we pad with zeros or truncate.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] == dim:
        return vec
    if vec.shape[0] > dim:
        return vec[:dim]
    out = np.zeros((dim,), dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


class VectorStore:
    """Vector store for document embeddings and similarity search."""

    def __init__(self, db: Session):
        self.db = db
        self._ensure_extension()

    def _ensure_extension(self) -> None:
        """Ensure pgvector extension is enabled."""
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception:
            self.db.rollback()

    async def generate_embedding(self, content: str) -> np.ndarray:
        """Generate an embedding for the provided content.

        Preference order:
        1) OpenAI embeddings (if OPENAI_API_KEY is configured)
        2) Local sentence-transformers (all-MiniLM-L6-v2) as fallback
        """

        content = (content or "").strip()
        if not content:
            return np.zeros((settings.EMBEDDING_DIMENSION,), dtype=np.float32)

        if settings.OPENAI_API_KEY:
            # Lazy import to keep local-only setups lightweight
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=content,
            )
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            return _pad_or_truncate(vec, settings.EMBEDDING_DIMENSION)

        # Local fallback
        global _ST_MODEL
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        vec = _ST_MODEL.encode(content, normalize_embeddings=True)
        return _pad_or_truncate(np.asarray(vec, dtype=np.float32), settings.EMBEDDING_DIMENSION)

    # --- Storage API (README names) -------------------------------------------------

    async def store_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int) -> int:
        """Store a list of chunk dicts (content/page_number/chunk_index/metadata).

        Returns number of stored chunks.
        """
        stored = 0
        for ch in chunks:
            content = ch.get("content") or ""
            page_number = int(ch.get("page_number") or 0) or None
            chunk_index = int(ch.get("chunk_index") or stored)
            metadata = ch.get("metadata") or {}
            await self.store_chunk(
                content=content,
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
                metadata=metadata,
            )
            stored += 1
        return stored

    async def search_similar(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to query."""
        return await self.similarity_search(query=query, document_id=document_id, k=k)

    # --- Internal helpers (skeleton names) ------------------------------------------

    async def store_chunk(
        self,
        content: str,
        document_id: int,
        page_number: Optional[int],
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentChunk:
        """Store a single text chunk with its embedding."""
        emb = await self.generate_embedding(content)
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            embedding=emb.tolist(),
            page_number=page_number,
            chunk_index=chunk_index,
            meta=metadata or {},
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk

    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Similarity search using pgvector cosine distance."""
        k = int(k or settings.TOP_K_RESULTS)
        k = max(1, min(k, 50))

        q_emb = await self.generate_embedding(query)
        q_lit = _to_pgvector_literal(q_emb)
        # Use SQL for predictable performance and simple score calculation.
        # <=> is cosine distance in pgvector. Similarity = 1 - distance.
        base_sql = """
                SELECT
                    id,
                    content,
                    page_number,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> (:q)::vector) AS similarity
                FROM document_chunks
            """
        where = ""
        params: Dict[str, Any] = {"q": q_lit, "k": k}
        if document_id is not None:
            where = "WHERE document_id = :doc_id"
            params["doc_id"] = int(document_id)

        sql = text(
            base_sql
            + "\n"
            + where
            + "\nORDER BY embedding <=> (:q)::vector\nLIMIT :k"
        )
        rows = self.db.execute(sql, params).mappings().all()

        results: List[Dict[str, Any]] = []
        for r in rows:
            md = r.get("metadata") or {}
            results.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "score": float(r.get("similarity") or 0.0),
                    "page_number": r.get("page_number"),
                    "chunk_index": r.get("chunk_index"),
                    "metadata": md,
                    "related_images": md.get("related_images", []),
                    "related_tables": md.get("related_tables", []),
                }
            )
        return results

    async def get_related_content(self, chunk_ids: List[int]) -> Dict[str, List[Dict[str, Any]]]:
        """Return merged related image/table ids from chunk metadata.

        The actual DB fetch is handled by ChatEngine to keep this service DB-light.
        """
        if not chunk_ids:
            return {"images": [], "tables": []}

        q = self.db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
        image_ids: List[int] = []
        table_ids: List[int] = []
        for ch in q:
            md = ch.meta or {}
            image_ids.extend(md.get("related_images", []) or [])
            table_ids.extend(md.get("related_tables", []) or [])

        # De-dupe while keeping order
        def _uniq(xs: List[int]) -> List[int]:
            seen = set()
            out = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        return {"images": [{"id": i} for i in _uniq(image_ids)], "tables": [{"id": t} for t in _uniq(table_ids)]}
