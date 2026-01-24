"""
Vector store service using PostgreSQL + pgvector.

Improvements vs previous:
- Strong local embeddings by default (BGE small, 384-d)
- No "zero-padding to 1536" hacks (dimension must match DB schema)
- Filters out reference/bibliography chunks by metadata + heuristics
- Similarity search with stable SQL and cosine distance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import DocumentChunk

# Process-level cache for local embedding model
_ST_MODEL = None


def _to_pgvector_literal(vec: np.ndarray) -> str:
    """Convert a numpy vector to pgvector literal string, e.g. [0.1,0.2,0.3]."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{x:.8f}" for x in v.tolist()) + "]"


def _looks_like_references(text_block: str) -> bool:
    """
    Heuristic filter: references/bibliography sections are high-frequency retrieval junk
    for questions like "architecture diagram" / "conclusion".
    """
    t = (text_block or "").strip().lower()
    if not t:
        return False

    head = t[:4000]  # only need first chunk
    triggers = [
        "references",
        "bibliography",
        "works cited",
        "reference",
        "acknowledg",  # acknowledgement(s)
    ]
    if any(x in head[:250] for x in triggers):
        return True

    # Stronger heuristic: lots of citations patterns like [12], [3], et al., year-heavy
    bracket_cites = head.count("[") + head.count("]")
    year_hits = sum(head.count(str(y)) for y in range(1990, 2031))
    etal_hits = head.count("et al")

    # If it's mostly citation soup, mark as references-like
    if bracket_cites >= 30 and (year_hits >= 15 or etal_hits >= 8):
        return True

    return False


class VectorStore:
    """Vector store for document embeddings and similarity search."""

    def __init__(self, db: Session):
        self.db = db
        self._ensure_extension()
    
    async def store_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int) -> int:
        """
        Compatibility method expected by DocumentProcessor.
        It stores the passed chunks with embeddings and returns how many were stored.
        """
        return await self.store_chunks(document_id=document_id, chunks=chunks)


    def _ensure_extension(self) -> None:
        """Ensure pgvector extension is enabled."""
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception:
            self.db.rollback()

    async def generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate embedding for content.

        Preference order:
        1) OpenAI embeddings (if OPENAI_API_KEY configured)
        2) Local sentence-transformers (BGE small by default)
        """
        content = (content or "").strip()
        if not content:
            return np.zeros((settings.EMBEDDING_DIMENSION,), dtype=np.float32)

        # 1) OpenAI
        if getattr(settings, "OPENAI_API_KEY", None):
            from openai import OpenAI  # lazy import

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.embeddings.create(
                model=getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=content,
            )
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            if vec.shape[0] != settings.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"OpenAI embedding dim={vec.shape[0]} but settings.EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}. "
                    "Fix config or DB schema."
                )
            return vec

        # 2) Local
        global _ST_MODEL
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            model_name = getattr(settings, "LOCAL_EMBED_MODEL", None) or "BAAI/bge-small-en-v1.5"
            _ST_MODEL = SentenceTransformer(model_name)

        vec = _ST_MODEL.encode(content, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)

        if vec.shape[0] != settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Local embedding dim={vec.shape[0]} but settings.EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}. "
                "Fix config or DB schema."
            )
        return vec

    async def store_chunk(
        self,
        content: str,
        document_id: int,
        page_number: Optional[int],
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentChunk:
        """Store one chunk with embedding."""
        md = metadata or {}

        # If doc processor didn't mark references, infer here (cheap + helpful)
        if "is_references" not in md:
            md["is_references"] = bool(_looks_like_references(content))

        emb = await self.generate_embedding(content)

        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            embedding=emb.tolist(),
            page_number=page_number,
            chunk_index=chunk_index,
            meta=md,
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk

    async def store_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> int:
        """Store a list of chunks. Returns count stored."""
        stored = 0
        for ch in chunks:
            content = ch.get("content") or ""

            raw_page = ch.get("page_number", None)
            page_number = None
            if raw_page is not None and str(raw_page).strip() != "":
                try:
                    p = int(raw_page)
                    page_number = p if p > 0 else None
                except Exception:
                    page_number = None

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

    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 12,
        include_references: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search using pgvector cosine distance.
        <=> is cosine distance in pgvector. Similarity = 1 - distance.
        """
        k = int(k or settings.TOP_K_RESULTS)
        k = max(1, min(k, 50))

        q_emb = await self.generate_embedding(query)
        q_lit = _to_pgvector_literal(q_emb)

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

        where_clauses: List[str] = []
        params: Dict[str, Any] = {"q": q_lit, "k": k}

        if document_id is not None:
            where_clauses.append("document_id = :doc_id")
            params["doc_id"] = int(document_id)

        if not include_references:
            # metadata JSONB column is commonly called "metadata" at DB level
            # your ORM uses meta=... but select uses metadata, matching your existing SQL.
            where_clauses.append("(metadata->>'is_references') IS DISTINCT FROM 'true'")

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        sql = text(base_sql + where_sql + " ORDER BY embedding <=> (:q)::vector LIMIT :k")
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
                    "related_images": (md.get("related_images") or []),
                    "related_tables": (md.get("related_tables") or []),
                }
            )
        return results

    async def search_similar(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 12,
    ) -> List[Dict[str, Any]]:
        """Compat wrapper used by ChatEngine."""
        return await self.similarity_search(query=query, document_id=document_id, k=k)

    async def get_related_content(self, chunk_ids: List[int]) -> Dict[str, List[Dict[str, Any]]]:
        """Return merged related image/table ids from chunk metadata."""
        if not chunk_ids:
            return {"images": [], "tables": []}

        q = self.db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
        image_ids: List[int] = []
        table_ids: List[int] = []
        for ch in q:
            md = ch.meta or {}
            image_ids.extend(md.get("related_images", []) or [])
            table_ids.extend(md.get("related_tables", []) or [])

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
