"""app.services.chat_engine

Multimodal RAG chat engine.

Core flow:
1) Load recent conversation history
2) Retrieve relevant chunks from VectorStore
3) Fetch related images/tables based on chunk metadata
4) Generate response with an LLM (OpenAI if configured, otherwise Ollama)
5) Return answer + formatted sources
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.conversation import Message
from app.models.document import DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


def _safe_trim(s: str, n: int = 900) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3].rstrip() + "..."


class ChatEngine:
    """Multimodal chat engine with RAG."""

    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)

    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        history = await self._load_conversation_history(conversation_id, limit=8)

        # retrieve context
        context_chunks = await self._search_context(message, document_id=document_id)

        # media from pages
        media = await self._find_related_media(context_chunks)

        answer = await self._generate_response(message=message, history=history, context_chunks=context_chunks, media=media)
        sources = self._format_sources(context_chunks=context_chunks, media=media)

        return {
            "answer": answer,
            "sources": sources,
            "timing": {"total_sec": round(time.time() - start, 3)},
        }

    async def _load_conversation_history(self, conversation_id: int, limit: int = 8) -> List[Dict[str, Any]]:
        q = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(int(limit))
            .all()
        )
        q = list(reversed(q))
        return [{"role": m.role, "content": m.content} for m in q]

    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # final K used by the LLM
        k_final = int(k or settings.TOP_K_RESULTS or 12)
        k_final = max(3, min(k_final, 20))

        # retrieve more candidates first, then rerank
        k_candidates = min(30, max(12, k_final * 3))

        chunks = await self.vector_store.search_similar(query=query, document_id=document_id, k=k_candidates)

        # Optional reranker (if installed + enabled)
        chunks = await self._maybe_rerank(query, chunks, top_n=k_final)

        # final trim
        return chunks[:k_final]

    async def _maybe_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        enable = str(getattr(settings, "ENABLE_RERANKER", "false")).lower() in ("1", "true", "yes", "y")
        if not enable:
            return chunks

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            # reranker not available, fallback silently
            return chunks

        if not chunks:
            return chunks

        model_name = getattr(settings, "RERANKER_MODEL", None) or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            reranker = CrossEncoder(model_name)
        except Exception:
            return chunks

        pairs = [(query, ch.get("content", "")) for ch in chunks]
        scores = reranker.predict(pairs)  # higher is better

        merged = []
        for ch, s in zip(chunks, scores):
            ch2 = dict(ch)
            ch2["_rerank"] = float(s)
            merged.append(ch2)

        merged.sort(key=lambda x: (x.get("_rerank", 0.0), x.get("score", 0.0)), reverse=True)
        return merged[: max(top_n, 1)]

    async def _find_related_media(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        pages: List[int] = []
        for ch in context_chunks:
            p = ch.get("page_number")
            if isinstance(p, int) and p > 0:
                pages.append(p)

        # de-dupe pages
        pages = list(dict.fromkeys(pages))

        images: List[Dict[str, Any]] = []
        if pages:
            rows = self.db.query(DocumentImage).filter(DocumentImage.page_number.in_(pages)).all()
            for r in rows[:8]:
                images.append(
                    {
                        "id": r.id,
                        "url": f"/uploads/images/{os.path.basename(r.file_path)}",
                        "caption": r.caption,
                        "page": r.page_number,
                    }
                )

        tables: List[Dict[str, Any]] = []
        if pages:
            rows = self.db.query(DocumentTable).filter(DocumentTable.page_number.in_(pages)).all()
            for r in rows[:5]:
                tables.append(
                    {
                        "id": r.id,
                        "url": f"/uploads/tables/{os.path.basename(r.image_path)}",
                        "caption": r.caption,
                        "page": r.page_number,
                        "data": r.data,
                    }
                )

        return {"images": images, "tables": tables}

    async def _generate_response(
        self,
        message: str,
        history: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        # Build context block
        if context_chunks:
            ctx_lines = []
            for i, ch in enumerate(context_chunks, start=1):
                pg = ch.get("page_number")
                sc = ch.get("score", 0.0)
                snippet = _safe_trim(ch.get("content", ""), 900)
                ctx_lines.append(f"[{i}] (page={pg}, score={sc:.3f}) {snippet}")
            ctx_block = "\n\n".join(ctx_lines)
        else:
            ctx_block = "NO_CONTEXT_FOUND"

        # Media hint block (lightweight)
        if media.get("images"):
            media_hint_block = "Images available on pages: " + ", ".join(str(x.get("page")) for x in media["images"][:8])
        else:
            media_hint_block = "No images extracted for this context."

        # Optional: include table data (small amount only)
        table_block = ""
        if media.get("tables"):
            t = media["tables"][0]
            table_block = f"Table snippet (page={t.get('page')}): {str(t.get('data'))[:1200]}"

        system = (
            "You are a helpful assistant answering questions about an uploaded PDF.\n"
            "Rules:\n"
            "1) Use ONLY the retrieved context snippets as the source of truth.\n"
            "2) If the answer is not present in the context, say you don't know.\n"
            "3) If the question asks for 'architecture diagram' or a figure, look for figure captions, "
            "keywords like 'Figure', 'architecture', 'model', 'encoder', 'decoder', 'attention', and summarize what the figure describes.\n"
            "4) When useful, cite the most relevant page numbers.\n"
        )

        user = (
            f"User question: {message}\n\n"
            f"Retrieved context:\n{ctx_block}\n\n"
            f"Media info:\n{media_hint_block}\n\n"
            f"{table_block}".strip()
        )

        # Add a compact recent history
        compact_hist = []
        for h in history[-6:]:
            role = h.get("role", "")
            content = _safe_trim(h.get("content", ""), 350)
            compact_hist.append(f"{role}: {content}")
        hist_block = "\n".join(compact_hist)

        prompt = f"{system}\n\nConversation history:\n{hist_block}\n\n{user}"

        # Choose model provider
        if getattr(settings, "OPENAI_API_KEY", None):
            from openai import OpenAI  # lazy import

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=getattr(settings, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Conversation history:\n{hist_block}\n\n{user}"},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()

        # Ollama fallback
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Conversation history:\n{hist_block}\n\n{user}"},
            ],
            "options": {"temperature": 0.2},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # { message: { role: ..., content: ... }, ... }
        return (data.get("message", {}) or {}).get("content", "").strip()

    def _format_sources(self, context_chunks: List[Dict[str, Any]], media: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        for ch in context_chunks[:8]:
            sources.append(
                {
                    "type": "text",
                    "page": ch.get("page_number"),
                    "score": ch.get("score"),
                    "snippet": _safe_trim(ch.get("content", ""), 280),
                }
            )

        for img in media.get("images", [])[:5]:
            sources.append(
                {
                    "type": "image",
                    "url": img.get("url"),
                    "caption": img.get("caption"),
                    "page": img.get("page"),
                }
            )

        for table in media.get("tables", [])[:5]:
            sources.append(
                {
                    "type": "table",
                    "url": table.get("url"),
                    "caption": table.get("caption"),
                    "page": table.get("page"),
                    "data": table.get("data"),
                }
            )

        return sources
