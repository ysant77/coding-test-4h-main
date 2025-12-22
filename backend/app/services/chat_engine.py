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
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.conversation import Conversation, Message
from app.models.document import DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


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
        context = await self._search_context(message, document_id=document_id, k=settings.TOP_K_RESULTS)
        media = await self._find_related_media(context)
        answer = await self._generate_response(message=message, context=context, history=history, media=media)
        sources = self._format_sources(context=context, media=media)

        return {
            "answer": answer,
            "sources": sources,
            "processing_time": round(time.time() - start, 3),
        }

    async def _load_conversation_history(self, conversation_id: int, limit: int = 5) -> List[Dict[str, str]]:
        conv = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return []

        msgs = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(max(0, int(limit)))
            .all()
        )
        msgs = list(reversed(msgs))
        return [{"role": m.role, "content": m.content} for m in msgs]

    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        return await self.vector_store.search_similar(query=query, document_id=document_id, k=k)

    async def _find_related_media(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        import os

        # pages from retrieved chunks
        pages = []
        for ch in context_chunks:
            p = ch.get("page_number")
            if isinstance(p, int) and p > 0:
                pages.append(p)
        pages = sorted(set(pages))[:6]  # cap pages to avoid huge media pulls

        images = []
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

        tables = []
        if pages:
            rows = self.db.query(DocumentTable).filter(DocumentTable.page_number.in_(pages)).all()
            for r in rows[:8]:
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
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        # Build a compact context block
        ctx_lines: List[str] = []
        for i, ch in enumerate(context[:8], start=1):
            page = ch.get("page_number")
            page_str = str(page) if page else "unknown"
            score = ch.get("score")
            snippet = (ch.get("content") or "").strip()
            snippet = snippet[:1200]
            ctx_lines.append(f"[{i}] (page {page_str}, score {score:.3f}) {snippet}")
        ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no relevant context found)"

        media_hint = []
        if media.get("images"):
            media_hint.append(f"Images available: {len(media['images'])} (refer by page when relevant)")
        table_block = ""
        if media.get("tables"):
            lines = []
            for t in media["tables"][:3]:
                cap = (t.get("caption") or "").strip()
                pg = t.get("page")
                data = t.get("data") or []
                # keep it compact
                preview = data[:8] if isinstance(data, list) else []
                lines.append(f"- Table (page {pg}) {cap}\n  preview: {preview}")
            table_block = "\n\nTables preview:\n" + "\n".join(lines)
        media_hint_block = "\n".join(media_hint) if media_hint else "No images/tables extracted for this context."

        system = (
            "You are a helpful assistant answering questions about an uploaded PDF. "
            "Use the provided context snippets as the source of truth. "
            "If the answer is not in the context, say you don't know. "
            "When useful, mention relevant pages and refer to tables/figures by page."
        )
        

        user = (
            f"User question: {message}\n\n"
            f"Retrieved context:\n{ctx_block}\n\n"
            f"Media info:\n{media_hint_block}\n\n"
            f"{table_block}\n\n"
            "Answer clearly and concisely."
        )

        # Choose LLM provider: OpenAI if configured, otherwise Ollama
        if settings.OPENAI_API_KEY:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            messages = [{"role": "system", "content": system}]
            # include short history (excluding the current user message which is already passed)
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user})

            resp = client.chat.completions.create(model=settings.OPENAI_MODEL, messages=messages)
            return (resp.choices[0].message.content or "").strip()

        # Ollama local LLM
        return await self._ollama_chat(system=system, history=history, user=user)

    async def _ollama_chat(self, system: str, history: List[Dict[str, str]], user: str) -> str:
        # Ollama chat API: POST /api/chat
        url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"

        messages = [{"role": "system", "content": system}]
        for h in history[-6:]:
            role = h.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (h.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user})

        payload = {"model": settings.OLLAMA_MODEL, "messages": messages, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                # { message: { role, content }, ... }
                return (data.get("message", {}) or {}).get("content", "").strip() or ""
        except Exception as e:
            return (
                "LLM is not configured or unreachable. "
                f"If using Ollama, ensure it's running at {settings.OLLAMA_BASE_URL}. "
                f"Error: {e}"
            )

    def _format_sources(self, context: List[Dict[str, Any]], media: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        for chunk in context[:3]:
            sources.append(
                {
                    "type": "text",
                    "content": chunk.get("content"),
                    "page": chunk.get("page_number"),
                    "score": chunk.get("score", 0.0),
                }
            )

        for image in media.get("images", [])[:5]:
            sources.append(
                {
                    "type": "image",
                    "url": image.get("url"),
                    "caption": image.get("caption"),
                    "page": image.get("page"),
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
