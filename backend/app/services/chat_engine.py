from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.conversation import Message
from app.models.document import DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


def _safe_trim(s: str, n: int = 900) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _expand_query(q: str) -> List[str]:
    ql = (q or "").lower()
    out = [q]

    # Target typical academic sections
    if any(k in ql for k in ["main conclusion", "conclusion", "conclude", "summary", "key takeaway"]):
        out.append(q + " conclusion summary key takeaway contribution")
        out.append("conclusion summary contribution " + q)
    if any(k in ql for k in ["main contribution", "contribution", "novelty", "what does this paper propose"]):
        out.append(q + " main contribution novelty propose introduce")
        out.append("introduction contributions " + q)
    if any(k in ql for k in ["architecture", "diagram", "model", "pipeline", "figure"]):
        out.append(q + " figure architecture model encoder decoder attention")
        out.append("figure caption architecture " + q)

    # Always add a lightweight “section anchor” query to bias towards useful text
    out.append(q + " abstract introduction conclusion")
    return list(dict.fromkeys(out))  # dedupe preserve order


class ChatEngine:
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

        # 0) Persist the user message (crucial for reliable multi-turn)
        try:
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=message,
                sources=None,
            )
            self.db.add(user_msg)
            self.db.commit()
        except Exception:
            self.db.rollback()
            # Don't fail the whole request if logging fails

        # 1) Load history AFTER saving user message (so it includes the latest user turn)
        history = await self._load_conversation_history(conversation_id, limit=8)

        # 2) Retrieve text context
        context = await self._search_context(
            message,
            document_id=document_id,
            k=settings.TOP_K_RESULTS,
        )

        # Defensive: ensure it's a list of dict-like chunks
        if context is None:
            context = []
        if isinstance(context, dict):
            # If your _search_context returns {"chunks": [...]} or similar
            context = context.get("chunks", []) or []

        # 2b) Ensure every chunk dict has a "metadata" field (populated from ORM .meta)
        # If your _search_context already maps this correctly, this is harmless.
        for ch in context:
            if "metadata" not in ch or ch["metadata"] is None:
                # backward-compat if someone used "meta" key in dict
                ch["metadata"] = ch.get("meta") or {}

            # Normalize page: never allow 0 to propagate
            p = ch.get("page_number")
            try:
                p = int(p) if p is not None else None
            except Exception:
                p = None
            ch["page_number"] = p if (p and p > 0) else None

        # 3) Media retrieval (IDs-first in _find_related_media is the key fix you applied/should apply)
        media = await self._find_related_media(context)

        # 3b) Extra: keyword routing fallback (helps "show architecture diagram"/"experimental results")
        # This is optional but very effective.
        q = (message or "").lower()
        if document_id is not None:
            try:
                import os

                want_diagram = any(k in q for k in ["architecture", "diagram", "figure", "pipeline", "block diagram"])
                want_results = any(k in q for k in ["experimental results", "results table", "ablation", "benchmark", "table"])

                if want_diagram and not (media.get("images") or []):
                    rows = (
                        self.db.query(DocumentImage)
                        .filter(DocumentImage.document_id == document_id)
                        .filter(DocumentImage.caption.ilike("%arch%") | DocumentImage.caption.ilike("%diagram%") | DocumentImage.caption.ilike("%figure%"))
                        .order_by(DocumentImage.page_number.asc().nullslast(), DocumentImage.id.asc())
                        .limit(6)
                        .all()
                    )
                    media["images"] = [
                        {
                            "id": r.id,
                            "url": f"/uploads/images/{os.path.basename(r.file_path)}",
                            "caption": r.caption,
                            "page": (r.page_number if (r.page_number and r.page_number > 0) else None),
                        }
                        for r in rows
                    ]

                if want_results and not (media.get("tables") or []):
                    rows = (
                        self.db.query(DocumentTable)
                        .filter(DocumentTable.document_id == document_id)
                        .filter(DocumentTable.caption.ilike("%result%") | DocumentTable.caption.ilike("%experiment%") | DocumentTable.caption.ilike("%ablation%") | DocumentTable.caption.ilike("%benchmark%"))
                        .order_by(DocumentTable.page_number.asc().nullslast(), DocumentTable.id.asc())
                        .limit(6)
                        .all()
                    )
                    media["tables"] = [
                        {
                            "id": r.id,
                            "url": f"/uploads/tables/{os.path.basename(r.image_path)}",
                            "caption": r.caption,
                            "page": (r.page_number if (r.page_number and r.page_number > 0) else None),
                            "data": r.data,
                        }
                        for r in rows
                    ]
            except Exception:
                self.db.rollback()
                # ignore fallback failures

        # 4) Generate answer + sources
        answer = await self._generate_response(
            message=message,
            context=context,
            history=history,
            media=media,
        )

        sources = self._format_sources(context=context, media=media)

        # 4b) Persist assistant message with sources for debugging + stable multi-turn
        try:
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources,
            )
            self.db.add(assistant_msg)
            self.db.commit()
        except Exception:
            self.db.rollback()

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
            .limit(int(limit))
            .all()
        )
        q = list(reversed(q))
        return [{"role": m.role, "content": m.content} for m in q]

    async def _search_context(self, query: str, document_id: Optional[int], k: Optional[int] = None) -> List[Dict[str, Any]]:
        k_final = int(k or settings.TOP_K_RESULTS or 16)
        k_final = max(6, min(k_final, 24))

        # Retrieve a good pool per query, then union them
        candidates: List[Dict[str, Any]] = []
        seen = set()

        expanded = _expand_query(query)
        per_query_k = min(25, max(12, k_final))  # each query gives a small pool

        for q in expanded[:4]:  # cap to avoid over-retrieving
            res = await self.vector_store.search_similar(query=q, document_id=document_id, k=per_query_k)
            for ch in res:
                cid = ch.get("id")
                if cid is None or cid in seen:
                    continue
                seen.add(cid)
                candidates.append(ch)

        # Optional rerank
        candidates = await self._maybe_rerank(query, candidates)

        return candidates[:k_final]

    async def _maybe_rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enable = bool(getattr(settings, "ENABLE_RERANKER", False))
        if not enable or not chunks:
            # sort by vector score if no reranker
            return sorted(chunks, key=lambda x: float(x.get("score") or 0.0), reverse=True)

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            return sorted(chunks, key=lambda x: float(x.get("score") or 0.0), reverse=True)

        model_name = getattr(settings, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            reranker = CrossEncoder(model_name)
        except Exception:
            return sorted(chunks, key=lambda x: float(x.get("score") or 0.0), reverse=True)

        pairs = [(query, ch.get("content", "")) for ch in chunks]
        scores = reranker.predict(pairs)  # higher better

        merged = []
        for ch, s in zip(chunks, scores):
            ch2 = dict(ch)
            ch2["_rerank"] = float(s)
            merged.append(ch2)

        merged.sort(key=lambda x: (x.get("_rerank", 0.0), float(x.get("score") or 0.0)), reverse=True)
        return merged

    async def _find_related_media(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        pages = []
        for ch in context_chunks:
            p = ch.get("page_number")
            if isinstance(p, int) and p >= 1:
                pages.append(p)
        pages = list(dict.fromkeys(pages))

        images: List[Dict[str, Any]] = []
        if pages:
            rows = self.db.query(DocumentImage).filter(DocumentImage.page_number.in_(pages)).all()
            for r in rows[:8]:
                images.append({"id": r.id, "url": f"/uploads/images/{os.path.basename(r.file_path)}", "caption": r.caption, "page": r.page_number})

        tables: List[Dict[str, Any]] = []
        if pages:
            rows = self.db.query(DocumentTable).filter(DocumentTable.page_number.in_(pages)).all()
            for r in rows[:5]:
                tables.append({"id": r.id, "url": f"/uploads/tables/{os.path.basename(r.image_path)}", "caption": r.caption, "page": r.page_number, "data": r.data})

        # 2) Fallback: pages from retrieved chunks (coarse)
        if not images and not tables:
            pages: List[int] = []
            for ch in context_chunks:
                p = ch.get("page_number")
                try:
                    p = int(p) if p is not None else None
                except Exception:
                    p = None
                # treat 0 as unknown
                if p and p > 0:
                    pages.append(p)

            pages = sorted(set(pages))[:6]

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
        history: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        # Build context block
        if context_chunks:
            ctx_lines = []
            for i, ch in enumerate(context_chunks, start=1):
                pg = ch.get("page_number")
                sc = float(ch.get("score") or 0.0)
                snippet = _safe_trim(ch.get("content", ""), 900)
                ctx_lines.append(f"[{i}] (page={pg}, score={sc:.3f}) {snippet}")
            ctx_block = "\n\n".join(ctx_lines)
        else:
            ctx_block = "NO_CONTEXT_FOUND"

        compact_hist = []
        for h in history[-6:]:
            role = h.get("role", "")
            content = _safe_trim(h.get("content", ""), 350)
            compact_hist.append(f"{role}: {content}")
        hist_block = "\n".join(compact_hist)

        system = (
            "You answer questions about an uploaded document.\n"
            "Rules:\n"
            "- Use ONLY the retrieved context snippets.\n"
            "- If the answer is not present, say you don't know.\n"
            "- If asked about architecture/diagram/figure, search the context for figure captions and describe what it says.\n"
            "- When possible, mention the most relevant page numbers.\n"
        )

        user = (
            f"Conversation history:\n{hist_block}\n\n"
            f"User question: {message}\n\n"
            f"Retrieved context:\n{ctx_block}\n"
        )

        # OpenAI
        if getattr(settings, "OPENAI_API_KEY", None):
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=getattr(settings, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()

        # Ollama
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.2},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "").strip()

    def _format_sources(self, context_chunks: List[Dict[str, Any]], media: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for ch in context_chunks[:10]:
            sources.append({"type": "text", "page": ch.get("page_number"), "score": ch.get("score"), "snippet": _safe_trim(ch.get("content", ""), 280)})

        for img in media.get("images", [])[:5]:
            sources.append({"type": "image", "url": img.get("url"), "caption": img.get("caption"), "page": img.get("page")})

        for table in media.get("tables", [])[:5]:
            sources.append({"type": "table", "url": table.get("url"), "caption": table.get("caption"), "page": table.get("page"), "data": table.get("data")})

        return sources
