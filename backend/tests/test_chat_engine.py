import os

import pytest


@pytest.mark.asyncio
async def test_chat_engine_keyword_media_fallback(client, monkeypatch):
    """Cover the optional keyword-based media routing.

    We create a document with an image captioned 'architecture diagram'.
    The user asks for 'architecture', so the engine should attach the image.
    """

    from app.db.session import SessionLocal, get_engine
    from app.models.document import Document, DocumentImage
    from app.services import chat_engine as ce_mod

    # Patch context search + LLM generation
    async def _fake_search_context(self, query, document_id=None, k=5):
        return [{"chunk_id": 1, "content": "ctx", "page_number": 1, "score": None, "metadata": {}}]

    async def _fake_generate_response(self, message, context, history, media):
        # We assert media is present to ensure branch executed
        assert len(media.get("images") or []) == 1
        return "ok"

    monkeypatch.setattr(ce_mod.ChatEngine, "_search_context", _fake_search_context)
    monkeypatch.setattr(ce_mod.ChatEngine, "_generate_response", _fake_generate_response)

    # Insert document + image
    db = SessionLocal(bind=get_engine())
    try:
        doc = Document(filename="x.pdf", file_path="/tmp/x.pdf", processing_status="completed")
        db.add(doc)
        db.commit(); db.refresh(doc)

        img = DocumentImage(
            document_id=doc.id,
            file_path=os.path.join(os.getenv("UPLOAD_DIR", "./uploads"), "images", "arch.png"),
            page_number=1,
            caption="Architecture diagram",
            width=100,
            height=100,
            metadata={},
        )
        db.add(img)
        db.commit()

        # Create conversation
        r = client.post("/api/chat/conversations", json={"title": "t", "document_id": doc.id})
        conv_id = r.json()["conversation_id"]

        # Ask for architecture -> keyword routing should attach the image
        r = client.post("/api/chat", json={"message": "show architecture", "conversation_id": conv_id, "document_id": doc.id})
        assert r.status_code == 200
        out = r.json()
        assert out["answer"] == "ok"
        # sources should include an image entry
        assert any(s.get("type") == "image" for s in out.get("sources", []))
    finally:
        db.close()
