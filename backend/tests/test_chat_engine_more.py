import pytest


@pytest.mark.asyncio
async def test_chat_engine_history_and_formatting(monkeypatch):
    from app.db.session import SessionLocal, get_engine
    from app.services.chat_engine import ChatEngine
    from app.models.conversation import Conversation, Message

    db = SessionLocal(bind=get_engine())
    try:
        conv = Conversation(title="t")
        db.add(conv)
        db.commit(); db.refresh(conv)

        # Insert a bunch of messages to trigger history trimming
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            db.add(Message(conversation_id=conv.id, role=role, content=f"m{i}"))
        db.commit()

        engine = ChatEngine(db)
        hist = await engine._load_conversation_history(conv.id, limit=8)
        assert len(hist) == 8
        assert hist[0]["content"].startswith("m")

        # _search_context ...
        async def _fake_search_similar(query: str, document_id=None, k=5):
            return [{"document_id": 1, "page_number": 2, "content": "x"}]

        engine.vector_store.search_similar = _fake_search_similar  # type: ignore
        ctx = await engine._search_context("q", document_id=None, k=1)
        assert isinstance(ctx, list) and ctx

        sources = engine._format_sources(
            context=ctx,
            media={"images": [{"id": 1, "type": "image", "url": "/uploads/images/a.png"}], "tables": []},
        )
        assert any(s.get("type") == "image" for s in sources)
        assert any(s.get("type") == "text" for s in sources)
    finally:
        db.close()
