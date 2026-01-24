import os

import pytest


@pytest.mark.asyncio
async def test_create_conversation_and_http_chat_roundtrip(client, monkeypatch):
    """Covers: conversation creation, ChatEngine.process_message path, and API response."""

    # Patch ChatEngine internals to avoid any external LLM calls.
    from app.services import chat_engine as ce_mod

    async def _fake_search_context(self, query, document_id=None, k=5):
        return [
            {
                "chunk_id": 1,
                "content": "Attention is all you need introduces the Transformer.",
                "page_number": 1,
                "score": None,
                "metadata": {"type": "text"},
            }
        ]

    async def _fake_generate_response(self, message, context, history, media):
        return "Transformer paper summary."

    monkeypatch.setattr(ce_mod.ChatEngine, "_search_context", _fake_search_context)
    monkeypatch.setattr(ce_mod.ChatEngine, "_generate_response", _fake_generate_response)

    # 1) Create conversation
    r = client.post("/api/chat/conversations", json={"title": "Test", "document_id": None})
    assert r.status_code == 200
    conv_id = r.json()["conversation_id"]
    assert isinstance(conv_id, int)

    # 2) Send chat message
    r = client.post("/api/chat", json={"message": "What is the transformer?", "conversation_id": conv_id})
    assert r.status_code == 200
    data = r.json()
    assert data["conversation_id"] == conv_id
    assert data["answer"] == "Transformer paper summary."
    assert isinstance(data["message_id"], int)
    assert isinstance(data["sources"], list)


def test_global_search_endpoint(client, monkeypatch):
    from app.services import vector_store as vs_mod

    async def _fake_search(self, query, document_id=None, k=5):
        return [{"chunk_id": 1, "content": "X", "page_number": 1, "score": None, "metadata": {}}]

    monkeypatch.setattr(vs_mod.VectorStore, "search_similar", _fake_search)

    r = client.get("/api/chat/search", params={"q": "attention", "k": 3})
    assert r.status_code == 200
    out = r.json()
    assert out["query"] == "attention"
    assert out["k"] == 3
    assert len(out["results"]) == 1


def test_websocket_chat_works(client, monkeypatch):
    """Basic websocket roundtrip; WS-first frontend relies on this."""

    from app.services import chat_engine as ce_mod

    async def _fake_search_context(self, query, document_id=None, k=5):
        return []

    async def _fake_generate_response(self, message, context, history, media):
        return "pong"

    monkeypatch.setattr(ce_mod.ChatEngine, "_search_context", _fake_search_context)
    monkeypatch.setattr(ce_mod.ChatEngine, "_generate_response", _fake_generate_response)

    # Create conversation first
    r = client.post("/api/chat/conversations", json={"title": "WS"})
    conv_id = r.json()["conversation_id"]

    with client.websocket_connect(f"/api/chat/ws/chat/{conv_id}") as ws:
        ws.send_json({"message": "ping", "document_id": None})
        msg = ws.receive_json()
        assert msg["type"] == "assistant"
        assert msg["data"]["conversation_id"] == conv_id
        assert msg["data"]["answer"] == "pong"
