def test_lazy_engine_and_models_import(app):
    """Sanity check: session.get_engine is initialized and models load."""

    from app.db import session as session_mod

    eng = session_mod.get_engine()
    assert eng is not None
    # Ensure Base has expected tables registered
    table_names = set(session_mod.Base.metadata.tables.keys())
    assert "documents" in table_names
    assert "conversations" in table_names
