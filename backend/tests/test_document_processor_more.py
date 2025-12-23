import pytest


@pytest.mark.asyncio
async def test_document_processor_helpers(monkeypatch):
    from app.db.session import SessionLocal, get_engine
    from app.services.document_processor import DocumentProcessor

    db = SessionLocal(bind=get_engine())
    try:
        dp = DocumentProcessor(db)

        # _collect_items should return list-like attributes
        class O:
            def __init__(self):
                self.tables = [1, 2]
                self.pictures = ["a"]

        out = dp._collect_items(O(), ["tables", "pictures"])  # prefers first attr
        assert out == [1, 2]

        # _norm_page normalizes weird values to >0
        assert dp._norm_page(None) == 0
        assert dp._norm_page(-3) == 0
        assert dp._norm_page(1) == 1

        # _infer_total_pages uses extracted text if doc doesn't expose pages
        extracted = [type("T", (), {"page_number": 1})(), type("T", (), {"page_number": 3})()]
        assert dp._infer_total_pages(doc=None, extracted=extracted) >= 3

        # _render_table_image returns a PIL Image for basic table rows
        img = dp._render_table_image({"rows": [["A", "B"], ["1", "2"]]}, caption="cap")
        assert img.size[0] > 10 and img.size[1] > 10

        # _safe_dict returns dict or None
        assert dp._safe_dict({"a": 1}) == {"a": 1}
        assert dp._safe_dict("x") is None
    finally:
        db.close()
