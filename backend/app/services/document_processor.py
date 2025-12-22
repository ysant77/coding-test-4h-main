"""app.services.document_processor

PDF processing using Docling.

Responsibilities:
- Parse a PDF and extract text, images, and tables
- Persist extracted media to disk and DB
- Chunk text and store embeddings via VectorStore
- Update Document processing status and statistics

Notes:
Docling's object model can differ slightly across versions. This
implementation is defensive: it tries multiple attribute names and
falls back to exporting full text if page-level access isn't available.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


@dataclass
class _ExtractedText:
    page_number: Optional[int]
    text: str


class DocumentProcessor:
    """Process PDF documents and extract multimodal content."""

    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)

        os.makedirs(os.path.join(settings.UPLOAD_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(settings.UPLOAD_DIR, "tables"), exist_ok=True)

    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        start = time.time()
        await self._update_document_status(document_id, "processing")

        try:
            doc = self._load_docling_document(file_path)

            # Extract media first so we can reference ids in chunk metadata
            page_to_image_ids = await self._extract_images(doc, document_id)
            page_to_table_ids = await self._extract_tables(doc, document_id)

            # Extract text, chunk, and store embeddings
            extracted_text = self._extract_text(doc)
            chunks: List[Dict[str, Any]] = []
            for item in extracted_text:
                page = item.page_number
                for ch in self._chunk_text(item.text, page_number=page):
                    ch["metadata"] = {
                        **(ch.get("metadata") or {}),
                        "related_images": page_to_image_ids.get(page or 0, []),
                        "related_tables": page_to_table_ids.get(page or 0, []),
                    }
                    chunks.append(ch)

            stored_chunks = await self.vector_store.store_text_chunks(chunks, document_id=document_id)

            # Update document stats
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = "completed"
                document.error_message = None
                document.text_chunks_count = stored_chunks
                document.images_count = sum(len(v) for v in page_to_image_ids.values())
                document.tables_count = sum(len(v) for v in page_to_table_ids.values())
                # total_pages best-effort
                document.total_pages = self._infer_total_pages(doc, extracted_text)
                self.db.commit()

            return {
                "status": "success",
                "text_chunks": stored_chunks,
                "images": sum(len(v) for v in page_to_image_ids.values()),
                "tables": sum(len(v) for v in page_to_table_ids.values()),
                "processing_time": round(time.time() - start, 3),
            }

        except Exception as e:
            await self._update_document_status(document_id, "error", error_message=str(e))
            return {
                "status": "error",
                "text_chunks": 0,
                "images": 0,
                "tables": 0,
                "processing_time": round(time.time() - start, 3),
                "error": str(e),
            }

    # --------------------------- Docling extraction ---------------------------

    def _load_docling_document(self, file_path: str) -> Any:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)

        # Different docling versions return different wrappers
        doc = getattr(result, "document", None) or getattr(result, "doc", None) or result
        return doc

    def _extract_text(self, doc: Any) -> List[_ExtractedText]:
        # Best effort: page-level text if available
        pages = getattr(doc, "pages", None)
        out: List[_ExtractedText] = []

        if pages:
            for i, p in enumerate(pages, start=1):
                txt = getattr(p, "text", None)
                if callable(getattr(p, "export_to_text", None)):
                    txt = p.export_to_text()
                if txt is None and callable(getattr(p, "to_text", None)):
                    txt = p.to_text()
                if not txt:
                    continue
                out.append(_ExtractedText(page_number=int(getattr(p, "page_number", i) or i), text=str(txt)))

        if out:
            return out

        # Fallback to full-document text
        if callable(getattr(doc, "export_to_text", None)):
            full = doc.export_to_text()
        elif callable(getattr(doc, "export_to_markdown", None)):
            full = doc.export_to_markdown()
        else:
            full = getattr(doc, "text", "") or ""

        return [_ExtractedText(page_number=None, text=str(full))]

    async def _extract_images(self, doc: Any, document_id: int) -> Dict[int, List[int]]:
        page_to_ids: Dict[int, List[int]] = {}

        images = (
            getattr(doc, "images", None)
            or getattr(doc, "pictures", None)
            or getattr(doc, "figures", None)
            or []
        )

        for idx, img in enumerate(images):
            page = int(getattr(img, "page_number", None) or getattr(img, "page", None) or 0)
            pil = self._to_pil_image(img)
            if pil is None:
                continue

            # Save
            filename = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(settings.UPLOAD_DIR, "images", filename)
            pil.save(out_path, format="PNG")

            rec = DocumentImage(
                document_id=document_id,
                file_path=out_path,
                page_number=page or None,
                caption=getattr(img, "caption", None),
                width=int(pil.width),
                height=int(pil.height),
                meta=self._safe_dict(getattr(img, "metadata", None)) or {"index": idx},
            )
            self.db.add(rec)
            self.db.commit()
            self.db.refresh(rec)

            page_to_ids.setdefault(page, []).append(rec.id)

        return page_to_ids

    async def _extract_tables(self, doc: Any, document_id: int) -> Dict[int, List[int]]:
        page_to_ids: Dict[int, List[int]] = {}
        tables = getattr(doc, "tables", None) or getattr(doc, "table", None) or []

        for idx, tbl in enumerate(tables):
            page = getattr(tbl, "page_number", None) or getattr(tbl, "page", None)
            page = int(page) if page is not None else None

            structured = self._table_to_structured(tbl)
            img = self._render_table_image(structured or {"rows": []}, caption=getattr(tbl, "caption", None))

            filename = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(settings.UPLOAD_DIR, "tables", filename)
            img.save(out_path, format="PNG")

            rows = len(structured.get("rows", [])) if structured else None
            cols = 0
            if structured and structured.get("rows"):
                cols = max(len(r) for r in structured["rows"])

            rec = DocumentTable(
                document_id=document_id,
                image_path=out_path,
                data=structured,
                page_number=page,
                caption=getattr(tbl, "caption", None),
                rows=rows,
                columns=cols or None,
                meta=self._safe_dict(getattr(tbl, "metadata", None)) or {"index": idx},
            )
            self.db.add(rec)
            self.db.commit()
            self.db.refresh(rec)

            page_to_ids.setdefault(page, []).append(rec.id)

        return page_to_ids

    # ------------------------------ Chunking ----------------------------------

    def _chunk_text(self, text: str, page_number: Optional[int]) -> List[Dict[str, Any]]:
        """Chunk text with overlap in a simple, robust way."""

        text = (text or "").strip()
        if not text:
            return []

        size = max(200, int(settings.CHUNK_SIZE))
        overlap = max(0, min(int(settings.CHUNK_OVERLAP), size - 1))

        chunks: List[Dict[str, Any]] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "content": chunk,
                        "page_number": page_number,
                        "chunk_index": idx,
                        "metadata": {},
                    }
                )
                idx += 1
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    # ------------------------------ Utils -------------------------------------

    def _to_pil_image(self, img_obj: Any) -> Optional[Image.Image]:
        # Common patterns in Docling
        for attr in ("pil_image", "image", "value"):
            v = getattr(img_obj, attr, None)
            if isinstance(v, Image.Image):
                return v

        # bytes-like
        raw = getattr(img_obj, "data", None) or getattr(img_obj, "bytes", None)
        if isinstance(raw, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                return None

        # conversion method
        for fn in ("to_pil", "to_pil_image"):
            f = getattr(img_obj, fn, None)
            if callable(f):
                try:
                    out = f()
                    if isinstance(out, Image.Image):
                        return out
                except Exception:
                    pass
        return None

    def _table_to_structured(self, tbl: Any) -> Optional[Dict[str, Any]]:
        """
        Convert a Docling table object into a simple structured dict:
        {"rows": [[...], [...], ...]}
        """
        if tbl is None:
            return None

        # 1) Already structured
        if isinstance(tbl, dict):
            if "rows" in tbl:
                return tbl
            # common alt keys
            if "data" in tbl and isinstance(tbl["data"], list):
                return {"rows": tbl["data"]}
            return tbl

        # 2) Common helpers
        for fn in ("to_dict", "as_dict", "export_to_dict"):
            f = getattr(tbl, fn, None)
            if callable(f):
                try:
                    d = f()
                    if isinstance(d, dict):
                        if "rows" in d:
                            return d
                        if "data" in d and isinstance(d["data"], list):
                            return {"rows": d["data"]}
                        return d
                except Exception:
                    pass

        # 3) Pandas dataframe helpers
        for fn in ("to_pandas", "to_dataframe"):
            f = getattr(tbl, fn, None)
            if callable(f):
                try:
                    df = f()
                    # df.values -> list-of-lists
                    return {"rows": df.astype(str).values.tolist()}
                except Exception:
                    pass

        # 4) Sometimes it's exposed as attributes
        for attr in ("df", "dataframe"):
            df = getattr(tbl, attr, None)
            if df is not None:
                try:
                    return {"rows": df.astype(str).values.tolist()}
                except Exception:
                    pass

        # 5) Cell-based tables (row/col indexed cells)
        cells = getattr(tbl, "cells", None)
        if cells and isinstance(cells, list):
            try:
                # Expect each cell has row/col and text/value
                tmp = {}
                max_r = 0
                max_c = 0
                for c in cells:
                    r = int(getattr(c, "row", getattr(c, "row_idx", 0)) or 0)
                    k = int(getattr(c, "col", getattr(c, "col_idx", 0)) or 0)
                    v = getattr(c, "text", None) or getattr(c, "value", None) or ""
                    tmp[(r, k)] = str(v)
                    max_r = max(max_r, r)
                    max_c = max(max_c, k)

                rows = []
                for r in range(max_r + 1):
                    row = []
                    for k in range(max_c + 1):
                        row.append(tmp.get((r, k), ""))
                    rows.append(row)
                return {"rows": rows}
            except Exception:
                pass

        # 6) Last fallback: attribute `data`
        data = getattr(tbl, "data", None)
        if isinstance(data, dict):
            if "rows" in data:
                return data
            if "data" in data and isinstance(data["data"], list):
                return {"rows": data["data"]}
            return data
        if isinstance(data, list):
            return {"rows": data}

        return None


    def _render_table_image(self, structured: Dict[str, Any], caption: Optional[str]) -> Image.Image:
        rows = structured.get("rows") or structured.get("data") or []
        if isinstance(rows, dict):
            # best effort flatten
            rows = [[str(k), str(v)] for k, v in rows.items()]

        # Create a simple text rendering (good enough for UI display)
        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        lines: List[str] = []
        if caption:
            lines.append(str(caption))
        for r in rows[:50]:  # cap to avoid giant images
            if isinstance(r, (list, tuple)):
                lines.append(" | ".join(str(x) for x in r))
            else:
                lines.append(str(r))

        if not lines:
            lines = ["(empty table)"]

        line_h = 14
        w = 1200
        h = min(2000, max(200, 20 + line_h * (len(lines) + 1)))
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        y = 10
        for ln in lines:
            draw.text((10, y), ln[:2000], fill="black", font=font)
            y += line_h
        return img

    def _infer_total_pages(self, doc: Any, extracted: List[_ExtractedText]) -> int:
        pages = getattr(doc, "pages", None)
        if pages:
            try:
                return int(len(pages))
            except Exception:
                pass
        page_nums = [t.page_number for t in extracted if t.page_number]
        return int(max(page_nums)) if page_nums else 0

    def _safe_dict(self, obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        try:
            return dict(obj)
        except Exception:
            return None

    async def _update_document_status(self, document_id: int, status: str, error_message: str | None = None) -> None:
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return
        document.processing_status = status
        if error_message:
            document.error_message = error_message
        self.db.commit()
