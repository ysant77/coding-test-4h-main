"""
app.services.document_processor

PDF processing using Docling.

Key fixes for Docling v2.65.0:
- Configure PdfPipelineOptions so picture/page images + table structure are actually generated
- Prefer doc.pictures (and page.pictures) as the main source for extracted images
- Use Docling's official table API: table.export_to_dataframe(doc=doc)
- Normalize page numbering and store it correctly (no more page=null unless truly unknown)
"""

from __future__ import annotations

import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore

import logging

log = logging.getLogger("docling_debug")


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

            page_to_image_ids = await self._extract_images(doc, document_id)
            page_to_table_ids = await self._extract_tables(doc, document_id)

            extracted_text = self._extract_text(doc)

            chunks: List[Dict[str, Any]] = []
            for item in extracted_text:
                page_key = self._norm_page(item.page_number)  # always int >= 0
                for ch in self._chunk_text(item.text, page_number=(page_key if page_key > 0 else None)):
                    ch["metadata"] = {
                        **(ch.get("metadata") or {}),
                        "related_images": page_to_image_ids.get(page_key, []),
                        "related_tables": page_to_table_ids.get(page_key, []),
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
            log.exception("process_document failed: %s", e)
            await self._update_document_status(document_id, "error", error_message=str(e))
            return {
                "status": "error",
                "text_chunks": 0,
                "images": 0,
                "tables": 0,
                "processing_time": round(time.time() - start, 3),
                "error": str(e),
            }

    # --------------------------- Docling load ---------------------------

    def _load_docling_document(self, file_path: str) -> Any:
        """
        Docling MUST be configured to generate images and table structure,
        otherwise doc.pictures/doc.tables will exist but lack usable payloads.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        p = PdfPipelineOptions()
        p.images_scale = 2.0
        p.generate_picture_images = True
        p.generate_page_images = True
        p.do_table_structure = True

        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=p)
            }
        )

        result = converter.convert(file_path)
        doc = getattr(result, "document", None) or getattr(result, "doc", None) or result

        
        try:
            pics = getattr(doc, "pictures", None) or []
            tabs = getattr(doc, "tables", None) or []
            log.info("Docling loaded: pages=%s pictures=%s tables=%s",
                     len(getattr(doc, "pages", []) or []),
                     len(pics),
                     len(tabs))
        except Exception:
            pass

        return doc

    # --------------------------- Text extraction ---------------------------

    def _extract_text(self, doc: Any) -> List[_ExtractedText]:
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
                page_num = getattr(p, "page_number", None) or i
                out.append(_ExtractedText(page_number=int(page_num), text=str(txt)))

        if out:
            return out

        # Fallback full-doc text
        if callable(getattr(doc, "export_to_text", None)):
            full = doc.export_to_text()
        elif callable(getattr(doc, "export_to_markdown", None)):
            full = doc.export_to_markdown()
        else:
            full = getattr(doc, "text", "") or ""

        return [_ExtractedText(page_number=None, text=str(full))]

    # --------------------------- Helpers ---------------------------

    def _iter_doc_pages(self, doc: Any) -> Iterable[Any]:
        pages = getattr(doc, "pages", None)
        if pages and isinstance(pages, list):
            for p in pages:
                yield p

    def _collect_items(self, obj: Any, attr_names: List[str]) -> List[Any]:
        for a in attr_names:
            v = getattr(obj, a, None)
            if v:
                if isinstance(v, list):
                    return v
                try:
                    return list(v)
                except Exception:
                    return []
        return []

    def _norm_page(self, page: Optional[int]) -> int:
        try:
            p = int(page) if page is not None else 0
        except Exception:
            p = 0
        return p if p >= 0 else 0
    
    def _page_from_prov(self, item: Any) -> Optional[int]:
        prov = getattr(item, "prov", None) or getattr(item, "provenance", None)
        if not prov:
            return None

        # prov is a list; try each entry until we find a page-like field
        candidates = prov if isinstance(prov, (list, tuple)) else [prov]

        for pr in candidates:
            # 1) dict-like
            if isinstance(pr, dict):
                for k in ("page_no", "page_num", "page_number", "page", "pageno"):
                    if k in pr and pr[k] is not None:
                        try:
                            return int(pr[k])
                        except Exception:
                            pass

            # 2) object-like
            for attr in ("page_no", "page_num", "page_number", "page", "pageno"):
                v = getattr(pr, attr, None)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass

            # 3) nested common patterns (best-effort)
            for attr in ("page_info", "page"):
                obj = getattr(pr, attr, None)
                if obj is not None:
                    for a2 in ("page_no", "page_num", "page_number", "page", "pageno"):
                        v2 = getattr(obj, a2, None)
                        if v2 is not None:
                            try:
                                return int(v2)
                            except Exception:
                                pass

        return None


    # --------------------------- Image extraction ---------------------------

    async def _extract_images(self, doc: Any, document_id: int) -> Dict[int, List[int]]:
        import os, uuid
        from PIL import Image

        page_to_ids: Dict[int, List[int]] = {}

        pictures = self._collect_items(doc, ["pictures"])
        if not pictures:
            return page_to_ids

        for idx, pic in enumerate(pictures):
            page_no = None
            prov = getattr(pic, "prov", None)
            if prov and isinstance(prov, list):
                pr0 = prov[0]
                page_no = getattr(pr0, "page_no", None)
                try:
                    page_no = int(page_no) if page_no is not None else None
                except Exception:
                    page_no = None

            img_ref = getattr(pic, "image", None)
            pil = getattr(img_ref, "pil_image", None)

            if not isinstance(pil, Image.Image):
                continue

            pil = pil.convert("RGB")

            filename = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(settings.UPLOAD_DIR, "images", filename)
            pil.save(out_path, format="PNG")

            rec = DocumentImage(
                document_id=document_id,
                file_path=out_path,
                page_number=(page_no if page_no is not None else None),
                caption=getattr(pic, "caption", None),
                width=int(pil.width),
                height=int(pil.height),
                meta=self._safe_dict(getattr(pic, "metadata", None)) or {"index": idx},
            )
            self.db.add(rec)
            self.db.commit()
            self.db.refresh(rec)

            page_key = self._norm_page(page_no)
            page_to_ids.setdefault(page_key, []).append(rec.id)

        return page_to_ids




    def _to_pil_image(self, img_obj: Any) -> Optional[Image.Image]:
        
        for attr in ("pil_image", "image", "value"):
            v = getattr(img_obj, attr, None)
            if isinstance(v, Image.Image):
                return v

        
        raw = getattr(img_obj, "data", None) or getattr(img_obj, "bytes", None)
        if isinstance(raw, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                return None

        
        for attr in ("uri", "path", "file_path", "filepath", "filename"):
            p = getattr(img_obj, attr, None)
            if isinstance(p, str) and p:
                if p.startswith("file://"):
                    p = p.replace("file://", "", 1)
                p = os.path.expanduser(p)
                if os.path.exists(p):
                    try:
                        return Image.open(p).convert("RGB")
                    except Exception:
                        pass

        
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

    # --------------------------- Table extraction ---------------------------

    async def _extract_tables(self, doc: Any, document_id: int) -> Dict[int, List[int]]:
        page_to_ids: Dict[int, List[int]] = {}

        tables = self._collect_items(doc, ["tables"])
        if not tables:
            return page_to_ids

        for idx, tbl in enumerate(tables):
            page = self._page_from_prov(tbl)  
            page_key = self._norm_page(page)

            structured = self._table_to_structured_docling(tbl, doc)
            if not structured:
                structured = {"rows": [], "diagnostic": "No structured rows from Docling export_to_dataframe"}

            img = self._render_table_image(structured, caption=getattr(tbl, "caption", None))

            filename = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(settings.UPLOAD_DIR, "tables", filename)
            img.save(out_path, format="PNG")

            rows = len(structured.get("rows", []))
            cols = 0
            if structured.get("rows"):
                cols = max((len(r) if isinstance(r, list) else 1) for r in structured["rows"])

            rec = DocumentTable(
                document_id=document_id,
                image_path=out_path,
                data=structured,
                page_number=(page if page else None),
                caption=getattr(tbl, "caption", None),
                rows=rows,
                columns=(cols or None),
                meta=self._safe_dict(getattr(tbl, "metadata", None)) or {"index": idx},
            )
            self.db.add(rec)
            self.db.commit()
            self.db.refresh(rec)

            page_to_ids.setdefault(page_key, []).append(rec.id)

        return page_to_ids

    def _infer_item_page(self, item: Any) -> Optional[int]:
        """Best-effort page inference for Docling items."""
        for attr in ("page_number", "page", "page_num"):
            v = getattr(item, attr, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass

        
        prov = getattr(item, "prov", None) or getattr(item, "provenance", None)
        if prov is not None:
            
            try:
                prov0 = prov[0] if isinstance(prov, (list, tuple)) and prov else prov
            except Exception:
                prov0 = prov
            for attr in ("page_number", "page", "page_no", "page_num"):
                v = getattr(prov0, attr, None)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass
            
            if isinstance(prov0, dict):
                for k in ("page_number", "page", "page_no", "page_num"):
                    if k in prov0:
                        try:
                            return int(prov0[k])
                        except Exception:
                            pass

        return None


    def _infer_bbox_xyxy(self, item: Any):
        """Try to read bbox as (x0,y0,x1,y1). Returns None if not available."""
        bbox = getattr(item, "bbox", None) or getattr(item, "bounding_box", None) or getattr(item, "box", None)
        if bbox is None:
            return None

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return bbox

        if isinstance(bbox, dict):
            keys = ("x0", "y0", "x1", "y1")
            if all(k in bbox for k in keys):
                return (bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])

        
        try:
            return (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        except Exception:
            return None

    def _table_to_structured_docling(self, tbl: Any, doc: Any) -> Optional[Dict[str, Any]]:
        """
        Use Docling's official table -> dataframe exporter.
        Requires pandas installed in the backend container.
        """
       
        f = getattr(tbl, "export_to_dataframe", None)
        if callable(f):
            try:
                df = f(doc=doc)  
               
                rows = df.astype(str).values.tolist()
                cols = [str(c) for c in getattr(df, "columns", [])]
                return {"columns": cols, "rows": rows}
            except Exception as e:
                log.warning("export_to_dataframe failed: %s", e)

       
        return self._table_to_structured_fallback(tbl)

    def _table_to_structured_fallback(self, tbl: Any) -> Optional[Dict[str, Any]]:
        if tbl is None:
            return None

        if isinstance(tbl, dict):
            if "rows" in tbl:
                return tbl
            if "data" in tbl and isinstance(tbl["data"], list):
                return {"rows": tbl["data"]}
            return tbl

        cells = getattr(tbl, "cells", None)
        if cells and isinstance(cells, list):
            try:
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
        rows = structured.get("rows") or []
        cols = structured.get("columns") or []

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        lines: List[str] = []
        if caption:
            lines.append(str(caption))
        if cols:
            lines.append(" | ".join(str(c) for c in cols))

        for r in rows[:60]:
            if isinstance(r, (list, tuple)):
                lines.append(" | ".join(str(x) for x in r))
            else:
                lines.append(str(r))

        if not lines:
            lines = ["(empty table)"]

        line_h = 14
        w = 1400
        h = min(2400, max(220, 20 + line_h * (len(lines) + 1)))
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)

        y = 10
        for ln in lines:
            draw.text((10, y), ln[:3000], fill="black", font=font)
            y += line_h

        return img

    # ------------------------------ Chunking ----------------------------------

    def _chunk_text(self, text: str, page_number: Optional[int]) -> List[Dict[str, Any]]:
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
