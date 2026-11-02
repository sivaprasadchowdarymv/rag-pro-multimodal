"""
Render PDF pages to JPEG images, once per document.

The images are what ColPali embeds and what the vision model reads. They live
under data/documents/<doc_id>/pages/ so they are synced to the Hugging Face
dataset with the rest of the document and survive restarts (the PDF itself is
still never kept).
"""
from __future__ import annotations

import gc
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

import pymupdf

from config.settings import Settings, get_logger
from storage.cache import DocumentPaths

log = get_logger("visual.pages")

ProgressFn = Optional[Callable[[float, str], None]]
_PAGE_FILE = re.compile(r"^p(\d{4})\.jpg$")
JPEG_QUALITY = 85


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_pages(pdf_bytes: bytes, paths: DocumentPaths, settings: Settings, progress: ProgressFn = None) -> int:
    """Render every page that has no image yet. Returns the number of page images on disk."""
    paths.pages_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = doc.page_count
        done = 0
        for i, page in enumerate(doc):
            target = paths.page_image(i + 1)
            if not target.exists():
                rect = page.rect
                zoom = min(settings.page_dpi / 72.0, settings.page_max_side / max(rect.width, rect.height, 1.0))
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                _write_atomic(target, pix.tobytes("jpg", jpg_quality=JPEG_QUALITY))
                del pix
                done += 1
                if done % 25 == 0:
                    gc.collect()
            if progress:
                progress((i + 1) / total, f"Page images {i + 1}/{total}")
    finally:
        doc.close()
    if done:
        log.info("Rendered %d page image(s) for %s", done, paths.doc_id)
    return len(page_images(paths))


def page_images(paths: DocumentPaths) -> Dict[int, Path]:
    """page number -> image path, for the pages rendered so far."""
    if not paths.pages_dir.is_dir():
        return {}
    out: Dict[int, Path] = {}
    for f in paths.pages_dir.iterdir():
        m = _PAGE_FILE.match(f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def wanted(settings: Settings) -> bool:
    """Render page images only when something will use them (ColPali or a vision model)."""
    from visual import colpali  # local import: avoid a cycle at module load

    if colpali.status(settings).enabled:
        return True
    if settings.multimodal_answer == "off":
        return False
    if settings.multimodal_answer == "on":
        return True
    from llm.model_router import ModelRouter

    return ModelRouter(settings).has_role("vision")
