"""
ColPali page retrieval (late interaction over page images).

Each page image becomes a bag of patch vectors (~700-1000 x 128). A question
becomes a few token vectors. Score(page) = sum over question tokens of the
best-matching patch (MaxSim). It reads tables, graphs, pinouts and scanned
pages directly from the image: no OCR, no chunking.

Storage: one compressed NumPy file per (document, ColPali model) in
data/cache/<doc_id>/colpali_<key>.npz (float16). Search is pure NumPy, so
torch is only needed to EMBED pages and questions. Torch and colpali-engine
are optional (requirements-visual.txt); without them visual search is off.

Models (any colpali-engine checkpoint works):
    vidore/colSmol-256M       CPU default, ~0.5 GB download
    vidore/colpali-v1.3       GPU default, ~6 GB
    vidore/colqwen2.5-v0.2    best on dense tables, GPU
"""
from __future__ import annotations

import gc
import importlib.util
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.settings import Settings, get_logger
from storage.cache import DocumentPaths, short_hash

log = get_logger("visual.colpali")

VISUAL_FORMAT = 1
ProgressFn = Optional[Callable[[float, str], None]]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisualStatus:
    enabled: bool
    reason: str = ""
    model: str = ""
    device: str = ""


_FACTORY: Optional[Callable[[Settings], "Embedder"]] = None  # tests / custom embedders
_BROKEN: Dict[str, str] = {}  # model -> why it could not be loaded (not retried every question)
_lock = threading.RLock()


def set_embedder_factory(factory: Optional[Callable[[Settings], "Embedder"]]) -> None:
    """Replace the real ColPali model (tests use a deterministic fake). None restores it."""
    global _FACTORY
    with _lock:
        _FACTORY = factory
        _EMBEDDERS.clear()
        _VECTORS.clear()
        _QUERY_CACHE.clear()
        _BROKEN.clear()


def _deps_installed() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("colpali_engine") is not None


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model(settings: Settings, device: str) -> str:
    if settings.colpali_model and settings.colpali_model != "auto":
        return settings.colpali_model
    return "vidore/colpali-v1.3" if device == "cuda" else "vidore/colSmol-256M"


def status(settings: Settings) -> VisualStatus:
    if settings.visual_retrieval == "off":
        return VisualStatus(False, "turned off (VISUAL_RETRIEVAL=off)")
    if _FACTORY is not None:
        name = getattr(_FACTORY, "model_name", "custom")
        return VisualStatus(True, "", name, "test")
    if not _deps_installed():
        return VisualStatus(False, "not installed (pip install -r requirements-visual.txt)")
    device = _device()
    model = resolve_model(settings, device)
    if model in _BROKEN:
        return VisualStatus(False, _BROKEN[model], model, device)
    return VisualStatus(True, "", model, device)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class Embedder:
    """Interface: page images -> multi-vectors; question -> multi-vector."""

    model_name: str = ""

    def embed_images(self, paths: Sequence[Path]) -> List[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _model_classes(name: str):
    import colpali_engine.models as m

    n = name.lower()
    if "colqwen2.5" in n or "colqwen2_5" in n:
        return m.ColQwen2_5, m.ColQwen2_5_Processor
    if "colqwen2" in n:
        return m.ColQwen2, m.ColQwen2Processor
    if "colsmol" in n or "idefics" in n:
        return m.ColIdefics3, m.ColIdefics3Processor
    return m.ColPali, m.ColPaliProcessor


class ColPaliEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, batch_size: int, cache_dir: str):
        self.model_name, self.device, self.batch_size, self.cache_dir = model_name, device, batch_size, cache_dir
        self._model = None
        self._proc = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        Model, Proc = _model_classes(self.model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        kwargs = {"dtype": dtype, "cache_dir": self.cache_dir}
        if self.device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
            kwargs["attn_implementation"] = "flash_attention_2"
        if self.device == "cpu":
            torch.set_num_threads(max(1, int(os.getenv("COLPALI_THREADS", "2") or 2)))
        log.info("Loading ColPali model %s on %s", self.model_name, self.device)
        try:
            model = Model.from_pretrained(self.model_name, **kwargs)
        except TypeError:  # older transformers: the argument was called torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = Model.from_pretrained(self.model_name, **kwargs)
        self._model = model.to(self.device).eval()
        self._proc = Proc.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    @staticmethod
    def _strip(emb) -> np.ndarray:
        keep = emb.float().norm(dim=-1) > 0  # padding positions are exact zeros
        emb = emb[keep] if bool(keep.any()) else emb
        return emb.float().cpu().numpy()

    def embed_images(self, paths: Sequence[Path]) -> List[np.ndarray]:
        import torch
        from PIL import Image

        self._load()
        out: List[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            imgs = []
            for p in paths[i:i + self.batch_size]:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            with torch.inference_mode():
                batch = self._proc.process_images(imgs).to(self._model.device)
                embs = self._model(**batch)
            out.extend(self._strip(e).astype(np.float16) for e in embs)
            del batch, embs, imgs
            gc.collect()
        return out

    def embed_query(self, query: str) -> np.ndarray:
        import torch

        self._load()
        with torch.inference_mode():
            batch = self._proc.process_queries([query]).to(self._model.device)
            return self._strip(self._model(**batch)[0]).astype(np.float32)


_EMBEDDERS: Dict[str, Embedder] = {}


def get_embedder(settings: Settings) -> Embedder:
    with _lock:
        if _FACTORY is not None:
            key = "factory"
            if key not in _EMBEDDERS:
                _EMBEDDERS[key] = _FACTORY(settings)
            return _EMBEDDERS[key]
        device = _device()
        model = resolve_model(settings, device)
        if model not in _EMBEDDERS:
            _EMBEDDERS.clear()  # keep at most one ColPali model in memory
            cache = os.getenv("HF_HOME") or str(settings.data_dir / "models")
            _EMBEDDERS[model] = ColPaliEmbedder(model, device, settings.colpali_batch, cache)
        return _EMBEDDERS[model]


# ---------------------------------------------------------------------------
# Per-document page vectors
# ---------------------------------------------------------------------------
@dataclass
class PageVectors:
    pages: List[int] = field(default_factory=list)
    vectors: List[np.ndarray] = field(default_factory=list)  # one [n_patches, dim] float16 per page
    _matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _offsets: Optional[np.ndarray] = field(default=None, repr=False)

    def add(self, page: int, vec: np.ndarray) -> None:
        self.pages.append(int(page))
        self.vectors.append(np.asarray(vec, dtype=np.float16))
        self._matrix = self._offsets = None

    def _packed(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            self._matrix = np.concatenate(self.vectors).astype(np.float32) if self.vectors else np.zeros((0, 0), np.float32)
            self._offsets = np.cumsum([0] + [len(v) for v in self.vectors[:-1]]).astype(np.int64)
        return self._matrix, self._offsets

    def search(self, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if not self.vectors:
            return []
        matrix, offsets = self._packed()
        sim = matrix @ np.asarray(q, dtype=np.float32).T            # [all patches, n_query_tokens]
        per_page = np.maximum.reduceat(sim, offsets, axis=0)         # MaxSim per page and query token
        scores = per_page.sum(axis=1)
        order = np.argsort(-scores)[:k]
        return [(self.pages[i], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.array([len(v) for v in self.vectors], dtype=np.int64)
        matrix = np.concatenate(self.vectors).astype(np.float16) if self.vectors else np.zeros((0, 0), np.float16)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(fh, matrix=matrix, lengths=lengths, pages=np.array(self.pages, dtype=np.int64),
                                    version=np.array([VISUAL_FORMAT]))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> Optional["PageVectors"]:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["version"][0]) != VISUAL_FORMAT:
                    return None
                matrix, lengths, pages = data["matrix"], data["lengths"], data["pages"]
        except Exception as exc:
            log.warning("Discarding unreadable ColPali cache (%s)", exc)
            return None
        pv = cls()
        start = 0
        for page, n in zip(pages.tolist(), lengths.tolist()):
            pv.add(page, matrix[start:start + n])
            start += n
        return pv


def visual_key(model: str) -> str:
    return short_hash("colpali", VISUAL_FORMAT, model)


_VECTORS: "OrderedDict[Tuple[str, str, str], PageVectors]" = OrderedDict()  # (data_dir, doc_id, key)
_VECTORS_MAX = 16
_QUERY_CACHE: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()


def _cache_put(key: Tuple[str, str, str], pv: PageVectors) -> None:
    with _lock:
        _VECTORS[key] = pv
        _VECTORS.move_to_end(key)
        while len(_VECTORS) > _VECTORS_MAX:
            _VECTORS.popitem(last=False)


def _vectors(doc_id: str, settings: Settings, model: str) -> Optional[PageVectors]:
    key = (str(settings.data_dir), doc_id, visual_key(model))
    with _lock:
        if key in _VECTORS:
            _VECTORS.move_to_end(key)
            return _VECTORS[key]
    try:
        pv = PageVectors.load(DocumentPaths(settings.data_dir, doc_id).visual_file(key[2]))
    except ValueError:
        return None
    if pv is not None:
        _cache_put(key, pv)
    return pv


def ensure_document(doc_id: str, settings: Settings, progress: ProgressFn = None) -> Tuple[int, str]:
    """Embed every page image that has no ColPali vectors yet.

    Returns (pages indexed, user-facing warning or "")."""
    from visual.pages import page_images

    st = status(settings)
    if not st.enabled:
        return 0, ""
    paths = DocumentPaths(settings.data_dir, doc_id)
    images = page_images(paths)
    images = {p: f for p, f in images.items() if p <= settings.visual_max_pages}
    if not images:
        return 0, ""
    key = visual_key(st.model)
    pv = _vectors(doc_id, settings, st.model) or PageVectors()
    have = set(pv.pages)
    todo = [p for p in images if p not in have]
    if todo:
        try:
            emb = get_embedder(settings)
            step = max(1, settings.colpali_batch)
            for i in range(0, len(todo), step):
                chunk = todo[i:i + step]
                for page, vec in zip(chunk, emb.embed_images([images[p] for p in chunk])):
                    pv.add(page, vec)
                if progress:
                    progress(min(1.0, (i + len(chunk)) / len(todo)),
                             f"Visual index (ColPali) {i + len(chunk)}/{len(todo)} pages")
                if (i // step) % 8 == 7:
                    pv.save(paths.visual_file(key))  # checkpoint: a long run is not lost
            pv.save(paths.visual_file(key))
        except Exception as exc:
            log.exception("ColPali indexing failed")
            if pv.pages:
                pv.save(paths.visual_file(key))
            if not pv.pages:
                with _lock:
                    _BROKEN[st.model] = f"could not load {st.model} ({type(exc).__name__})"
            _cache_put((str(settings.data_dir), doc_id, key), pv)
            return len(pv.pages), ("Visual page search could not index this document "
                                   f"({type(exc).__name__}). Text search still works.")
        log.info("ColPali indexed %d page(s) of %s with %s", len(todo), doc_id, st.model)
    _cache_put((str(settings.data_dir), doc_id, key), pv)
    return len(pv.pages), ""


def indexed_pages(doc_id: str, settings: Settings) -> int:
    st = status(settings)
    if not st.enabled:
        return 0
    pv = _vectors(doc_id, settings, st.model)
    return len(pv.pages) if pv else 0


def query_vector(query: str, settings: Settings, model: str) -> np.ndarray:
    key = (model, query.strip().lower())
    with _lock:
        if key in _QUERY_CACHE:
            _QUERY_CACHE.move_to_end(key)
            return _QUERY_CACHE[key]
    vec = get_embedder(settings).embed_query(query)
    with _lock:
        _QUERY_CACHE[key] = vec
        while len(_QUERY_CACHE) > 256:
            _QUERY_CACHE.popitem(last=False)
    return vec


def search(doc_id: str, query: str, settings: Settings, k: int) -> List[Tuple[int, float]]:
    """Top-k (page, MaxSim score) for one document; [] when visual search is off or not built."""
    st = status(settings)
    if not st.enabled:
        return []
    pv = _vectors(doc_id, settings, st.model)
    if not pv or not pv.pages:
        return []
    return pv.search(query_vector(query, settings, st.model), k)
