"""
Vision reader: the vision model (e.g. Gemma 3 on local Ollama) reads the most
relevant PAGE IMAGES and writes down what they show for this question.

Its reading becomes ordinary cited evidence ([REF-n: PAGE pX]), so:
  * the master model can stay text-only (Groq gpt-oss, Ollama Cloud gpt-oss);
  * the deterministic verifier still checks every number in the answer
    against the source it cites, including these readings.

When it runs (MULTIMODAL_ANSWER):
  off   never
  on    every question
  auto  only when it can add something the text layer lacks: the question is
        about a figure/graph/diagram, a selected page is scanned or contains a
        figure, or the text retrievers found nothing. (Keeps text questions fast
        and saves quota.)
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from config.settings import Settings, get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from rag.models import DocumentIndex, Evidence
from storage.cache import DocumentPaths
from visual.stage import image_heavy_pages

log = get_logger("visual.reader")

NONE_REPLY = "NONE"
_PROMPT = (
    "You are reading page {page} of the technical document \"{doc}\" (the attached page image).\n"
    "Question: {question}\n\n"
    "Write down ONLY what this page shows that helps answer the question: exact values with units and test "
    "conditions (min/typ/max), relevant table rows with their column names, values read from graphs (mark them "
    "\"approx., read from graph\"), pin numbers and names, equations. Short bullet points. Do not guess and do "
    "not answer from general knowledge. If the page has nothing relevant, reply exactly: " + NONE_REPLY
)
_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_CACHE_MAX = 256
_VISUAL_INTENTS = {"figure"}


@dataclass
class ReadOutcome:
    evidence: List[Evidence] = field(default_factory=list)
    pages_read: int = 0
    failures: int = 0
    latency_ms: float = 0.0
    model: str = ""
    reason: str = ""


def enabled(settings: Settings, router: ModelRouter) -> bool:
    return settings.multimodal_answer != "off" and settings.vision_read_pages > 0 and router.has_role("vision")


def choose_pages(indexes: Sequence[DocumentIndex], visual_ranking: Sequence[Tuple[str, int]],
                 fused: Sequence[Evidence], limit: int) -> List[Tuple[DocumentIndex, int]]:
    """Visual ranking first (ColPali), then the pages of the best fused evidence (figures/tables first)."""
    by_id = {ix.doc_id: ix for ix in indexes}
    order = list(visual_ranking)
    prio = {"figure": 0, "page": 0, "table": 1, "row": 1, "pin": 1}
    for ev in sorted(fused, key=lambda e: prio.get(e.type, 2)):
        if ev.type != "metadata":
            order.append((ev.doc_id, ev.page))
    out: List[Tuple[DocumentIndex, int]] = []
    for doc_id, page in dict.fromkeys(order):
        ix = by_id.get(doc_id)
        if ix is not None and len(out) < limit:
            out.append((ix, page))
    return out


def should_read(settings: Settings, intents: Sequence[str], targets: Sequence[Tuple[DocumentIndex, int]],
                fused: Sequence[Evidence]) -> Tuple[bool, str]:
    if not targets:
        return False, "no page images"
    if settings.multimodal_answer == "on":
        return True, "always on"
    if _VISUAL_INTENTS & set(intents):
        return True, "question about a figure"
    if any(page in image_heavy_pages(ix) for ix, page in targets):
        return True, "page with a figure or scanned content"
    if not [e for e in fused if e.type != "metadata"]:
        return True, "no text evidence"
    return False, "text evidence is enough"


def read(question: str, targets: Sequence[Tuple[DocumentIndex, int]], router: ModelRouter,
         settings: Settings) -> ReadOutcome:
    out = ReadOutcome()
    t0 = time.perf_counter()
    chain = router.chain("vision")
    out.model = chain[0][1] if chain else ""
    for index, page in targets:
        path = DocumentPaths(settings.data_dir, index.doc_id).page_image(page)
        if not path.exists():
            continue
        key = (index.doc_id, page, out.model, " ".join(question.lower().split()))
        text = _CACHE.get(key)
        if text is None:
            prompt = _PROMPT.format(page=page, doc=index.filename, question=question)
            try:
                text = router.chat("vision", [{"role": "user", "content": prompt}], images=[path.read_bytes()],
                                   max_tokens=450).result.content.strip()
            except LLMError as exc:
                log.info("Vision reading of page %s skipped: %s", page, exc.kind)
                out.failures += 1
                continue
            _CACHE[key] = text
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        out.pages_read += 1
        if not text or text.strip().upper().startswith(NONE_REPLY):
            continue
        out.evidence.append(Evidence(
            key=f"{index.doc_id}#vread{page}", doc_id=index.doc_id, doc_name=index.filename, type="page",
            page=page, section=f"Page {page} (image)", agent="vision_reader", item_id=f"PAGE_{page}",
            content=f"Vision reading of page {page} image (model: {out.model}):\n{text[:settings.max_chars_per_chunk]}",
            excerpt=text[:600], meta={"page_image": path.name, "vision": True}, ranks={"vision": 1}))
    out.latency_ms = (time.perf_counter() - t0) * 1000
    return out


def clear_cache() -> None:
    _CACHE.clear()
