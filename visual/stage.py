"""
ColPali as one more retriever inside RAG∞ Pro's evidence fusion.

For each question:
  1. ColPali ranks the pages of the active documents by their IMAGES.
  2. Every specialist result (text chunk, table, row, equation, figure) that
     sits on a top visual page gets a "visual" rank, so Reciprocal Rank
     Fusion boosts it; evidence the text retrievers already agree on wins.
  3. A top visual page that no text retriever reached, or a scanned page,
     is added as PAGE evidence (at most 2), so it can be cited and read by
     the vision model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from config.settings import Settings, get_logger
from rag.models import DocumentIndex, Evidence
from storage.cache import DocumentPaths
from visual import colpali

log = get_logger("visual.stage")
MAX_PAGE_EVIDENCE = 2  # page-level items per question (they are long; chunks are more precise)


@dataclass
class VisualOutcome:
    evidence: List[Evidence] = field(default_factory=list)
    ranking: List[Tuple[str, int]] = field(default_factory=list)  # (doc_id, page), best first
    stats: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def page_text(index: DocumentIndex, page: int, limit: int) -> str:
    seen, parts = set(), []
    for n in index.nodes:
        body = n.raw_content.strip()
        if n.page == page and n.type in ("text", "table") and body and body not in seen:
            seen.add(body)
            parts.append(body)
    return "\n".join(parts)[:limit]


def image_heavy_pages(index: DocumentIndex) -> set:
    scanned = set((index.doc_meta or {}).get("scanned_pages") or [])
    return scanned | {f.page for f in index.figures}


def page_evidence(index: DocumentIndex, page: int, rank: int, settings: Settings) -> Evidence:
    text = page_text(index, page, settings.max_chars_per_chunk)
    body = text or "(image-only page: no text layer; the vision model can read the page image)"
    return Evidence(key=f"{index.doc_id}#vpage{page}", doc_id=index.doc_id, doc_name=index.filename, type="page",
                    page=page, section=f"Page {page}", agent="visual_agent", item_id=f"PAGE_{page}",
                    content=f"PAGE_{page} (page {page}, matched by visual page search):\n{body}",
                    excerpt=f"Page {page} image (visual match). {text[:300]}".strip(),
                    meta={"page_image": DocumentPaths(settings.data_dir, index.doc_id).page_image(page).name},
                    ranks={"visual": rank})


def run(query: str, indexes: Sequence[DocumentIndex], settings: Settings,
        groups: Sequence[Sequence[Evidence]]) -> VisualOutcome:
    out = VisualOutcome()
    st = colpali.status(settings)
    if not st.enabled or not indexes:
        return out
    t0 = time.perf_counter()
    hits: List[Tuple[float, DocumentIndex, int]] = []
    failures = 0
    for index in indexes:
        try:
            for page, score in colpali.search(index.doc_id, query, settings, settings.visual_top_k):
                hits.append((score, index, page))
        except Exception as exc:  # visual search is an enhancement: never break retrieval
            log.exception("visual search failed on %s", index.filename)
            failures += 1
            out.warnings.append(f"Visual page search failed ({type(exc).__name__}); text search was used.")
    hits.sort(key=lambda h: -h[0])
    hits = hits[: settings.visual_top_k]
    rank_of = {(ix.doc_id, page): r + 1 for r, (_, ix, page) in enumerate(hits)}
    out.ranking = [(ix.doc_id, page) for _, ix, page in hits]

    covered = set()
    for group in groups:
        for ev in group:
            r = rank_of.get((ev.doc_id, ev.page))
            if ev.type != "metadata":
                covered.add((ev.doc_id, ev.page))
            if r is not None and ev.type != "metadata":
                ev.ranks["visual"] = min(r, ev.ranks.get("visual", r))
    # Only real gaps become PAGE evidence: pages no text retriever reached, and scanned pages
    # (their text layer is empty). Figure pages are covered by the figure agent + vision reader.
    for _, ix, page in hits:
        scanned = page in set((ix.doc_meta or {}).get("scanned_pages") or [])
        if ((ix.doc_id, page) not in covered or scanned) and len(out.evidence) < MAX_PAGE_EVIDENCE:
            out.evidence.append(page_evidence(ix, page, rank_of[(ix.doc_id, page)], settings))

    out.stats = {"calls": len(indexes), "results": len(hits), "failures": failures,
                 "latency_ms": (time.perf_counter() - t0) * 1000, "pages_added": len(out.evidence)}
    return out
