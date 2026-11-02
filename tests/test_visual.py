"""Multimodal layer: page images, ColPali index + fusion, vision reader, provider fallbacks.

ColPali itself is replaced by a deterministic fake embedder (no torch, no download):
page N's image -> a patch pointing along dimension N; a question about a topic ->
a token pointing at the page that holds that topic. Everything else is the real code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import server_log

DIM = 16
TOPIC_PAGE = {"dissipation": 3, "pin": 4, "characteristics": 2, "features": 1}


class FakeColPali:
    model_name = "fake/colpali"

    def __init__(self):
        self.image_calls = 0
        self.query_calls = 0

    def embed_images(self, paths):
        self.image_calls += len(paths)
        out = []
        for p in paths:
            n = int(Path(p).stem[1:])
            vec = np.zeros((3, DIM), np.float16)
            vec[0, n % DIM] = 1.0
            vec[1:, 0] = 0.05
            out.append(vec)
        return out

    def embed_query(self, query):
        self.query_calls += 1
        page = next((pg for word, pg in TOPIC_PAGE.items() if word in query.lower()), 15)
        q = np.zeros((2, DIM), np.float32)
        q[0, page] = 1.0
        q[1, 0] = 0.1
        return q


@pytest.fixture
def fake_colpali():
    from visual import colpali

    made = {}

    def factory(settings):
        made["e"] = FakeColPali()
        return made["e"]

    factory.model_name = FakeColPali.model_name
    colpali.set_embedder_factory(factory)
    yield made
    colpali.set_embedder_factory(None)


@pytest.fixture
def visual_index(sample_pdf, make_settings, fake_colpali):
    from rag.pipeline import get_index

    settings = make_settings(visual_retrieval="on", multimodal_answer="off")
    index, warnings = get_index(sample_pdf, "LM7805X.pdf", settings)
    assert not warnings
    return index, settings


# --------------------------------------------------------------------- indexing
def test_upload_renders_pages_and_builds_colpali_index(visual_index, fake_colpali):
    from storage.cache import DocumentPaths
    from visual import colpali
    from visual.pages import page_images

    index, settings = visual_index
    paths = DocumentPaths(settings.data_dir, index.doc_id)
    images = page_images(paths)
    assert sorted(images) == [1, 2, 3, 4]
    assert all(f.read_bytes()[:3] == b"\xff\xd8\xff" for f in images.values())  # JPEG
    assert colpali.indexed_pages(index.doc_id, settings) == 4
    assert paths.visual_file(colpali.visual_key("fake/colpali")).exists()
    assert fake_colpali["e"].image_calls == 4


def test_reopening_reuses_saved_vectors(visual_index, fake_colpali):
    from rag import pipeline
    from visual import colpali

    index, settings = visual_index
    pipeline._registry.clear()
    colpali._VECTORS.clear()
    calls_before = fake_colpali["e"].image_calls
    reopened = pipeline.open_index(index.doc_id, index.filename, settings)
    assert reopened is not None
    assert colpali.indexed_pages(index.doc_id, settings) == 4
    assert fake_colpali["e"].image_calls == calls_before  # loaded from disk, not re-embedded


def test_clear_cache_removes_page_images(visual_index):
    from rag.pipeline import clear_cache
    from storage.cache import DocumentPaths

    index, settings = visual_index
    clear_cache(index.doc_id, settings)
    assert not DocumentPaths(settings.data_dir, index.doc_id).pages_dir.exists()


def test_page_vectors_maxsim_and_roundtrip(tmp_path):
    from visual.colpali import PageVectors

    pv = PageVectors()
    pv.add(1, np.array([[1, 0], [0, 0.2]], np.float16))
    pv.add(2, np.array([[0, 1]], np.float16))
    pv.add(7, np.array([[0.6, 0.8], [0.1, 0.1], [0, 0]], np.float16))
    q = np.array([[1, 0], [0, 1]], np.float32)
    # MaxSim: page1 = 1 + 0.2, page2 = 0 + 1, page7 = 0.6 + 0.8
    assert [p for p, _ in pv.search(q, 3)] == [7, 1, 2]
    pv.save(tmp_path / "v.npz")
    again = PageVectors.load(tmp_path / "v.npz")
    assert again.pages == [1, 2, 7]
    assert [p for p, _ in again.search(q, 1)] == [7]


# -------------------------------------------------------------------- retrieval
def test_colpali_rank_joins_fusion(visual_index):
    from agents.orchestrator import plan
    from agents.retrieval_stage import run_retrieval

    index, settings = visual_index
    out = run_retrieval(plan("How is the power dissipation calculated?"), [index], settings)
    assert out.visual_ranking[0] == (index.doc_id, 3)
    assert out.agent_stats["visual_agent"]["results"] >= 1
    assert "visual" in out.latency_ms
    boosted = [e for e in out.fusion.evidence if e.ranks.get("visual") == 1]
    assert boosted and all(e.page == 3 for e in boosted)


def test_uncovered_visual_page_added_as_page_evidence(visual_index):
    from visual import stage

    index, settings = visual_index
    out = stage.run("pin configuration", [index], settings, groups=[])  # no text retriever found anything
    assert len(out.evidence) <= stage.MAX_PAGE_EVIDENCE
    pages = {e.page: e for e in out.evidence}
    assert 4 in pages  # ColPali's best page for the pin question
    ev = pages[4]
    assert ev.type == "page" and ev.item_id == "PAGE_4" and ev.agent == "visual_agent"
    assert ev.meta["page_image"] == "p0004.jpg"
    assert "INPUT | Unregulated input" in ev.content  # the page's pin table, as text


def test_visual_off_without_dependencies(sample_pdf, make_settings, monkeypatch):
    from rag.pipeline import get_index
    from storage.cache import DocumentPaths
    from visual import colpali

    monkeypatch.setattr(colpali, "_deps_installed", lambda: False)
    settings = make_settings()  # VISUAL_RETRIEVAL=auto, Groq only (no vision model)
    st = colpali.status(settings)
    assert not st.enabled and "requirements-visual" in st.reason
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    assert not DocumentPaths(settings.data_dir, index.doc_id).pages_dir.exists()  # nothing rendered for nothing


# ---------------------------------------------------------------- vision reader
def _ollama_settings(make_settings, server, **kw):
    base = dict(llm_providers=("ollama_local",), enable_ollama_local=True, ollama_local_host=server,
                master_model="gpt-oss:20b", visual_retrieval="on", multimodal_answer="auto")
    base.update(kw)
    return make_settings(**base)


def test_vision_reader_reads_page_images_and_is_cited(sample_pdf, make_settings, server, fake_colpali):
    from agents.rag_pro import answer_pro
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index

    settings = _ollama_settings(make_settings, server)
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    result = answer_pro("What does the power dissipation figure show?", [index], settings, ModelRouter(settings))

    assert any(s.text.startswith("Vision reader (gemma3:4b): read 2 page image(s)") for s in result.steps)
    assert ("ollama", "gemma3:4b") in [tuple(r) for r in server_log(server)["requests"]]
    first = result.sources[0]
    assert first.type == "page" and first.agent == "vision_reader" and first.tag == "PAGE"
    assert first.meta["page_image"].startswith("p000")
    from ui.widgets import page_image_path
    shown = page_image_path(settings, first)  # what the Evidence panel displays
    assert shown is not None and shown.exists()
    assert result.agent_stats["vision_reader"]["results"] == 2
    assert "vision" in result.stage_latency
    assert result.verification["verdict"] in ("verified", "partial")
    assert not server_log(server)["errors"]


def test_vision_reader_works_without_colpali(sample_pdf, make_settings, server, monkeypatch):
    """No torch/colpali: pages still rendered for the vision model, chosen from the fused evidence."""
    from agents.rag_pro import answer_pro
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index
    from visual import colpali

    monkeypatch.setattr(colpali, "_deps_installed", lambda: False)
    settings = _ollama_settings(make_settings, server, visual_retrieval="auto")
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    result = answer_pro("Describe Figure 2, the functional block diagram", [index], settings, ModelRouter(settings))
    assert "visual_agent" not in result.agent_stats
    assert result.agent_stats["vision_reader"]["calls"] >= 1


def test_should_read_policy(visual_index):
    from visual.reader import should_read

    index, settings = visual_index
    from dataclasses import replace

    from rag.models import Evidence
    text_ev = [Evidence(key="k", doc_id=index.doc_id, doc_name="d", type="text", page=2, section="", content="x",
                        excerpt="x", agent="text_agent")]
    auto = replace(settings, multimodal_answer="auto")
    assert should_read(auto, ["text", "table"], [(index, 2)], text_ev)[0] is False  # text page, text found
    assert should_read(auto, ["text"], [(index, 3)], text_ev)[0] is True            # page 3 holds a figure
    assert should_read(auto, ["figure"], [(index, 2)], text_ev)[0] is True          # figure question
    assert should_read(auto, ["text"], [(index, 2)], [])[0] is True                  # nothing found by text
    assert should_read(replace(settings, multimodal_answer="on"), ["text"], [(index, 2)], text_ev)[0] is True
    assert should_read(auto, ["figure"], [], text_ev)[0] is False                    # no page images


def test_multimodal_off_never_calls_vision(sample_pdf, make_settings, server, fake_colpali):
    from agents.rag_pro import answer_pro
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index

    settings = _ollama_settings(make_settings, server, multimodal_answer="off")
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    result = answer_pro("What does the power dissipation figure show?", [index], settings, ModelRouter(settings))
    assert "vision_reader" not in result.agent_stats
    assert not any(s.text.startswith("Vision reader") for s in result.steps)
    assert not any(src.agent == "vision_reader" for src in result.sources)


def test_ollama_model_without_tools_still_answers(sample_pdf, make_settings, server):
    """gemma3 rejects tool definitions on real Ollama; the provider retries without tools."""
    from agents.rag_pro import answer_pro
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index

    settings = _ollama_settings(make_settings, server, master_model="gemma3:4b", visual_retrieval="off",
                                multimodal_answer="off")
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    result = answer_pro("Calculate the power dissipation at VIN = 12 V and IOUT = 1.5 A", [index], settings,
                        ModelRouter(settings))
    assert result.used_llm and result.model == "gemma3:4b"
    assert "[REF-" in result.answer


def test_groq_vision_request_carries_jpeg(sample_pdf, make_settings, server, fake_colpali):
    from agents.rag_pro import answer_pro
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index

    settings = make_settings(vision_model="qwen/qwen3.8-27b", visual_retrieval="on", multimodal_answer="on")
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    result = answer_pro("What is the pin configuration?", [index], settings, ModelRouter(settings))
    readings = [s for s in result.sources if s.agent == "vision_reader"]
    assert readings, "the Groq vision model should have read at least one page image"
    assert ("groq", "qwen/qwen3.8-27b") in [tuple(r) for r in server_log(server)["requests"]]


def test_covered_pages_only_boost_and_unrelated_reading_is_dropped(sample_pdf, make_settings, server, fake_colpali):
    from agents.orchestrator import plan
    from agents.retrieval_stage import run_retrieval
    from llm.model_router import ModelRouter
    from rag.pipeline import get_index
    from visual import reader

    settings = _ollama_settings(make_settings, server)
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    out = run_retrieval(plan("How is the power dissipation calculated?"), [index], settings)
    # page 3 is covered by the equation/text agents: it is boosted, not duplicated as a PAGE item
    assert not any(e.type == "page" and e.page == 3 for e in out.fusion.evidence)
    got = reader.read("an unrelated question", [(index, 3)], ModelRouter(settings), settings)
    assert got.pages_read == 1 and got.evidence == []  # the model said NONE
