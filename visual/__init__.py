"""
Multimodal layer for RAG∞ Pro.

    visual/pages.py     PDF page -> JPEG image (data/documents/<doc_id>/pages/)
    visual/colpali.py   ColPali late-interaction page index (multi-vector, MaxSim)
    visual/stage.py     ColPali as one more retriever inside the evidence fusion
    visual/reader.py    vision model reads the most relevant page images -> cited evidence

Everything here is optional and fails soft: without torch/colpali-engine, or
without a vision model, the text pipeline works exactly as before.
"""
