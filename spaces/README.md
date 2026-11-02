---
title: RAG Pro Multimodal
emoji: ♾️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Verified, cited answers from datasheets — ColPali + Gemma
---

# RAG∞ Pro — multimodal

Upload datasheets or technical PDFs and ask questions. Specialist agents gather
text, table, equation and figure evidence; **ColPali** ranks pages by their
images; **Gemma 3** (running inside this Space) reads the most relevant page
images; a master model writes a cited answer and a verifier checks every number.

* Runs on free CPU hardware: answers from local Gemma take about 1–3 minutes.
  Add a free `GROQ_API_KEY` secret for answers in seconds (Gemma still reads the page images).
* Add `HF_TOKEN` and `HF_DATASET_REPO` secrets to keep chats and documents across restarts.

Source code: see the linked GitHub repository.
