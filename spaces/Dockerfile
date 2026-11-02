# RAG∞ Pro on a Hugging Face Space (Docker SDK, free "CPU basic": 2 vCPU, 16 GB RAM).
# One container: Ollama + Gemma 3 (vision + fallback answers) + ColPali (colSmol-256M) + the app on port 7860.
# Assembled into the Space repo by spaces/build_space.py (the GitHub Action does this for you).
FROM python:3.12-slim

ARG OLLAMA_MODEL=gemma3:4b
ARG COLPALI=vidore/colSmol-256M

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates zstd \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://ollama.com/install.sh | sh \
 && useradd -m -u 1000 user

WORKDIR /home/user/app
COPY requirements.txt requirements-visual.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "torch>=2.3,<2.14" torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-visual.txt

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    DATA_DIR=/home/user/app/data HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/app/data/models \
    OLLAMA_MODELS=/home/user/.ollama/models OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 \
    # Free-first: Groq answers if GROQ_API_KEY is set as a Space secret; local Gemma always works (no key).
    LLM_PROVIDERS=groq,ollama_local \
    ENABLE_OLLAMA_LOCAL=true OLLAMA_LOCAL_HOST=http://127.0.0.1:11434 \
    OLLAMA_LOCAL_MASTER_MODEL=${OLLAMA_MODEL} OLLAMA_LOCAL_FAST_MODEL=${OLLAMA_MODEL} \
    OLLAMA_LOCAL_VERIFIER_MODEL=${OLLAMA_MODEL} OLLAMA_LOCAL_VISION_MODEL=${OLLAMA_MODEL} \
    VISUAL_RETRIEVAL=on COLPALI_MODEL=${COLPALI} COLPALI_BATCH=2 \
    MULTIMODAL_ANSWER=auto VISION_READ_PAGES=1 LLM_TIMEOUT=300

# Bake Gemma into the image: no model download when the Space starts
RUN (ollama serve > /tmp/ollama-build.log 2>&1 &) \
 && for i in $(seq 1 60); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done \
 && ollama pull "${OLLAMA_MODEL}"

COPY --chown=user:user . .

# Bake ColPali weights too
RUN python -c "from config.settings import load_settings; from visual.colpali import get_embedder; \
get_embedder(load_settings())._load(); print('ColPali cached')"

EXPOSE 7860
CMD ["sh", "spaces/start.sh"]
