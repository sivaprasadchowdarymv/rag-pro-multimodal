# RAG∞ Pro: production image (CPU). LLMs run on Groq / Ollama Cloud / an Ollama container.
#   docker build -t rag-pro .                      # with ColPali visual search (default)
#   docker build -t rag-pro --build-arg VISUAL=0 . # text-only, much smaller image
FROM python:3.12-slim

ARG VISUAL=1
# CPU wheels by default; for an NVIDIA GPU use https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Bake the ColPali weights into the image (empty = download on first use)
ARG PRELOAD_COLPALI=vidore/colSmol-256M

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data HF_HOME=/app/hf FASTEMBED_CACHE_PATH=/app/data/models

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt requirements-visual.txt ./
RUN pip install -r requirements.txt \
    && if [ "$VISUAL" = "1" ]; then \
         pip install "torch>=2.3,<2.14" torchvision --index-url "$TORCH_INDEX" \
         && pip install -r requirements-visual.txt; \
       fi
COPY --chown=app:app . .
RUN mkdir -p /app/data /app/hf && chown -R app:app /app/data /app/hf
USER app

# Optional: download ColPali now so the first upload does not wait for it
RUN if [ "$VISUAL" = "1" ] && [ -n "$PRELOAD_COLPALI" ]; then \
      COLPALI_MODEL="$PRELOAD_COLPALI" python -c "from config.settings import load_settings; \
from visual.colpali import get_embedder; get_embedder(load_settings())._load(); print('ColPali cached')"; \
    fi

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
