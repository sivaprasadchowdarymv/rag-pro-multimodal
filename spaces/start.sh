#!/bin/sh
# Hugging Face Space entry point: start Ollama, wait for it, warm Gemma, run the app on port 7860.
set -e

ollama serve > /tmp/ollama.log 2>&1 &

echo "waiting for ollama..."
for i in $(seq 1 90); do
  if curl -fs http://127.0.0.1:11434/api/tags > /dev/null; then echo "ollama up"; break; fi
  sleep 1
done

# Load Gemma into RAM now, so the first question doesn't pay for it
model="${OLLAMA_LOCAL_VISION_MODEL:-gemma3:4b}"
curl -s http://127.0.0.1:11434/api/generate \
  -d "{\"model\":\"$model\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 &

# The Space is shown inside an iframe on huggingface.co: uploads need XSRF/CORS off there.
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false
