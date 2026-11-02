# RAG∞ Pro Multimodal — execution guide, from scratch

A fresh, standalone project: new folder, new GitHub repo, new Hugging Face Space, new storage dataset. Nothing depends on any earlier repo. Every step is free. The only optional paid step is the last one.

| Name used in this guide | Value (change if you like) |
|---|---|
| Project folder | `rag-pro-multimodal` |
| GitHub repo | `<github-user>/rag-pro-multimodal` |
| Hugging Face Space | `<hf-user>/rag-pro-multimodal` |
| Storage dataset (private) | `<hf-user>/rag-pro-multimodal-data` |

```text
Your PC ──git push──► GitHub (CI: lint + 93 tests + Docker build) ──auto-deploy──► Hugging Face Space (free, public URL)
                                                                └──────────────► Streamlit Cloud (optional, text-only)

Per question inside the app:
  orchestrator → text · table · equation · figure agents ─┐
                 ColPali page-image search ───────────────┴→ RRF fusion + reranker
  → vision reader (Gemma 3 reads top page images) → master writes → verifier checks → cited answer
```

---

## Step 0 — Accounts and tools (one time)

| Account | Where | What for |
|---|---|---|
| GitHub | github.com/signup | the new repo, CI, auto-deploy |
| Hugging Face | huggingface.co/join | the Space (hosting) and the private dataset (storage) |
| Groq (recommended) | console.groq.com/keys → Create API Key | fast free answers (`gsk_...`) |

| Tool | Windows | Ubuntu | Check |
|---|---|---|---|
| Python 3.12 | python.org installer, tick **Add to PATH** | `sudo apt install python3.12 python3.12-venv` | `python --version` |
| Git + Git LFS | git-scm.com, git-lfs.com | `sudo apt install git git-lfs` | `git --version` |
| GitHub CLI (optional) | cli.github.com | `sudo apt install gh` | `gh --version` |
| Ollama | ollama.com/download | `curl -fsSL https://ollama.com/install.sh \| sh` | `ollama --version` |
| Docker (Step 6 only) | Docker Desktop (WSL 2) | `curl -fsSL https://get.docker.com \| sh` | `docker run --rm hello-world` |

One-time Git setup:
```bash
git config --global user.name  "your-name"
git config --global user.email "you@example.com"
git lfs install
```
Hardware: 8 GB RAM minimum (16 GB comfortable). About 10 GB free disk. A GPU is optional.

---

## Step 1 — Create the project folder

1. Unzip `rag-pro-multimodal.zip` into a NEW location, e.g. `D:\projects\` or `~/projects/`. Do not unzip it inside an old project.
2. Open a terminal in it:
   ```bash
   cd rag-pro-multimodal
   ```
3. Check the contents:
   * `app.py` (UI entry)
   * `agents/`, `rag/`, `retrieval/`, `visual/` (ColPali + vision reader)
   * `llm/`, `storage/`, `ui/`
   * `spaces/` (Hugging Face Space)
   * `.github/workflows/`, `tests/`
   * `requirements*.txt`, `.env.example`

---

## Step 2 — Install everything locally

1. **Create and activate the virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Ubuntu
   ```
   The prompt shows `(venv)`. Activate it again in every new terminal.
2. **Install the app:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install torch + torchvision.** Keep `< 2.14`, because colpali-engine 0.3 requires it:
   ```bash
   pip install "torch>=2.3,<2.14" torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU
   pip install "torch>=2.3,<2.14" torchvision --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU
   ```
4. **Install ColPali:**
   ```bash
   pip install -r requirements-visual.txt
   python -c "import torch, colpali_engine; print(torch.__version__, torch.cuda.is_available())"
   ```
   `True` = GPU (colpali-v1.3 is used); `False` = CPU (colSmol-256M is used).
5. **Pull Gemma 3** (free local model that reads page images):
   ```bash
   ollama pull gemma3:4b
   ollama list
   ollama run gemma3:4b "describe this image" ./any_picture.png
   ```
   On Ubuntu, if `ollama list` cannot connect: `ollama serve`.
6. **(Optional) Run the tests:**
   ```bash
   pip install -r requirements-dev.txt
   python -m pytest tests -q        # expect: 93 passed (offline, no GPU, no downloads)
   ```

---

## Step 3 — Configure `.env` (fresh keys)

```bash
copy .env.example .env       # Windows
cp .env.example .env         # Ubuntu
```
Choose ONE setup and edit `.env`:

**A. Recommended (free): Groq writes answers, local Gemma reads page images**
```ini
GROQ_API_KEY=gsk_your_new_key
LLM_PROVIDERS=groq,ollama_local
ENABLE_OLLAMA_LOCAL=true
OLLAMA_LOCAL_HOST=http://localhost:11434
VISUAL_RETRIEVAL=auto
MULTIMODAL_ANSWER=auto
```

**B. Fully offline, no keys**
```ini
LLM_PROVIDERS=ollama_local
ENABLE_OLLAMA_LOCAL=true
OLLAMA_LOCAL_MASTER_MODEL=gemma3:4b
OLLAMA_LOCAL_FAST_MODEL=gemma3:4b
OLLAMA_LOCAL_VERIFIER_MODEL=gemma3:4b
```
Slower on CPU, but nothing leaves your PC. Gemma has no tool calling on Ollama, so the app answers without tools automatically.

**C. Cloud only (no Ollama)**
```ini
GROQ_API_KEY=gsk_your_new_key
OLLAMA_API_KEY=your_ollama_cloud_key
LLM_PROVIDERS=groq,ollama_cloud
```
Page reading then needs a vision model your cloud account lists, e.g. `GROQ_VISION_MODEL=<model id>`.

**Storage (recommended; required on the Space):**
1. huggingface.co → profile → **Settings → Access Tokens → Create new token**.
2. Set type **Write**, name `rag-pro-multimodal`, and copy the token (`hf_...`).
3. Add to `.env`:
   ```ini
   HF_TOKEN=hf_your_new_write_token
   HF_DATASET_REPO=your-hf-username/rag-pro-multimodal-data
   ```
   The app creates this dataset as **private** on first start. You don't create it yourself.

Optional: `APP_PASSWORD=...` locks the app. `HF_HOME=D:\hf_cache` keeps model downloads off drive C:.

---

## Step 4 — Run, verify, ask

1. **Start:**
   ```bash
   streamlit run app.py        # http://localhost:8501
   ```
   Use a terminal, not Spyder's Run button.
2. **Sidebar → Status should show:**

   | Line | Expected |
   |---|---|
   | Groq | 🟢 connected (A, C) |
   | Ollama (local) | 🟢 connected (A, B) |
   | Visual search | 🟢 ColPali (colSmol-256M, cpu), or colpali-v1.3 on cuda |
   | Page reading | 🟢 gemma3:4b |
   | Storage | 🟢 saved to Hugging Face (if Step 3 storage is set) |
3. **Save your workspace link:** Settings → Workspace & storage → copy `?ws=...` and bookmark it. It is your private workspace.
4. **Index a datasheet:** Chat page → drop a PDF.
   * The bar runs through four phases: reading pages → embedding chunks → page images → visual index (ColPali).
   * The first upload also downloads ColPali (about 0.5 GB on CPU, about 6 GB on GPU) plus the small text models (~210 MB).
   * After that, indexing takes about 1–3 s per page on CPU.
5. **Library:** the card shows **👁 N pages in visual search**.
6. **Ask one of each:**
   * Graph: "From the dropout voltage vs output current graph, what is the dropout at 1 A?"
   * Pinout: "What is on pin 3?"
   * Table: "What is the maximum input voltage?"
   * Equation: "How is the power dissipation calculated?"
7. **Check the multimodal parts:**
   * **Agent activity** shows `Visual page search (ColPali): N page(s) ranked`. For figure questions or figure/scanned pages it also shows `Vision reader (gemma3:4b): read N page image(s)`.
   * **Evidence** shows `PAGE` sources with the page image next to the text and table sources.
   * The answer has a badge: **✓ Verified**, **≈ Partly verified** or **⚠ Check sources**.
8. **Tune: Settings → Advanced → Multimodal:**
   * Visual search: auto / on / off.
   * Vision reading: auto / on / off.
   * Pages read per question: each page is one Gemma call, tens of seconds on CPU.
9. **Validate (optional):** Validate page → pick the document → Generate test questions → **Check retrieval** (free) → **Full test**.

---

## Step 5 — Create the new GitHub repo and push

1. **Initialise Git** in the project folder:
   ```bash
   git init -b main
   git add .
   git status        # must NOT list .env, data/, venv/ or any .pdf (.gitignore blocks them)
   git commit -m "RAG∞ Pro Multimodal: first version"
   ```
2. **Create the repo and push.**
   * With GitHub CLI:
     ```bash
     gh auth login
     gh repo create rag-pro-multimodal --private --source . --push
     ```
   * Without the CLI:
     1. On github.com, click **+ → New repository**, name it `rag-pro-multimodal`, choose Private, and don't add a README.
     2. Then run:
        ```bash
        git remote add origin https://github.com/<github-user>/rag-pro-multimodal.git
        git push -u origin main
        ```
3. **Repo → Actions:**

   | Workflow | Runs | Does |
   |---|---|---|
   | CI | every push / pull request | lint, 93 tests (no torch needed), Docker build |
   | Deploy to Hugging Face Space | after CI passes on `main` | deploys Step 7 (idle until configured) |
4. **Wait for a green CI tick** (5–10 min the first time). If it's red, click the job and read the first error line.

Daily updates afterwards:
```bash
git add -A
git commit -m "describe the change"
git push
```

---

## Step 6 — (Optional) Docker on your PC

1. In `.env`:
   ```ini
   ENABLE_OLLAMA_LOCAL=true
   OLLAMA_LOCAL_HOST=http://ollama:11434
   LLM_PROVIDERS=groq,ollama_local
   ```
   For no keys at all, also add the three `OLLAMA_LOCAL_*_MODEL=gemma3:4b` lines from setup B.
2. Start the stack:
   ```bash
   docker compose --profile vision up -d --build
   docker compose ps                     # app "healthy", ollama-pull "exited (0)"
   docker compose logs -f ollama-pull    # gemma3:4b download, ~3.3 GB, once
   ```
3. Open http://localhost:8501 and repeat the Step 4 checks.

| Command | Result |
|---|---|
| `docker compose up -d --build` | app only (Groq / Ollama Cloud), ColPali included |
| `docker build -t rag-pro-multimodal --build-arg VISUAL=0 .` | small text-only image |
| `docker compose down` / `docker compose down -v` | stop / stop and delete data |

---

## Step 7 — Go live on a new Hugging Face Space (Docker, free)

The Space runs one container: Ollama + Gemma 3 4B (baked in) + ColPali colSmol-256M (baked in) + the app, on free **CPU basic** hardware (2 vCPU, 16 GB RAM).

1. **Create the Space** at huggingface.co/new-space:
   * Space name: `rag-pro-multimodal`
   * SDK: **Docker** → **Blank**
   * Hardware: **CPU basic · Free**
   * Visibility: Private or Public
   * Click **Create Space**.
2. **Space secrets** (Space → Settings → Variables and secrets):

   | Name | Type | Value | Needed? |
   |---|---|---|---|
   | `GROQ_API_KEY` | Secret | `gsk_...` | recommended (answers in seconds) |
   | `HF_TOKEN` | Secret | `hf_...` (write) | keeps data across restarts |
   | `HF_DATASET_REPO` | Variable | `<hf-user>/rag-pro-multimodal-data` | keeps data across restarts |
   | `APP_PASSWORD` | Secret | any | optional lock |

   Without `GROQ_API_KEY`, local Gemma answers: no key and no cost, just slower.
3. **Link GitHub to the Space** (GitHub repo → Settings → Secrets and variables → Actions):
   * **Secrets** tab → **New repository secret** → `HF_TOKEN` = a Hugging Face **write** token.
   * **Variables** tab → **New repository variable** → `HF_SPACE` = `<hf-user>/rag-pro-multimodal`.
4. **Deploy:** GitHub → Actions → **Deploy to Hugging Face Space** → **Run workflow**. From now on every push to `main` that passes CI redeploys automatically.

   Manual alternative:
   ```bash
   python spaces/build_space.py              # creates space_build/
   cd space_build
   git init -b main
   git lfs install
   git add -A
   git commit -m "deploy"
   git push --force https://<hf-user>:<HF_TOKEN>@huggingface.co/spaces/<hf-user>/rag-pro-multimodal main
   ```
5. **First build:** Space → **Logs → Build**, about 20–40 minutes. It installs CPU torch and bakes Gemma (~3.3 GB) and colSmol into the image, so restarts download nothing.
6. **Verify:** when the badge says **Running**, open **App**.
   * Sidebar should show:
     * Ollama (local) connected
     * Visual search ColPali (colSmol-256M, cpu)
     * Page reading gemma3:4b
     * Storage saved to Hugging Face
   * Copy your workspace link, upload a datasheet, and ask a figure question.
7. **Persistence test:** Space → Settings → **Factory reboot**, then open your workspace link. Chats and documents return without re-uploading.

On free CPU the Space reads 1 page image per question (`VISION_READ_PAGES=1`); answers take about 1–3 min with local Gemma, or seconds with Groq. A free Space sleeps when unused; the next visit wakes it.

---

## Step 8 — (Optional) Streamlit Community Cloud (text-only)

Streamlit Cloud has ~2.7 GB RAM, too little for ColPali. It runs the text pipeline only; the sidebar says "Visual search: not installed".
1. Go to share.streamlit.io → **Create app** → deploy from GitHub.
2. Choose repo `<github-user>/rag-pro-multimodal`, branch `main`, file `app.py`.
3. **Advanced settings** → Python **3.12** → **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   HF_TOKEN = "hf_..."
   HF_DATASET_REPO = "your-hf-username/rag-pro-multimodal-data"
   # optional
   OLLAMA_API_KEY = "..."
   APP_PASSWORD = "choose-a-password"
   ```
4. Click **Deploy** and wait 3–6 min. Every `git push` redeploys.

It shares the same storage dataset, so a workspace link works on both the Space and Streamlit Cloud.

---

## Step 9 — Last, optional, paid: faster Space

Only if the free Space is too slow:
1. huggingface.co/settings/billing → add a payment method → set a monthly spending limit.
2. Space → Settings → **Space hardware** → a CPU upgrade or a small GPU.
3. For a GPU only:
   * in `spaces/Dockerfile`, change `/whl/cpu` → `/whl/cu126`;
   * add the Space variable `COLPALI_MODEL=vidore/colpali-v1.3`;
   * push.
4. Set **Sleep time** so paid hardware stops when idle. Switch back to **CPU basic · Free** any time.

---

## Settings reference

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDERS` | `groq,ollama_cloud,ollama_local` | fallback order; only configured providers are called |
| `VISUAL_RETRIEVAL` | `auto` | ColPali on when torch + colpali-engine are installed |
| `COLPALI_MODEL` | `auto` | colpali-v1.3 on GPU, colSmol-256M on CPU |
| `COLPALI_BATCH` | `2` | pages per ColPali pass (lower = less RAM) |
| `VISUAL_TOP_K` | `4` | pages ColPali adds to the fusion per question |
| `VISUAL_MAX_PAGES` | `300` | pages per document embedded by ColPali |
| `PAGE_DPI` / `PAGE_MAX_SIDE` | `110` / `1400` | page image resolution / longest side (px) |
| `MULTIMODAL_ANSWER` | `auto` | when Gemma reads pages: auto (figure questions, figure or scanned pages, no text match) / on / off |
| `VISION_READ_PAGES` | `2` | page images read per question |
| `OLLAMA_LOCAL_VISION_MODEL` | `gemma3:4b` | vision model (or `VISION_MODEL`, `GROQ_VISION_MODEL`) |
| `AGENT_MODE` | `agent` | `agent` (tools when needed) or `quick` (one call) |
| `RATE_LIMIT_PER_MIN` | `8` | questions per browser session per minute |

The full list is in `.env.example`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Visual search "not installed" | Step 2.3–2.4 |
| Visual search "could not load …" | ColPali download failed (network/disk); check the log, restart to retry |
| No "Page reading" line | `ENABLE_OLLAMA_LOCAL=true` + `ollama pull gemma3:4b`, or set a vision model |
| Vision reader never runs | Normal for plain text questions in `auto`; `MULTIMODAL_ANSWER=on` forces it |
| Slow answers on CPU | Add a Groq key; `VISION_READ_PAGES=1` |
| Ollama "model not found" | `ollama pull gemma3:4b` (Docker: `docker compose logs ollama-pull`) |
| pip conflict on torch | Reinstall `"torch>=2.3,<2.14"` from the PyTorch index |
| CUDA out of memory | `COLPALI_BATCH=1` or `COLPALI_MODEL=vidore/colSmol-256M` |
| Storage "temporary" | Check `HF_TOKEN` (write) and `HF_DATASET_REPO` |
| Space upload fails with 403 | The Space must start with `spaces/start.sh` (turns off XSRF for the HF iframe) |
| CI red | Actions → failed job → first error line |

## Final checklist

- [ ] Accounts: GitHub, Hugging Face, Groq
- [ ] Tools: Python 3.12, Git + LFS, Ollama
- [ ] New folder `rag-pro-multimodal` unzipped
- [ ] venv; `requirements.txt`; torch < 2.14; `requirements-visual.txt`
- [ ] `ollama pull gemma3:4b`
- [ ] `.env` with new keys and the new dataset name
- [ ] App running; Status all green; workspace link bookmarked
- [ ] Datasheet indexed; figure question answered with a PAGE source
- [ ] New GitHub repo pushed; CI green
- [ ] New Space created; secrets set; GitHub `HF_TOKEN` + `HF_SPACE` set; deploy ran
- [ ] Space running; persistence tested
- [ ] (Only if needed) spending limit, then hardware upgrade
