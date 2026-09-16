Swiss Courts Assistant

Conversational search over Swiss case law ([voilaj/swiss-caselaw](https://huggingface.co/datasets/voilaj/swiss-caselaw)).

## Web app

React + Vite frontend in `web/`, styled after `DESIGN.md`: conversation history on the left,
chat in the middle, and the cited decision on the right with the passage highlighted.
Every statement in an answer carries a numbered citation `[n]`; clicking it opens the source
decision with the cited passage highlighted.

The backend is a FastAPI app in `src/swiss_court_assistant/server/` (port 8090). It keeps
conversation history in SQLite (`data/app/conversations.sqlite`), serves decisions from the
subset parquet for the preview, and streams chat answers as server-sent events. Development
runs two processes; Vite proxies `/api` to the backend:

```bash
uv run python -m swiss_court_assistant.server --reload   # API on :8090
cd web && npm install && npm run dev                      # UI on :5173
```

Or build the UI once and let FastAPI serve it too: `cd web && npm run build`, then open
http://localhost:8090.

**On NVIDIA LaunchPad**, open the app through the environment's code-server port proxy:
`https://<environment>.apps.launchpad.nvidia.com/coder/proxy/8090/` (with the trailing slash).
The UI uses relative asset and API paths (`base: "./"` in `vite.config.ts`, `api/…` in
`web/src/api/http.ts`), so it works under such a prefix. The bare environment URL answered 403
at LaunchPad's gateway even when logged in; the Traefik route below (port 30081) was an attempt
at that and is not needed for the code-server path:

```bash
docker run -d --name sca-launchpad-route --restart unless-stopped --network host --entrypoint sleep \
  --label traefik.enable=true --label 'traefik.http.routers.sca.rule=PathPrefix(`/`)' \
  --label traefik.http.routers.sca.priority=1 --label traefik.http.routers.sca.entrypoints=web \
  --label traefik.http.services.sca.loadbalancer.server.port=8090 traefik:v3.7.1 infinity
```

Remove it with `docker rm -f sca-launchpad-route`.

| Route | |
|---|---|
| `GET /api/health` | agent name, number of decisions loaded |
| `GET /api/conversations`, `GET/DELETE /api/conversations/{id}` | history |
| `GET /api/decisions/{decision_id}` | metadata + full text for the preview |
| `POST /api/translate` `{text, source, target}` | machine translation of a cited passage by the Riva Translate NIM; statute abbreviations and BGE/ATF/DTF are passed as do-not-translate phrases mapped to the target language's official form (OR → CO); cached |
| `POST /api/speech` `{text, language}` | read aloud by the Magpie TTS NIM: 16-bit mono PCM streamed sentence by sentence (rate in `X-Sample-Rate`); `[n]` markers and markdown are dropped |
| `POST /api/chat` `{conversationId?, message}` | `text/event-stream` of `conversation`, `meta` (the question's language), `status`, `tool_start`, `tool_end`, `delta`, `citation`, then `done` or `error` |

In the preview, a highlighted citation has an **Explain** button (the agent's reason for citing
it) and, when the decision is in another language than the question, a **Translate** button
(into the question's language, detected from its function words).

Selecting any text in the answer or in the decision opens a small toolbar (`SelectionTools.tsx`):
**Read aloud**, **Translate** (only when the selection is not in the conversation's language) and
**Reply**, which shows the selection as a card above the message box ("Replying to <court docket>",
not editable; × or Esc removes it) for a follow-up question. The message is sent with the quote
as `> …` lines plus a `> — court docket` line, and the thread shows it as the same card. Quoted lines are ignored when detecting the question's language
and for the conversation title; the agent's prompt says they are the text the question is about.

### The agent

`server/react_agent.py` is a LangChain `create_agent` ReAct agent on the Nemotron LLM NIM
(OpenAI-compatible API) with three research tools:

| Tool | What it does |
|---|---|
| `semantic_search(query_de, query_fr, query_it)` | the agent writes the search in each corpus language (with that language's statute abbreviations); each query runs a KNN over the sqlite-vec embeddings (Nemotron embedder once fully built, else bge-m3) filtered to decisions in its language (the three scans run in parallel), and its top 40 are reranked against it by the Nemotron reranker NIM; returns 8 passages, at least 2 per language and at most 2 per decision |
| `keyword_search(keyword)` | SQLite FTS5 over all passages; `"quoted text"` is an exact phrase, other words must all appear; docket numbers in the query are matched to decisions |
| `read_decision(decision_id, offset=0)` | metadata, Regeste and 8,000 characters of the full text per call |

It works in two phases:

1. **Research.** Every model call must be a tool call (`tool_choice="required"`), so the model
   cannot answer from memory or in free text. It calls a fourth tool, `write_answer`, once it has
   enough (it is forced to after 8 tool calls).
2. **Answer.** A middleware (`ResearchThenAnswer`) intercepts `write_answer` and makes one call
   whose output is constrained to the JSON schema of `AgentAnswer` (vLLM structured outputs):
   `{"answer": [{"type": "text", "text": …} | {"type": "citation", "decision_id": …, "chunk_id": …, "quote": …, "explanation": …}, …]}`.
   Free-form JSON from the model was often invalid; constrained decoding always yields valid JSON.
   The model occasionally pads that JSON with whitespace without end, so the stream stops an
   answer after 64 blank characters and keeps the parts already written. (A whitespace-free EBNF
   grammar rules this out, but decodes about six times slower on this NIM.)

The answer streams: text parts as they are written, each citation once it is complete. The
server locates the `quote` in the decision's full text (tolerant to whitespace, quote styles and
`…`), so the UI highlights exactly that span, with the `explanation` shown in the preview. A quote
pinned on the wrong decision is moved to the decision it comes from, and a quote found nowhere
verbatim is flagged. Tool calls stream to the UI as they start and finish.

Build the keyword index once (about a minute, 0.5 GB, a separate file so it can be built while
the vector DB is being written):

```bash
uv run python -m swiss_court_assistant.fts build
```

The LLM NIM needs tool calling enabled. Port 8000 is taken by the embedder NIM, so map it
elsewhere (the agent defaults to 9100), and keep it on GPU 0: GPU 1 holds the reranker and
the query embedder. On this machine the weights are cached in `~/.cache/nim/dragos` (~2.5 min to
start; pointing at a folder without them downloads 21 GB):

```bash
set -a; . ./.env; set +a; export NGC_API_KEY="$NVDA_KEY"
docker run -d --name nim-llm --gpus '"device=0"' --shm-size=16GB -e NGC_API_KEY \
  -e NIM_PASSTHROUGH_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3" \
  -v ~/.cache/nim/dragos:/opt/nim/.cache -u $(id -u) -p 9100:8000 nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:latest
```

The **Translate** button uses the Riva Translate NIM over gRPC. GPU 0 is full with the LLM, so it
runs on GPU 1 (~1–4 s per passage):

```bash
docker run -d --name riva-translate --gpus '"device=1"' --shm-size=8GB -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 -e NIM_GRPC_API_PORT=50051 \
  -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 9000:9000 -p 50051:50051 \
  nvcr.io/nim/nvidia/riva-translate-1_6b:latest
curl localhost:9000/v1/health/ready
```

**Read aloud** (answers, cited passages, explanations, translations) uses the Magpie TTS NIM. Its
default ports are taken by the translator, so map them to 9001/50052. The first start builds its
TensorRT engines (~30 min on GPU 1). They live in the container, not in the cache mount, so use
`docker stop`/`docker start magpie-tts` afterwards; `docker rm` (or `--rm`) means rebuilding:

```bash
docker run -d --name magpie-tts --gpus '"device=1"' --shm-size=8GB -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 -e NIM_GRPC_API_PORT=50051 \
  -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 9001:9000 -p 50052:50051 \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
curl localhost:9001/v1/health/ready
```

Settings (environment): `SCA_TRANSLATE_URI` (default `localhost:50051`), `SCA_TRANSLATE_MODEL`
(default: the NIM's only model), `SCA_TTS_URI` (default `localhost:50052`), `SCA_TTS_VOICE_<LANG>`
(e.g. `SCA_TTS_VOICE_DE`; default: the first voice the NIM lists for the language), `SCA_LLM_URL` (default `http://localhost:9100/v1`), `SCA_LLM_MODEL`
(default: the first model the server lists), `SCA_LLM_THINKING=0` (stops the reasoning before each research step, which the UI streams as a grey "Thinking" line; ~1 s per step), `SCA_EMBED_MODEL`,
`SCA_EMBED_DEVICE` (default `cuda:1`), `SCA_RERANK=0`, `SCA_DECISIONS`, `SCA_VECTOR_DB`, `SCA_DB`.
`SCA_AGENT=stub` swaps in a canned agent (`server/agent.py`) that needs no LLM.

## Data

Download the corpus (~7.5 GB decisions + citation/statute graph):

```bash
uv run hf download voilaj/swiss-caselaw --repo-type dataset --include "data/*.parquet" --local-dir data/raw
uv run hf download voilaj/swiss-caselaw --repo-type dataset --include "graph/*.parquet" --local-dir data/raw
```

Build a stratified development subset (keeps the joint distribution over
language × jurisdiction × legal branch × period; deterministic per seed):

```bash
uv run python -m swiss_court_assistant.sampling --n 50000 --seed 42
```

Output: `data/subset/decisions_50k_seed42.parquet` plus a `.manifest.json`
comparing full-corpus vs subset shares for every stratum.

## Retrieval pipeline

1. **Chunk** decisions into ~1.4k-char citable passages (char offsets into
   `full_text`, Erwägung numbers from `structure/erwaegungen_paragraphs.parquet`,
   Regeste as its own chunk):

   ```bash
   uv run hf download voilaj/swiss-caselaw --repo-type dataset --include "structure/*.parquet" --local-dir data/raw
   uv run python -m swiss_court_assistant.chunking data/subset/decisions_50k_seed42.parquet
   ```

2. **Build indexes** (BM25 with per-language stemming; dense embeddings):

   ```bash
   C=data/subset/decisions_50k_seed42.chunks.parquet
   uv run python -m swiss_court_assistant.retrieval build bm25 $C
   uv run python -m swiss_court_assistant.retrieval build bge-m3 $C --device cuda:1
   uv run python -m swiss_court_assistant.retrieval build qwen3-4b $C --device cuda:0
   ```

3. **Generate the test set** — a local LLM writes one question per gold passage
   (balanced over language × period) and translates it to de/fr/it/rm/en.
   Needs an OpenAI-compatible server:

   ```bash
   docker run -d --name vllm-qwen --gpus '"device=0"' --ipc=host -p 8000:8000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     vllm/vllm-openai:latest --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 16384
   uv run python -m swiss_court_assistant.evalset data/subset/decisions_50k_seed42.parquet
   ```

   **NVIDIA NIM models** (Nemotron embedder + reranker) run as containers, one GPU
   each, on different host ports. `.env` holds the NGC key as `NVDA_KEY`:

   ```bash
   set -a; . ./.env; set +a; export NGC_API_KEY="$NVDA_KEY"
   mkdir -p ~/.cache/nim
   docker run -d --name nim-embed --gpus '"device=0"' --shm-size=16GB -e NGC_API_KEY \
     -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 8000:8000 nvcr.io/nim/nvidia/nemotron-3-embed-1b:latest
   docker run -d --name nim-rerank --gpus '"device=1"' --shm-size=16GB -e NGC_API_KEY -e HF_HOME=/tmp/huggingface \
     -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 8001:8000 nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:latest
   uv run python -m swiss_court_assistant.retrieval build nemotron-embed $C
   ```

   Override the endpoints with `NIM_EMBED_URL` / `NIM_RERANK_URL` if needed.

4. **Evaluate** (decision- and passage-level MRR/hit@k by query language,
   decision language, period, style, cross-lingual):

   ```bash
   uv run python -m swiss_court_assistant.evaluate $C --systems bm25 bge-m3 qwen3-4b hybrid hybrid+rerank
   ```

## Vector database (SQLite)

`data/vectordb/decisions_50k_seed42.sqlite` holds everything retrieval needs in one
file, using the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension
(`src/swiss_court_assistant/vectordb.py`):

| Table | Contents |
|---|---|
| `decisions` | every metadata column of the subset (court, canton, docket, dates, language, regeste, …) plus `full_text` |
| `chunks` | citable passages: `chunk_id`, `decision_id`, `section` (regeste / erwaegung / body), `char_start`/`char_end` into `full_text`, `erwaegungen` (JSON list of Erwägung numbers), `text` |
| `vec_<model>` | one vec0 KNN table per embedding model (cosine), with filter columns `language`, `canton`, `jurisdiction`, `court`, `branch`, `period`, `year`, `section` |
| `embedding_models` | which models are in the file, their dimension, and how many chunks are embedded |

### Rebuild from scratch

Everything under `data/` can be regenerated. The Hugging Face dataset is updated daily, so
pin the revision the subset was drawn from to get the same 48,774 decisions back:

```bash
REV=cc1023a967052f1c63124eeea6c8cad3c25dc9a0   # voilaj/swiss-caselaw, 2026-09-13
uv run hf download voilaj/swiss-caselaw --repo-type dataset --revision $REV --include "data/*.parquet" --local-dir data/raw
uv run hf download voilaj/swiss-caselaw --repo-type dataset --revision $REV --include "structure/*.parquet" --local-dir data/raw
uv run python -m swiss_court_assistant.sampling --n 50000 --seed 42
uv run python -m swiss_court_assistant.chunking data/subset/decisions_50k_seed42.parquet
```

Then build the database with one or more embedding models:

```bash
# bge-m3, computed locally on a GPU (~25 min on an H100)
uv run python -m swiss_court_assistant.vectordb build --model bge-m3 --device cuda:1

# or import existing retrieval.py embeddings in a few minutes instead of recomputing
uv run python -m swiss_court_assistant.vectordb build --model bge-m3 --from-npy data/index/decisions_50k_seed42/bge-m3.npy

# NVIDIA Nemotron embedder via the nim-embed container (start it first, see above; ~100 min)
uv run python -m swiss_court_assistant.vectordb build --model nemotron-embed
```

The build is **resumable**: embeddings are committed every 8,192 chunks, and rerunning the
same command only embeds what is missing, so an interrupted build can simply be restarted.
`--from-npy` checks that the `.npy` rows match the chunk order before importing.

### Query

```bash
uv run python -m swiss_court_assistant.vectordb info
uv run python -m swiss_court_assistant.vectordb search "Kündigung wegen Eigenbedarf" --model bge-m3 --k 5 --language fr --year-from 2010
```

From Python:

```python
from swiss_court_assistant.vectordb import VectorDB

db = VectorDB("data/vectordb/decisions_50k_seed42.sqlite", device="cuda:1")
hits = db.search("Kündigung wegen Eigenbedarf", model="bge-m3", k=10, canton="ZH", year_from=2015)
# each hit: similarity, chunk_id, decision_id, text, char_start/char_end, erwaegungen,
#           court, canton, docket_number, decision_date, language, title, source_url
```

sqlite-vec searches exhaustively (no approximate index). Measured per query over 797k chunks
on CPU: 1–2 s with bge-m3 (1024-dim), 1.3–2.8 s with nemotron-embed (2048-dim), less with
selective filters. Nemotron queries are embedded by the `nim-embed` container (~0.02 s). Filters are applied inside the search,
so the `k` results always match them.
