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
| `GET /api/decisions/{id}/citations` | from the corpus citation graph: how often later decisions cite this one, how many it cites, and the most recent decisions at either end (`inCorpus` marks the ones this app serves) |
| `POST /api/translate` `{text, source, target}` | machine translation of a cited passage by the Riva Translate NIM; statute abbreviations and BGE/ATF/DTF are passed as do-not-translate phrases mapped to the target language's official form (OR → CO); cached |
| `POST /api/speech` `{text, language}` | read aloud by the Magpie TTS NIM: 16-bit mono PCM streamed sentence by sentence (rate in `X-Sample-Rate`); `[n]` markers and markdown are dropped |
| `WS /api/voice` | voice mode: 16 kHz PCM from the microphone in; `partial` transcripts, the same turn events as `/api/chat`, and the spoken answer (PCM frames between `speech_start`/`speech_end`) out. Any speech sends `cancel_speech` (the browser drops audio it has not played yet — the server is always ahead) and cancels a running turn, which is saved with a "cut off" marker. The answer starts only after `SCA_VOICE_PAUSE` seconds of silence (default 1.2, ≈2 s after the speaker stops), joining everything heard since the last pause, so an end-of-utterance mid-sentence does not trigger an answer |
| `POST /api/chat` `{conversationId?, message}` | `text/event-stream` of `conversation`, `meta` (the question's language), `status`, `tool_start`, `tool_end`, `delta`, `citation`, then `done` or `error` |
| `GET /api/matters`, `GET/DELETE /api/matters/{id}` | matters (see below) |
| `POST /api/matters` (multipart `file`/`text`/`title`) | opens a matter on what the client handed over: a PDF, a Word file, plain text, or a recording already decoded to 16 kHz PCM (`*.pcm`), which is transcribed by the ASR NIM |
| `POST /api/matters/text` `{text, title?}` | the same, for facts typed in |
| `POST /api/matters/{id}/run` | `text/event-stream` of the four stages: `stage`, `intake`, then per issue `issue_start`/`issue_tool`/`issue_delta`/`issue_citation`/`issue_verdict`/`issue_done`, then `assessment_delta`, `done` or `error` |
| `GET /api/matters/{id}/memo` | the drafted memo as a Word file (`?format=md` for the Markdown behind it) |

**Grounding check.** Every citation is checked against the sentence it supports: a short constrained
call (`{"supported": true|false}`) asks whether that passage states that sentence, judged on the
passage alone. The checks run while the answer streams and arrive as `verdict` events; a citation
whose passage does not state the claim is marked in red in the answer and in the saved message
(`Source.supported`). Quote wording is checked separately (`Source.verified`, verbatim match).

In the preview, a highlighted citation has an **Explain** button (the agent's reason for citing
it) and, when the decision is in another language than the question, a **Translate** button
(into the question's language, detected from its function words).

Selecting any text in the answer or in the decision opens a small toolbar (`SelectionTools.tsx`):
**Read aloud**, **Translate** (only when the selection is not in the conversation's language) and
**Reply**, which shows the selection as a card above the message box ("Replying to <court docket>",
not editable; × or Esc removes it) for a follow-up question. The message is sent with the quote
as `> …` lines plus a `> — court docket` line, and the thread shows it as the same card. Quoted lines are ignored when detecting the question's language
and for the conversation title; the agent's prompt says they are the text the question is about.

### Matters: intake → research → assessment → drafting

A second page (`Matters` in the top bar) works a whole client case rather than one question. A case
in a firm moves through five stages; four of them are things software can touch, and the fifth is
shown greyed out because it is practice management, not case law:

1. **Intake** — one constrained call over the client's document or recording returns the matter
   title, the facts as given, the parties, a dated timeline and up to four legal questions
   (`Pipeline._intake`). A question that names a statute article the client's own text never
   mentions has it stripped (`without_invented_articles`): the model guesses article numbers from
   memory, and a wrong one sends the research after the wrong provision.
2. **Research** — each issue runs through the ordinary agent (`Agent.answer`), so it gets the same
   per-language searches, citations, verbatim quotes and grounding checks as a chat question. The
   page shows the searches as they happen and streams each answer.
3. **Assessment** — one call over the researched issues only: where the client stands, what is in
   their favour, what the other side will argue, what is still open. The citation markers are
   renumbered onto the memo's running series first, so `[3]` means the same decision everywhere.
4. **Drafting** — the memo is *assembled*, not rewritten by a model (`matters.memo` for Markdown,
   `matters.docx_memo` for Word): facts, parties, timeline, every issue with its researched answer,
   the assessment, and a table of authorities where a citation that failed its grounding check is
   flagged. Headings follow the matter's language. The page downloads it as `.docx` for the client
   file; the Markdown stays behind `?format=md` and the "Copy as Markdown" button.

Opening a matter whose stage is still `new` starts the run by itself. That has to happen in the
load effect rather than right after the upload: selecting the freshly created matter re-runs the
effect, which aborts whatever stream is open — a run started before that arrived was cancelled a
moment after it began, and the page just sat there. While the upload is being read (ASR on a
recording takes about as long as the recording lasted) the intake screen says so, and a running
matter shows which stage and which issue it is on.

Recordings are decoded in the browser (`web/src/audio.ts`): `AudioContext.decodeAudioData` plus an
`OfflineAudioContext` at 16 kHz turn any format the browser can play — or a recording made in the
page — into the mono PCM the ASR NIM wants, so the server needs no ffmpeg. Matters are stored in
`data/app/matters.sqlite` (`SCA_MATTERS_DB`), and each stage is saved as it finishes, so a browser
that disconnects loses the stream, not the work.

### The agent

`server/react_agent.py` is a LangChain `create_agent` ReAct agent on the Nemotron LLM NIM
(OpenAI-compatible API) with these research tools (the statute tools only when the index has statutes):

| Tool | What it does |
|---|---|
| `semantic_search(query_de, query_fr, query_it, …filters)` | the agent writes the search in each corpus language (with that language's statute abbreviations); each query runs a KNN over the decisions in its language (the in-memory vector matrix, else sqlite-vec; the three run in parallel), and its top 40 are reranked against it by the Nemotron reranker NIM; returns 8 passages, at least 2 per language and at most 2 per decision |
| `keyword_search(keyword, …filters)` | SQLite FTS5 over all passages; `"quoted text"` is an exact phrase, other words must all appear; docket numbers in the query are matched to decisions |
| `list_decisions(…filters, oldest=False)` | how many decisions match the filters, split by court, area and decade, and the ten newest (or oldest) with Regeste or title — for questions about the corpus or a court's latest decisions |
| `read_decision(decision_id, offset=0)` | metadata, Regeste and 8,000 characters of the full text per call |
| `citing_decisions(decision_id)` | how often later decisions cite it, and the most recent ones (citation graph) |
| `read_law(code, article, canton="CH")` | one statute article verbatim, in German, French and Italian. The code may be the abbreviation in any language (OR or CO, ZGB or CC) or the SR number: it is resolved to the act's SR number first, so "OR" also finds the French text |
| `search_laws(query_de, query_fr, query_it, canton="CH", code="")` | statute articles by meaning, over the statute rows of the vector matrix: federal law, or one canton's law; `code` keeps to one act ("OR", "StGB", an SR number); reranked per language, one entry per article (its best language version), 6 articles |

**Filters** (the full-corpus index only; `facets.py`): `canton` ("GE", "GE,VD", names like "Genf"
are understood; "CH" = the federal courts), `court` (`federal_supreme`, `leading_cases` = BGE,
`federal_administrative`, `federal_criminal`, `federal_patent`, `federal_other`, `cantonal`), `area`
(`civil`, `criminal`, `public`, `social_insurance`), `proceeding` (`appeal`, `objection`,
`debt_enforcement`, `constitutional_complaint`, `revision`, `first_instance`), `year_from`,
`year_to`. They are masks over the vector-matrix rows applied *inside* the exact search, so "canton
Uri" ranks Uri's 285 decisions rather than filtering a top 40 that holds none of them; a keyword
search ranks up to 20,000 matches and keeps those inside. What the data allows: court, canton and
date are always recorded. The area is recorded for 64 % of decisions, and for the rest is inferred
from the acts they cite (OR/ZGB/ZPO → civil, StGB/StPO → criminal, IVG/UVG/ATSG → social insurance,
VwVG/AIG/RPG → public): the inference agrees with the recorded area on 89 % of a sample and leaves
10 % of all decisions without an area. The proceeding is recorded for 56 %, so that filter drops the
rest, and a filter combination that matches nothing says how many each filter matches alone. Every
result line shows the decision's area, proceeding and outcome where known, `read_decision` its
chamber too, and cantonal courts are named from their code ("Obergericht (ZH)", "Mietgericht
(ZH)") instead of "Cantonal court ZH". The research list shows each step's filters as chips.

**Asking back.** A fifth research tool, `ask_user(question, options, found_so_far)`, lets the agent end
the turn with one question instead of an answer, when the answer turns on a fact the question leaves
open and the passages go different ways on it ("Kündigungsfrist für meinen Vertrag" — employment or
lease?). The middleware turns that call into the end of the turn; the question streams as the
turn's text, a `clarify` event carries two to four suggested answers (buttons under the question, on
the last turn only; "other" options are dropped, the user can type) and `found_so_far` plus the
decision ids found, saved as `Message.clarification`. The reply turn researches the *original*
question with the reply added (`with_clarification`, also used for its language: "Wohnmietvertrag"
alone would be detected as English), and reads the saved notes back from the history. Rules: only
after a first search, at most one question per question (not right after one), never in matters
(`ask=False`), and `POST /api/chat` takes `allowQuestions: false` for callers that need an answer
every time, such as an eval. Clear questions are still answered directly; the prompt says not to
settle the open fact by assuming one case, which the model did before it said so. The question is asked in the
user's language: the research searches in German, French and Italian and pulled an English turn's
question into German, so the prompt names the language and a question that still comes out in
another one is translated (question and options, one short JSON call). The reply turn takes its
language from the original question (`original_question`), not from the reply, whose suggested
answers may be in another language.

It works in two phases:

1. **Research.** Every model call must be a tool call (`tool_choice="required"`), so the model
   cannot answer from memory or in free text. It calls a fourth tool, `write_answer`, once it has
   enough (it is forced to after 8 tool calls). Left to itself it did not stop: for a question this
   corpus cannot answer, it spent all 8 calls rewording one search and was then forced to answer
   from whatever had come back. Three things end the loop now — the prompt says that reporting
   "this corpus does not answer it" *is* a correct answer, a reminder to that effect is added to the
   request after 4 tool calls (`NUDGE_AFTER`), and a search that repeats an earlier one is refused
   instead of run: `_already_searched()` keeps the turn's queries as token sets (in a `ContextVar`,
   because `make_tools` runs once and its tools are shared between turns) and rejects a query whose
   overlap with an earlier one reaches `SAME_SEARCH = 0.8` by Jaccard **or** by containment.
   Containment is what catches real repeats — rewording drops words, so "DSGVO Wettbewerbsrecht
   Sanktionen SVKG" followed by "DSGVO Wettbewerbsrecht" is only 0.73 Jaccard but fully contained.
   The opposite failure showed up once the agent eval existed: offered `write_answer` from the first
   step, it answered after a single search on 19 of 33 questions. `write_answer` is now left out of
   the tool list until two research calls have been made (`MIN_TOOL_CALLS`), which raised rubric
   coverage on the exam questions from 63 % to 69 % and their pass rate from 43 % to 56 %, averaged
   over three judgings (`agent-eval/RESULTS.md`), at no cost in latency.
2. **Answer.** A middleware (`ResearchThenAnswer`) intercepts `write_answer` and makes one call
   whose output is constrained to the JSON schema of `AgentAnswer` (vLLM structured outputs):
   `{"answer": [{"type": "text", "text": …} | {"type": "citation", "decision_id": …, "chunk_id": …, "quote": …, "explanation": …}, …]}`.
   Free-form JSON from the model was often invalid; constrained decoding always yields valid JSON.
   The model occasionally pads that JSON with whitespace without end, so the stream stops an
   answer after 64 blank characters and keeps the parts already written. (A whitespace-free EBNF
   grammar rules this out, but decodes about six times slower on this NIM.) The instruction closing
   that call names the question being answered: while it only said "write the final answer now", a
   follow-up question was answered with the *previous* turn's answer. For the same reason
   `_history()` strips the `[n]` markers from earlier answers — they used to be replaced by the
   cited docket number, which made an earlier answer's invented sentences look sourced.
   A turn can also end with nothing to show, and nothing logged: `{"answer": []}` is valid
   against the schema, and a research step once returned neither a tool call nor text, which ends
   the graph. An empty answer is asked for once more, a stalled step is answered from what was
   found, and a turn that still ends empty says so instead of returning an empty message — which
   the client cannot tell from an answer.

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

**Citation graph.** `data/raw/graph/citations.parquet` holds 11.8M edges for the whole corpus; the
index keeps the 1.1M that touch the subset, with a court/docket/date label for every decision they
mention (413k, mostly citing decisions from outside the subset). ~70 s, 193 MB:

```bash
uv run python -m swiss_court_assistant.citations build
uv run python -m swiss_court_assistant.citations show bge_BGE_122_V_157
```

Search results then carry "cited by N later decisions" (an authority signal the agent is told to
prefer), the agent can call `citing_decisions(decision_id)`, and the preview shows the panel.
Caveat: some decisions exist under two ids ("bge_122 V 157" and "bge_BGE_122_V_157"), so counts can
be split between the twins; the twin is filtered out of the lists.

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

**Voice mode** (the microphone button next to the composer) transcribes with the Nemotron streaming
ASR NIM, again on the next free ports (the first start builds TensorRT engines, ~35 min). The image
ships two models and **defaults to English** (`NIM_TAGS_SELECTOR=type=en-US,batch_size=128`);
`type=multi` serves the multilingual model instead (41 codes incl. de-DE, fr-FR, it-IT, plus `auto`),
which is what voice mode needs here:

```bash
docker run -d --name nemotron-asr-multi --gpus '"device=1"' --shm-size=8GB -e NGC_API_KEY \
  -e NIM_TAGS_SELECTOR="type=multi,batch_size=128" \
  -e NIM_HTTP_API_PORT=9000 -e NIM_GRPC_API_PORT=50051 \
  -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 9002:9000 -p 50053:50051 \
  nvcr.io/nim/nvidia/nemotron-asr-streaming:latest
curl localhost:9002/v1/health/ready
```

The server asks the NIM which languages it serves (`/api/health` reports them) and listens with
`auto`, so the speaker can switch language between questions.

Settings (environment): `SCA_TRANSLATE_URI` (default `localhost:50051`), `SCA_TRANSLATE_MODEL`
(default: the NIM's only model), `SCA_TTS_URI` (default `localhost:50052`), `SCA_TTS_VOICE_<LANG>`
(e.g. `SCA_TTS_VOICE_DE`; default: the first voice the NIM lists for the language), `SCA_ASR_URI`
(default `localhost:50053`), `SCA_ASR_LANGUAGE` (unset: `auto` when the NIM offers it, else the
question's language), `SCA_VOICE_PAUSE` (default 1.2 s of silence before answering), `SCA_LLM_URL` (default `http://localhost:9100/v1`), `SCA_LLM_MODEL`
(default: the first model the server lists), `SCA_LLM_THINKING=0` (stops the reasoning before each research step, which the UI streams as a grey "Thinking" line; ~1 s per step), `SCA_EMBED_MODEL`,
`SCA_EMBED_DEVICE` (default `cuda:1`), `SCA_RERANK=0`, `SCA_DECISIONS`, `SCA_VECTOR_DB`, `SCA_DB`,
`SCA_MLFLOW_URI` (unset: no tracing), `SCA_MLFLOW_EXPERIMENT` (default `swiss-court-assistant`).
`SCA_AGENT=stub` swaps in a canned agent (`server/agent.py`) that needs no LLM.

## Evaluating the agent

The retrieval evaluation further down scores the search: how often the gold decision comes back in
the top ten. That says nothing about the answer the user reads. `agent-eval/` scores that — 33 cases
put to the running assistant through its own HTTP API and graded by a local model against a
reference answer written for each one:

* **21 Swiss law exam questions** (`agent-eval/cases/exam.yaml`) over the areas the corpus covers —
  tenancy, employment, tort, contract, persons and family law, criminal law and procedure, debt
  enforcement, social insurance, constitutional law — eleven in German, six in French, three in
  Italian, one in English. Each carries the model answer a Swiss lawyer would give and a rubric of
  the points the answer has to make.
* **12 behavioural cases** (`agent-eval/cases/behaviour.yaml`): saying the corpus does not answer a
  question instead of assembling an answer out of loosely related passages, reading a decision
  instead of guessing its outcome, using the citation graph, answering in the language it was asked
  in, carrying a follow-up question's context, contradicting a false premise, ignoring an
  instruction embedded in a quoted passage.

The judge grades each rubric point on its own, plus grounding and usefulness on 1-5 and whether the
answer refused — grounding against the passages in the prompt, never against its own knowledge of
Swiss law. Legal accuracy is not asked for as a score at all: a judge on a scale marks an answer
down for what it leaves out, so it has to quote the sentence it says is wrong instead, and the
objection counts only if that sentence is really in the answer and the reference contradicts it.
Alongside the judge run checks that need no model:
the answer's language, whether the expected decision was cited and the expected tool called, how
many of the quotes were found verbatim in the decision, how many the agent's own grounding check
confirmed, and what the turn cost in calls and seconds.

```bash
uv run python agent-eval/run.py                   # every case; writes agent-eval/runs/<timestamp>/
uv run python agent-eval/run.py --suite behaviour  # one suite
uv run python agent-eval/report.py                 # re-render the last run's report
```

`agent-eval/README.md` has the case format, the scoring, and what the numbers do and do not mean;
`agent-eval/RESULTS.md` is the report of the last run.

## Model services (Docker)

Every model the app calls is a self-hosted NVIDIA NIM, and `deploy/compose.yaml` runs exactly those
six — nothing else:

| Service | Container | Host port | GPU | Used for |
|---|---|---|---|---|
| `llm` | `nim-llm` (Nemotron 3.5 Lightning) | 9100 | 0 | agent, answers, grounding checks, matters |
| `embed` | `nim-embed` (Nemotron 3 Embed 1B) | 8000 | 0 | query vectors |
| `rerank` | `nim-rerank` (Nemotron Rerank VL 1B) | 8001 | 1 | reranking search candidates |
| `translate` | `riva-translate` | 50051 / 9000 | 1 | Translate |
| `tts` | `magpie-tts` | 50052 / 9001 | 1 | Read aloud, voice mode |
| `asr` | `nemotron-asr-multi` | 50053 / 9002 | 1 | voice mode, recorded interviews |

```bash
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps
```

The NGC key comes from `~/.config/swiss-court-assistant/nim.env` (`NGC_API_KEY=…`, mode 600), not from
the repository, and only the LLM, embed and rerank services get it.

**Speech and translation start offline in under a minute.** A Riva NIM compiles its TensorRT engines
into the container whenever it finds its downloaded model package — on every start, restarts
included: 30 minutes each for ASR and TTS here. The compose file sets `NIM_DISABLE_MODEL_DOWNLOAD`
and `NIM_EXPORT_PATH`, so they unpack already-compiled models from
`~/.cache/nim/riva-export/<service>/` instead (measured: ASR 46 s, TTS 41 s, translate 21 s, no NGC
call). Those exports are tars of the running containers' `/data/models`; the header of the compose
file shows how to produce them on a fresh machine. The embed service's health check is disabled: the
image has no shell for its own check, and its health binary cannot create a CUDA context when run as
one, so it showed "unhealthy" while serving.

## Tracing the agent loop (MLflow)

Every turn is recorded as one MLflow trace: each research step with the exact prompt sent to the
model, every tool call with its arguments and result, the constrained final-answer call and the
grounding checks. `server/tracing.py` enables LangChain autologging and opens a root `turn` span
holding what the user saw — the answer, its citations and their verdicts — tagged with the
conversation id. Tracing is off unless `SCA_MLFLOW_URI` is set, and a tracing failure never costs
an answer.

```bash
docker run -d --name mlflow --restart unless-stopped -p 5000:5000 \
  -v "$PWD/data/mlflow:/mlflow" ghcr.io/mlflow/mlflow:v3.16.1 \
  mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts

SCA_MLFLOW_URI=http://localhost:5000 uv run python -m swiss_court_assistant.server --reload
```

The UI is at http://localhost:5000 (on LaunchPad, through the code-server proxy:
`https://<environment>.apps.launchpad.nvidia.com/coder/proxy/5000/`). Only the lightweight
`mlflow-tracing` client is a project dependency; the tracking server itself runs in the container.

## Full-corpus index

`swiss_court_assistant/index.py` builds one index over the whole upstream dataset — decisions,
passages, embeddings, keyword index and citation graph — with our own NIM embeddings, and updates it
incrementally. The dataset is fetched at build time; nothing is called at query time.

```bash
uv run python -m swiss_court_assistant.index build --courts bger,bge   # start small
uv run python -m swiss_court_assistant.index build                     # every shard
uv run python -m swiss_court_assistant.index update                    # apply new daily deltas
uv run python -m swiss_court_assistant.index citations                 # graph over what is indexed
uv run python -m swiss_court_assistant.index status
```

Everything is resumable and idempotent. Each upstream file is recorded by sha256 in `index_sources`,
so a rerun skips what has not changed; `pending_embeddings` holds passages still waiting for a
vector, committed batch by batch, so an interrupted run loses at most one batch. Updates ride on the
dataset's own `artifacts/manifest.json` — a dated snapshot plus one parquet per day, each with a
sha256, verified before it is applied — and `index_state.delta_date` is the watermark. Run `build`
before `update`: a shard carries the snapshot, so ingesting one after a delta would put those
decisions back to their older text (`build` clears the watermark so `update` replays them).

Chunk ids are append-only, because `chunks.id` is the rowid of both the vector table and the
keyword index. A decision that changes has its old passages deleted from all three (the contentless
FTS5 table needs the original text to delete a row) and new ones appended at the end.

The case-law dataset holds decisions only — plus the citation and statute-reference graphs and the
Erwägungen structure. Statute text comes from a second dataset, `voilaj/swiss-legislation`: `--laws`
indexes every federal and cantonal article as passages in the same `chunks` table (`section = 'law'`,
the article's `law_id` in place of a decision id, metadata in `laws`). There is still no commentary
or scholarship.

### What the app serves

The app serves this index by default (`SCA_INDEX=corpus`; `SCA_INDEX=subset` goes back to the 50k
evaluation subset). The current build is `build --laws --since 1980 --fraction 0.25`: **254,146
decisions** (a stratified quarter of everything since 1980), **725,481 statute articles**, 4,925,987
passages and a 57.8 GB index. What changed to serve it:

- **Decisions** are read from the index's `decisions` table on demand (`SqliteDecisionStore`) instead
  of loading a parquet into memory — only the docket lookup is kept in memory (1.1 s at startup).
- **Vector search** runs on an in-memory copy of the vectors (`vecmatrix.py`). sqlite-vec's vec0 scan
  reads every stored vector whatever the filter, so on this index one query took 11–15 s, and
  splitting it into parallel year-range shards made it slower (25 s: the scans only compete for
  memory bandwidth). The matrix — rows sorted by kind and language, 40 GB, exported from vec0's own
  storage tables in about three minutes — answers the same query in ~130 ms with the same passages
  in the same order (vectors are unit length, so the dot product ranks like cosine distance; only
  exact ties between duplicate passages can swap). A search in three languages plus reranking now
  takes ~1.3 s, faster than on the old subset. `build` and `update` re-export it; a matrix older
  than the index is ignored and the app falls back to sqlite-vec, so restart the app after an update.
- **Statute articles** share the passage table but are not decisions: the decision tools skip them
  (the keyword search over-fetches 8× to make up for it — e.g. "Eigenbedarf" matches 24 statute
  passages among its first 64).
- **Statutes are sources.** An answer cites an article the way it cites a decision: its `law_id` in
  the citation's `decision_id`, a verbatim quote from the article. The resolver finds the quote in
  whichever language version holds it, and — because the model once copied the result's label line
  into the quote — drops leading words until the rest is found verbatim (never fewer than 40
  characters). The Source has `section: "law"` and a decision-shaped summary ("Federal law", "Art.
  271a OR", the act's title, the Fedlex link), so chips, the grounding check, the memo and the
  preview show it without a second code path; `GET /api/decisions/{law_id}` serves the article and the
  preview is headed "Statute". Labels normalise the stored article numbers ("ikel 9", ". 1", "§ 9"),
  and fall back to the SR number for the many federal acts without an abbreviation ("Art. 5 SR
  832.10"). The repeat-search guard counts statute searches separately from decision searches.
- **Articles named in an answer are links.** In practice the agent names the articles it read ("Art.
  259d CO") but cites decisions for them: none of the first answers after the statute tools went live
  cited an article. So after each answer, `server/mentions.py` finds every article it names
  ("art. 336c al. 1 let. c CO", "Art. 56 Abs. 1 OR") that the index holds and saves it as a
  `StatuteRef` on the message (`Message.statutes`, also `Issue.statutes` in matters). The answer shows
  them dashed-underlined; clicking one opens the article in the preview. They are references, not
  evidence: kept out of `sources`, not grounding-checked, numbered 0 so the preview does not list them
  as citations, and the linked language version follows the abbreviation used (OR → German, CO →
  French). Answers saved earlier get their links when the conversation is loaded.
- **Filters** (`facets.py`): canton, court, area, year, proceeding and outcome of every decision,
  and canton and act of every statute article, as arrays aligned with the matrix rows
  (`corpus.nemotron-embed.facets.npz`, 40 s to build, tied to the matrix export it was built for;
  `build`/`update` rebuild it, and the app builds it at startup when missing or stale).
- **Citation graph**: `index citations` → `data/graph/corpus.citations.sqlite` (5.0M edges, 719 MB).

Everything cited in conversations and matters saved before the switch is in the new index with
identical full text, so their previews and highlights still line up.

```bash
uv run python -m swiss_court_assistant.vecmatrix status   # is the matrix current?
uv run python -m swiss_court_assistant.vecmatrix export   # re-export by hand
uv run python -m swiss_court_assistant.facets build       # rebuild the search filters by hand
```

Scale, measured on this box: 16.3 passages per decision and ~107 passages/s through the Nemotron
embedder, so the full corpus (~1.07M decisions ≈ 17.5M passages) is about **42 h of embedding and
~143 GB of vectors** for one model, plus the keyword index. A single court, or `--limit`, is the way
to try it first.

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
