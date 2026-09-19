# Swiss Court Assistant

A research assistant for Swiss law. Ask a legal question in German, French, Italian or English, and it
searches Swiss court decisions and federal and cantonal statutes, then answers with every statement
tied to a cited passage. Before the answer is shown, each citation is checked against the passage it
cites, and statements the sources do not support are removed.

- **Assistant:** questions and follow-ups, with citations that open the decision at the quoted
  passage. It asks one clarifying question when the answer depends on a fact you did not give.
- **Documents:** attach a PDF, a Word file, a scan or a photo and ask about it alongside the law.
- **Matters:** hand over a client's documents and recordings and get intake, research, an assessment
  and a memo in Word.
- **Translate, Read aloud and voice mode** for passages and answers.

The corpus is the public [voilaj/swiss-caselaw](https://huggingface.co/datasets/voilaj/swiss-caselaw)
and [voilaj/swiss-legislation](https://huggingface.co/datasets/voilaj/swiss-legislation) datasets.
Every model is a self-hosted NVIDIA NIM, so nothing leaves the machine at runtime.

How it works in detail - the agent, the grounding check, the indexes, the API - is in
[docs/TECHNICAL.md](docs/TECHNICAL.md). A demo script with sample questions and a client case is in
[demo/](demo/README.md).

## Requirements

- Two NVIDIA GPUs with 80 GB or more each (developed on 2× H100 NVL), Docker with the NVIDIA container toolkit
- [uv](https://docs.astral.sh/uv/) and Node.js 22
- An NGC API key to pull the NIM images

## 1. Run the model services

The seven NIMs (LLM, embedding, reranking, document parsing, translation, text-to-speech, speech
recognition) run from one compose file. Put the NGC key in `~/.config/swiss-court-assistant/nim.env`
(`NGC_API_KEY=…`, mode 600), then:

```bash
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps
```

The first start downloads the models and compiles the speech engines, which takes a while (about
30 minutes each for speech recognition and text-to-speech). The header of
[deploy/compose.yaml](deploy/compose.yaml) explains how later starts skip that.

## 2. Build the index

The index holds the decisions, statute articles, their embeddings, the keyword index and the citation
graph. The datasets are downloaded at build time, and the embeddings are computed by the embedding
service from step 1, so start that first. The build is resumable: rerun the same command after an
interruption.

```bash
uv sync
uv run python -m swiss_court_assistant.index build --courts bger,bge --laws   # small start: federal courts and statutes
uv run python -m swiss_court_assistant.index build --laws --since 1980 --fraction 0.25   # what the app serves
uv run python -m swiss_court_assistant.index citations                          # citation graph
uv run python -m swiss_court_assistant.index status
```

Keep it current with the dataset's daily updates:

```bash
uv run python -m swiss_court_assistant.index update
```

Embedding runs at about 100 passages per second, so the full corpus takes around 42 hours. Start with
a few courts.

## 3. Run the app

Build the web UI once, then start the server:

```bash
cd web && npm install && npm run build && cd ..
uv run python -m swiss_court_assistant.server
```

Open http://localhost:8090. On NVIDIA LaunchPad, use
`https://<environment>.apps.launchpad.nvidia.com/coder/proxy/8090/` (with the trailing slash).

For development, run the API with auto-reload and the UI with hot reload (on :5173):

```bash
uv run python -m swiss_court_assistant.server --reload
cd web && npm run dev
```

To record every turn as an MLflow trace, start MLflow (see [docs/TECHNICAL.md](docs/TECHNICAL.md#tracing-the-agent-loop-mlflow))
and set `SCA_MLFLOW_URI=http://localhost:5000` before starting the server.

## 4. Run the evaluation

`agent-eval/` puts Swiss law exam questions and behavioural cases to the running app through its
API, and a local model grades each answer against a reference answer. The app (step 3) and the LLM
service must be running:

```bash
uv run python agent-eval/run.py                    # every case; writes agent-eval/runs/<timestamp>/
uv run python agent-eval/run.py --suite behaviour  # one suite
uv run python agent-eval/report.py                 # re-render the last run's report
```

The cases and scoring are described in [agent-eval/README.md](agent-eval/README.md), and the latest
results are in [agent-eval/RESULTS.md](agent-eval/RESULTS.md). The retrieval evaluation (how often
search finds the right decision) is in [docs/TECHNICAL.md](docs/TECHNICAL.md#retrieval-pipeline).
