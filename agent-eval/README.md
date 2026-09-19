# Evaluating the agent

The retrieval pipeline already has an evaluation: a known-item test set, and metrics that say how
often the gold decision comes back in the top ten (`src/swiss_court_assistant/evaluate.py`). That
measures the search. It says nothing about the thing the user actually reads — the answer the agent
writes, whether it is right about Swiss law, and whether the passages under it say what the answer
claims they say.

This directory measures that. Thirty-five cases in three suites, put to the assistant through its own
HTTP API, graded by a local model against a reference answer written for each case, and scored
alongside checks that need no model at all.

```bash
uv run python agent-eval/run.py                      # every case, both suites
uv run python agent-eval/run.py --suite behaviour     # one suite
uv run python agent-eval/run.py --suite casefile      # the user's own documents: chat + Case Prep
uv run python agent-eval/run.py --only fr- lookup     # cases whose id contains a fragment
uv run python agent-eval/run.py --no-judge            # mechanical checks only
uv run python agent-eval/report.py                    # re-render the last run's report
uv run python agent-eval/run.py --rejudge runs/latest # grade a finished run again, agent untouched
uv run python agent-eval/compare.py runs/A runs/B     # did a change help? headline + flipped cases
```

`--rejudge` is how the judge itself is worked on: it grades the answers already on disk, so a change
to a grading prompt is tested against the same answers in two minutes instead of re-running the
agent for ten.

The server on `:8090` and the LLM NIM on `:9100` must be up; nothing else is needed and nothing
leaves the machine.

## The two suites

**`cases/exam.yaml` — 21 Swiss law exam questions.** Substantive questions of the kind a bar-exam
candidate answers, over the areas this corpus actually covers: tenancy, employment, tort, contract,
persons and family law, criminal law and criminal procedure, debt enforcement, social insurance and
constitutional law. Eleven are in German, six in French, three in Italian, one in English, so the
answer language is exercised as well as the law. Each case carries a `reference` answer written the
way a Swiss lawyer would give it, and a `rubric` of the four to six points the answer has to make.

**`cases/behaviour.yaml` — 12 behavioural cases.** What the assistant has to do besides knowing the
law:

| | what it tests |
|---|---|
| `abstain-eu-gdpr`, `abstain-invented-doctrine`, `abstain-future-ruling`, `unknown-docket` | saying the corpus does not answer the question, instead of assembling an answer out of loosely related passages |
| `lookup-4A-388-2016-outcome` | reading a decision rather than guessing it — the cantonal court annulled the termination, the Federal Supreme Court did the opposite, and an answer from memory gets it backwards |
| `citation-graph-authority` | using the citation index to say whether a precedent is still followed |
| `language-it-cross-lingual`, `language-en-cross-lingual` | answering in the language of the question while citing German and French decisions in their own words |
| `followup-context` | resolving "and how long do I have to challenge *it*?" against the previous turn |
| `false-premise-pregnancy` | contradicting a confident user who has the law wrong |
| `prompt-injection-in-quote` | ignoring an instruction embedded in a passage the user quoted |
| `advice-boundary` | giving the legal framework without promising an outcome |

**`cases/casefile.yaml` — 2 cases on the user's own documents.** The files are in
`fixtures/<name>/` (see `fixtures/README.md`) and are uploaded through `POST /api/documents`, the way
the UI attaches them, so the Nemotron Parse path, the document store and — for Case Prep — the
matter's search index are all exercised.

| | what it tests |
|---|---|
| `casefile-matter-retaliation` | a whole Case Prep run on four files too long to be shown whole: the case file is indexed, each issue's research finds the decisive facts in it (a rent-reduction request deep in an e-mail thread) and cites them to the file, and the law to decisions. The researched issues are graded as one answer. |
| `casefile-chat-notice-period` | a question in the chat about an attached lease whose notice clause is at Art. 19 of 24: the agent has to read or search past the beginning it is shown |

Two more fields and two more checks: `attachments` (files uploaded first) and `mode: matter` (run
Case Prep on them instead of asking `question`); `expect.documents` (attached files the answer must
cite) and `expect.indexed` (the case file must have been indexed). A run deletes what it uploaded —
the documents, the conversation, and the matter with its index collection — unless
`--keep-conversations` is given. The Case Prep case takes eight to ten minutes.

A case is one YAML entry:

```yaml
- id: fr-co-336c-grossesse
  suite: exam
  area: Employment law (CO)
  language: fr          # the language the answer must be written in
  level: basic          # basic | advanced
  question: |           # what is asked
  reference: |          # the model answer, for the judge to grade against
  rubric:               # the points the answer must make, graded one by one
    - "Le congé donné pendant la grossesse est nul, et non simplement abusif (art. 336c al. 2 CO)"
  setup:                # optional earlier turns of the same conversation, asked but not graded
    - "…"
  expect:               # the behavioural expectations, all optional
    abstain: false      # the answer must say the corpus does not answer this
    cites: true         # true | false | optional
    language: fr        # overrides the case's language
    decision: bger_…    # a decision the answer must be based on
    tool: citing_decisions   # a tool the agent must call
```

## How a case is scored

Two kinds of measurement, deliberately kept apart.

**The judge** (`judge.py`) sees the question, the reference answer, the expected behaviour, the
answer under review and the passages it cited, and grades each thing in its own short call: every
rubric point as covered / partial / missing, `grounding` and `usefulness` on 1–5, whether the answer
refused, and whether anything in it is wrong. Grounding is judged *only* against the passages
printed in the prompt, never against the judge's own knowledge of Swiss law, because an answer that
is legally right but cites a passage that does not say so is the failure this eval exists to catch.

**Legal accuracy is not asked for as a score.** Put on a 1–5 scale, this judge marked answers down
for what they left out however plainly it was told not to, and twice invented a rule of Swiss law to
mark a correct answer down with — it "corrected" a right answer on art. 404 al. 2 CO by asserting an
indemnity the provision does not give. So it has to name its objection instead: copy out the one
sentence it says is wrong. The objection counts only if that sentence is really in the answer (a
whitespace-folded substring check, not the judge's word for it) and the reference answer contradicts
it. What survives scores 3 if it is an aside and 2 if it is one of the answer's main propositions;
everything else leaves the answer at 5. The dropped objections are kept in the transcripts with the
reason they were dropped, so the filter itself can be audited.

**The mechanical checks** (`metrics.py`) need no model: the language the answer came back in
(the app's own `detect_language`), whether it cited anything when it should have, whether the
expected decision is among the sources, whether the expected tool ran, how many tool calls and
seconds the turn cost, how many searches simply repeated an earlier one — and two numbers the
assistant produces about itself: the share of citations whose quote was found character for
character in the decision (`verified`), and the share its own grounding check confirmed as stating
the sentence they support (`supported`).

Two headline numbers come out of that:

- **correct** — the strict bar: rubric coverage ≥ 60 %, no legal error the judge could make stick,
  and every behavioural expectation met. This is the pass rate.
- **score** — partial credit out of 100, so a thin but right answer and a wrong one do not land in
  the same place: rubric 40 %, accuracy 20 %, grounding 20 %, usefulness 10 %, behaviour 10 %.

## What comes out

Each run writes `runs/<timestamp>/`, with `runs/latest` pointing at the newest:

```
results.json     every case with its metrics, its judgment and its scores
report.md        the tables below, also printed to the terminal
transcripts/     per case: the answer, every passage it cited, every tool call and the judgment
```

The report gives the headline, then scores by suite, by legal area, by language and by difficulty; a
grounding and behaviour table (citations per answer, verbatim quotes, passages that support their
sentence, language match, abstention); what the turns cost; every case with the reason it failed;
and the judge's comment on each failure. The transcripts are the point of appeal: every grade can be
read back against the answer that earned it.

## Testing a change to the agent

Do not measure a change against the dev server on :8090. It reloads whenever a file under `src/`
changes, so an edit made anywhere while a run is going changes the code half-way through it — the
first measurement of `MIN_TOOL_CALLS` was spoiled exactly that way. Instead serve each arm from its
own copy of `src/`, differing only in the change, on a spare port without `--reload`:

```bash
cp -r src /tmp/ab/A-src && cp -r src /tmp/ab/B-src      # then make the change in one copy only
PYTHONPATH=/tmp/ab/A-src SCA_DB=/tmp/ab/conv.sqlite SCA_MATTERS_DB=/tmp/ab/matters.sqlite \
  uv run python -m swiss_court_assistant.server --host 127.0.0.1 --port 8093
uv run python agent-eval/run.py --server http://127.0.0.1:8093 --out agent-eval/runs/ab-A
```

`PYTHONPATH` puts the copy ahead of the installed package, and the separate conversation databases
keep the runs out of the app's sidebar. Run each arm at least twice, interleaved, and compare with
`compare.py`; the vectors are memory-mapped, so a second server shares them with the first.

The client sends `allowQuestions: false`: the assistant can ask a question back instead of
answering, and a graded case needs the answer. A case that is asked back anyway is recorded as
failed.

## What this measures, and what it does not

The judge is the same model the assistant answers with (`nvidia/nemotron-3.5-lightning`), because
nothing may leave this machine. A model grading its own family's output is a real bias, and it is
why the reference answers are written out in full rather than left to the judge's own knowledge, why
grounding is judged only against the printed passages, why every judgment is split into its own
narrow question, and why all of it is kept in the transcripts. Pass `--judge-url` and
`--judge-model` to grade with a different model — the numbers from two judges on the same run are
worth more than either alone.

**The judge is not deterministic**, even at temperature 0 — the server batches, and the same prompt
comes back differently. Judging the same 33 answers twice with the same prompts reproduced the
correct/incorrect verdict on 87 % of cases and the "no legal error" verdict on 90 %, moved a case's
score by 4.7 points on average (at most 22), and gave the same headline pass rate both times. So
read a difference of a few points between two runs as noise; `--rejudge` twice and compare if a
number matters.

**Neither is the agent**, and on the behavioural suite that is the larger effect. It samples its
research steps at temperature 0.2, and two runs of the same agent scored 63.3 and 70.8 on the
behavioural cases — `abstain-eu-gdpr` scored 5 in one and 47 in the other. The exam suite was
steadier (62.7 and 61.7). `--rejudge` averages out the judge; only running the agent again
averages out the agent. `compare.py` shows the two runs side by side, but before crediting or
blaming a change on the behavioural suite, run the agent more than once.

Beyond that: the reference answers are one lawyer's statement of the law, and where the law is
contested a defensible answer can lose points. Rubric coverage is a lower bound in the other
direction too — the corpus is a subset of Swiss case law, so a point may be unsupportable here even
though it is correct. Thirty-three cases is enough to see where the assistant stands and to catch a
regression that matters; it is not enough for two runs a point apart to mean anything.
