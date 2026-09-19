import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type DocumentInfo, type Matter, type MatterEvent, type MatterInput, type MatterStage,
  type Source } from "../api";
import Answer from "./Answer";
import CitedMarkdown from "./CitedMarkdown";
import MatterIntake from "./MatterIntake";
import { toolTitle } from "./Thread";

/** Every issue's cited sources, concatenated and renumbered on one running series — the same
 *  authorities list the memo builds server-side (see `numbered()` in matters.py), so a [n] marker
 *  in the assessment or the memo points at the same source here as it does there. */
function citationSources(matter: Matter): Source[] {
  const out: Source[] = [];
  for (const issue of matter.issues) {
    for (const s of issue.sources ?? []) out.push({ ...s, n: out.length + 1 });
  }
  return out;
}

/** An asset opened in the preview panel as a whole text: no passage is cited, so nothing is highlighted. */
function assetSource(d: DocumentInfo): Source {
  const label = d.kind === "recording" ? "Client recording" : d.kind === "notes" ? "Notes" : "Attached document";
  return {
    n: 0, chunkId: `${d.id}#0`, decisionId: d.id, text: "", section: "document", erwaegungen: [],
    charStart: null, charEnd: null, score: 0,
    decision: { decisionId: d.id, court: "document", courtLabel: label, canton: null, chamber: null, docket: d.name,
      date: d.createdAt.slice(0, 10), language: d.language ?? "en", title: null, regeste: null, legalArea: null,
      sourceUrl: null, pdfUrl: api.documentFileUrl(d.id) },
  };
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

function assetMeta(d: DocumentInfo): string {
  // which model read it is in the tooltip and the preview; the row keeps its width for the name
  if (d.kind === "recording") return `${clock(d.seconds ?? 0)} min`;
  if (d.kind === "notes") return `${d.chars.toLocaleString("en")} characters`;
  return `${d.pages} page${d.pages === 1 ? "" : "s"}`;
}

/** "2 documents · 1 recording · notes" — what the matter was opened on, at a glance. */
function contents(assets: DocumentInfo[]): string {
  const n = (k: string) => assets.filter((a) => (a.kind ?? "document") === k).length;
  const parts = [];
  if (n("document")) parts.push(`${n("document")} document${n("document") === 1 ? "" : "s"}`);
  if (n("recording")) parts.push(`${n("recording")} recording${n("recording") === 1 ? "" : "s"}`);
  if (n("notes")) parts.push("notes");
  return parts.join(" · ");
}

/** Everything the matter holds: what the client handed over, and what the assistant produced from it. */
function CaseFile({ matter, onOpenSource }: { matter: Matter; onOpenSource: (sources: Source[], n: number) => void }) {
  const assets = matter.assets ?? [];
  if (!assets.length && !matter.memo) return null;
  return (
    <section className="matter-block case-file">
      <h2>Case file · {assets.length + (matter.memo ? 1 : 0)} item{assets.length + (matter.memo ? 1 : 0) === 1 ? "" : "s"}</h2>
      {!!matter.indexed && (
        <p className="case-indexed" title="Cut into passages and embedded, so the research searches all of it by meaning">
          Indexed for the research · {matter.indexed.toLocaleString("en")} passage{matter.indexed === 1 ? "" : "s"}
        </p>
      )}
      <ul>
        {assets.map((d) => (
          <li key={d.id} className={`case-asset ${d.kind ?? "document"}`}>
            <div className="case-asset-head">
              <span className="case-kind">{d.kind === "recording" ? "Recording" : d.kind === "notes" ? "Notes" : "Document"}</span>
              <span className="case-name" title={d.name}>{d.name}</span>
              <span className="case-meta" title={`${d.kind === "recording" ? "transcribed" : "read"} with ${d.parser}`}>
                {assetMeta(d)}
              </span>
              <span className="case-actions">
                <button className="link-btn" onClick={() => onOpenSource([assetSource(d)], 0)}>
                  {d.kind === "recording" ? "Transcript" : "Text"}
                </button>
                {d.kind !== "notes" && (
                  <a href={api.documentFileUrl(d.id)} target="_blank" rel="noreferrer" download={d.kind === "recording" || undefined}>
                    {d.kind === "recording" ? "Download" : "Original ↗"}
                  </a>
                )}
              </span>
            </div>
            {d.kind === "recording" && <audio controls preload="none" src={api.documentFileUrl(d.id)} />}
          </li>
        ))}
        {matter.memo && (
          <li className="case-asset generated">
            <div className="case-asset-head">
              <span className="case-kind">Generated</span>
              <span className="case-name">Memo</span>
              <span className="case-meta">drafted {matter.updatedAt.slice(0, 10)}</span>
              <span className="case-actions">
                <a href={api.memoUrl(matter.id)} download>Word</a>
                <a href={api.memoUrl(matter.id, "md")} download>Markdown</a>
              </span>
            </div>
          </li>
        )}
      </ul>
    </section>
  );
}

interface Props {
  matterId: string | null;
  onOpenMatter: (id: string | null) => void;
  onChanged: () => void;
  onOpenSource: (sources: Source[], n: number) => void;
  /** Open the assistant on this matter: its case file and research go with every question. */
  onAsk: () => void;
}

/** The five stages a matter goes through in a firm. The fifth is here to be honest about the edge of
 *  what this app can do: deadlines and client files are practice management, not case law. */
const STAGES: { key: MatterStage | "filing"; label: string; hint: string }[] = [
  { key: "intake", label: "Intake", hint: "The story, and the legal questions hidden in it" },
  { key: "research", label: "Research", hint: "Swiss case law on each question" },
  { key: "assessment", label: "Assessment", hint: "Where the client stands, and what the other side will say" },
  { key: "drafting", label: "Drafting", hint: "A memo in which every proposition carries a citation" },
  { key: "filing", label: "Filing & deadlines", hint: "Practice management — outside this assistant" },
];

type Live = {
  text: Record<number, string>;
  sources: Record<number, Source[]>;
  tools: Record<number, string[]>;
  assessment: string;
};

const EMPTY: Live = { text: {}, sources: {}, tools: {}, assessment: "" };

export default function MatterPage({ matterId, onOpenMatter, onChanged, onOpenSource, onAsk }: Props) {
  const [matter, setMatter] = useState<Matter | null>(null);
  const [live, setLive] = useState<Live>(EMPTY);
  const [stages, setStages] = useState<Record<string, "running" | "done">>({});
  // what the server is doing with an upload; the ASR pass on a long recording is a slow silence
  const [busy, setBusy] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);


  const apply = (ev: MatterEvent) => {
    switch (ev.type) {
      case "stage":
        setStages((s) => ({ ...s, [ev.stage]: ev.status }));
        break;
      case "intake":
        setMatter((m) => m && { ...m, title: ev.title, intake: ev.intake, issues: ev.issues });
        break;
      case "indexed":
        setMatter((m) => m && { ...m, indexed: ev.passages });
        break;
      case "issue_start":
        setLive((l) => ({ ...l, text: { ...l.text, [ev.n]: "" }, sources: { ...l.sources, [ev.n]: [] } }));
        break;
      case "issue_tool":
        setLive((l) => ({
          ...l,
          tools: { ...l.tools, [ev.n]: [...(l.tools[ev.n] ?? []), `${toolTitle(ev.name)} · ${ev.arg}`] },
        }));
        break;
      case "issue_delta":
        setLive((l) => ({ ...l, text: { ...l.text, [ev.n]: (l.text[ev.n] ?? "") + ev.text } }));
        break;
      case "issue_citation":
        setLive((l) => {
          const had = l.sources[ev.n] ?? [];
          return {
            ...l,
            text: { ...l.text, [ev.n]: (l.text[ev.n] ?? "") + `[${ev.source.n}]` },
            sources: { ...l.sources, [ev.n]: had.some((s) => s.n === ev.source.n) ? had : [...had, ev.source] },
          };
        });
        break;
      case "issue_verdict":
        setLive((l) => ({
          ...l,
          sources: {
            ...l.sources,
            [ev.n]: (l.sources[ev.n] ?? []).map((s) => (s.n === ev.source ? { ...s, supported: ev.supported } : s)),
          },
        }));
        break;
      case "issue_done":
        setMatter((m) => m && { ...m, issues: m.issues.map((i) => (i.n === ev.n ? ev.issue : i)) });
        break;
      case "assessment_delta":
        setLive((l) => ({ ...l, assessment: l.assessment + ev.text }));
        break;
      case "done":
        setMatter(ev.matter);
        onChanged();
        break;
      case "error":
        setError(ev.message);
        break;
    }
  };

  const run = useCallback(
    async (id: string) => {
      const ctrl = new AbortController();
      abort.current = ctrl;
      setRunning(true);
      setError(null);
      setStages({ intake: "running" });  // the first event only arrives once the model has answered
      try {
        for await (const ev of api.runMatter(id, ctrl.signal)) apply(ev);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError((e as Error).message);
      } finally {
        setRunning(false);
        onChanged();
      }
    },
    [onChanged],
  );

  // Opening a matter that has not been researched yet starts it. This has to happen here and not in
  // `start()`: selecting the new matter re-runs this effect, which aborts whatever stream is open —
  // a run started before that arrives was cancelled a moment after it began.
  useEffect(() => {
    abort.current?.abort();
    setLive(EMPTY);
    setStages({});
    setError(null);
    setRunning(false);
    if (!matterId) {
      setMatter(null);
      return;
    }
    let current = true;
    api.getMatter(matterId).then(
      (m) => {
        if (!current) return;
        setMatter(m);
        if (m.stage === "done") setStages({ intake: "done", research: "done", assessment: "done", drafting: "done" });
        else if (m.stage === "new") run(m.id);
      },
      (e) => current && setError((e as Error).message),
    );
    return () => {
      current = false;
    };
  }, [matterId, run]);

  const start = async (input: MatterInput) => {
    setBusy("Opening the matter…");
    setError(null);
    try {
      const created = await api.createMatter(input);
      onOpenMatter(created.id);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  /** Run the whole matter again — intake, research, assessment, memo — on the same case file. */
  const rerun = (m: Matter) => {
    if (!window.confirm("Run this case again from the intake? The current research, assessment and memo are replaced.")) return;
    setMatter({ ...m, stage: "intake", assessment: null, memo: null });
    setLive(EMPTY);
    setStages({});
    run(m.id);
  };

  const authorities = useMemo(() => (matter ? citationSources(matter) : []), [matter]);

  if (!matter) return <div className="matter"><MatterIntake busy={busy} error={error} onStart={start} /></div>;

  const done = matter.stage === "done";
  const current = STAGES.find((s) => stages[s.key] === "running")?.key;

  return (
    <div className="matter">
      <div className="matter-inner">
        <header className="matter-head">
          <p className="eyebrow">
            {matter.assets?.length ? contents(matter.assets)
              : { recording: "Recording", document: "Document", text: "Notes", bundle: "Case file" }[matter.sourceKind]}
            {matter.sourceName && !matter.assets?.length ? ` · ${matter.sourceName}` : ""}
            {" · "}{matter.language.toUpperCase()}
          </p>
          <h1>{matter.title}</h1>
          <div className="matter-actions">
            {!running && !done && (
              <button className="btn-primary" onClick={() => run(matter.id)}>
                Run the matter
              </button>
            )}
            {running && (
              <button className="btn-outline" onClick={() => abort.current?.abort()}>
                Stop
              </button>
            )}
            {!running && done && (
              <button className="btn-outline" onClick={() => rerun(matter)}
                title="Run intake, research, assessment and drafting again on the same case file">
                Rerun
              </button>
            )}
            {!running && (
              <button className={done ? "btn-primary" : "btn-outline"} onClick={onAsk}
                title="Open the assistant with this case loaded: its documents, facts, issues and research">
                Ask the assistant about this case
              </button>
            )}
          </div>
        </header>

        <CaseFile matter={matter} onOpenSource={onOpenSource} />

        <ol className="stage-rail">
          {STAGES.map((s) => (
            <li
              key={s.key}
              className={`stage ${s.key === "filing" ? "out" : stages[s.key] ?? (done ? "done" : "")}${
                s.key === current ? " current" : ""
              }`}
            >
              <span className="stage-dot" />
              <span className="stage-label">{s.label}</span>
              <span className="stage-hint">{s.hint}</span>
            </li>
          ))}
        </ol>

        {running && (
          <p className="status-line" aria-live="polite">
            <span className="dot" />
            {stages.research === "running"
              ? `Researching issue ${matter.issues.findIndex((i) => !i.answer) + 1 || matter.issues.length} of ${matter.issues.length}…`
              : stages.assessment === "running"
                ? "Weighing the research…"
                : stages.drafting === "running"
                  ? "Drafting the memo…"
                  : "Reading the matter and naming the legal questions…"}
          </p>
        )}

        {error && <p className="msg-error">{error}</p>}

        {matter.intake && (
          <section className="matter-block" data-select-lang={matter.language} data-select-source="the facts">
            <h2>Facts as understood</h2>
            <p className="facts">{matter.intake.summary}</p>
            <div className="facts-grid">
              {matter.intake.parties.length > 0 && (
                <div>
                  <h3>Parties</h3>
                  <ul>{matter.intake.parties.map((p) => <li key={p}>{p}</li>)}</ul>
                </div>
              )}
              {matter.intake.timeline.length > 0 && (
                <div>
                  <h3>Timeline</h3>
                  <ul>{matter.intake.timeline.map((t) => <li key={t}>{t}</li>)}</ul>
                </div>
              )}
            </div>
          </section>
        )}

        {matter.issues.length > 0 && (
          <section className="matter-block">
            <h2>Issues</h2>
            {matter.issues.map((issue) => {
              const text = issue.answer ?? live.text[issue.n] ?? "";
              const sources = issue.sources ?? live.sources[issue.n] ?? [];
              const tools = live.tools[issue.n] ?? [];
              return (
                <article className="issue" key={issue.n}>
                  <h3>
                    <span className="issue-n">{issue.n}</span>
                    {issue.question}
                  </h3>
                  <p className="issue-why">
                    {issue.why}
                    {issue.area ? <span className="issue-area">{issue.area}</span> : null}
                  </p>
                  {tools.length > 0 && !issue.answer && (
                    <ul className="issue-tools">
                      {tools.map((t, i) => <li key={i}>{t}</li>)}
                    </ul>
                  )}
                  {text ? (
                    <Answer
                      id={`issue-${issue.n}`}
                      text={text}
                      sources={sources}
                      statutes={issue.statutes ?? []}
                      onStatute={(src) => onOpenSource([src], src.n)}
                      language={matter.language}
                      streaming={!issue.answer}
                      activeN={null}
                      onCite={(n) => onOpenSource(sources, n)}
                      selectSource={`issue ${issue.n}`}
                    />
                  ) : (
                    <p className="status-line">
                      <span className="dot" />
                      {running ? "Researching Swiss case law…" : "Not researched yet."}
                    </p>
                  )}
                </article>
              );
            })}
          </section>
        )}

        {(matter.assessment || live.assessment) && (
          <section className="matter-block" data-select-lang={matter.language} data-select-source="the assessment">
            <h2>Assessment</h2>
            <CitedMarkdown
              text={matter.assessment || live.assessment}
              sources={authorities}
              onCite={(n) => onOpenSource(authorities, n)}
            />
          </section>
        )}

        {matter.memo && (
          <section className="matter-block" data-select-lang={matter.language} data-select-source="the memo">
            <h2>Memo</h2>
            <div className="matter-actions">
              <a className="btn-primary" href={api.memoUrl(matter.id)} download>
                Download as Word
              </a>
            </div>
            <div className="memo">
              <CitedMarkdown text={matter.memo} sources={authorities} onCite={(n) => onOpenSource(authorities, n)} />
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
