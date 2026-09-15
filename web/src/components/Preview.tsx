import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Decision, type Source } from "../api";
import { erwLabel, formatDate, langName } from "../format";
import ReadAloud from "./ReadAloud";

interface Props {
  /** Every source of the selected answer; the preview shows the decision of `activeN`. */
  sources: Source[];
  activeN: number;
  /** Language of the conversation turn; passages in another language can be translated into it. */
  language: string | null;
  onSelect: (n: number) => void;
  onClose: () => void;
}

interface Segment {
  text: string;
  ns: number[];
}

const HEADING = /^\s*\d+(\.\d+)*\.?\s*$/;
const CITE_START = /^\s*(art\.|Art\.|BGE|ATF|DTF|consid\.|E\.)/;

/** The scraped texts put statute references on their own lines ("(\nart. 9 Cst.\n), ce qui").
 *  Join those breaks with a space for display. Same length, so passage offsets stay valid. */
function joinCitationBreaks(text: string): string {
  const lines = text.split("\n");
  let out = lines[0];
  for (let i = 1; i < lines.length; i++) {
    const prev = lines[i - 1];
    const next = lines[i];
    const join =
      prev.trim() !== "" &&
      next.trim() !== "" &&
      !HEADING.test(prev) &&
      (/[(\[']\s*$/.test(prev) || /^\s*[),;.:\]]/.test(next) || CITE_START.test(next) ||
        (CITE_START.test(prev) && prev.length < 60));
    out += (join ? " " : "\n") + next;
  }
  return out;
}

/** Cut the text at every passage boundary; each segment knows which passages cover it. */
function segments(text: string, ranges: { n: number; start: number; end: number }[]): Segment[] {
  const cuts = new Set([0, text.length]);
  for (const r of ranges) cuts.add(r.start).add(r.end);
  const pts = [...cuts].filter((p) => p >= 0 && p <= text.length).sort((a, b) => a - b);
  const out: Segment[] = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    out.push({ text: text.slice(a, b), ns: ranges.filter((r) => r.start <= a && r.end >= b).map((r) => r.n) });
  }
  return out;
}

// Translations outlive the component, so switching between citations does not refetch them.
const translations = new Map<string, string>();

type Panel = "explain" | "translate" | null;

/** Explain / Translate buttons shown right under the highlighted passage. */
function RefTools({ source, language }: { source: Source; language: string | null }) {
  const [panel, setPanel] = useState<Panel>(null);
  const [translation, setTranslation] = useState<{ state: "loading" | "done" | "error"; text: string } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const from = source.decision.language;
  const target = language && language !== from ? language : null;
  const key = `${source.decisionId}:${source.charStart ?? source.section}:${source.text.length}:${target}`;

  useEffect(() => {
    box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [panel, translation?.state]);

  const toggle = (p: Panel) => setPanel((cur) => (cur === p ? null : p));

  const translate = () => {
    toggle("translate");
    if (!target || translation?.state === "loading" || translation?.state === "done") return;
    const cached = translations.get(key);
    if (cached) {
      setTranslation({ state: "done", text: cached });
      return;
    }
    setTranslation({ state: "loading", text: "" });
    api.translate(source.text, from, target).then(
      (t) => {
        translations.set(key, t);
        setTranslation({ state: "done", text: t });
      },
      (e: Error) => setTranslation({ state: "error", text: e.message }),
    );
  };

  return (
    <div className="ref-tools">
      <div className="ref-actions">
        <button className={`ref-btn${panel === "explain" ? " on" : ""}`} onClick={() => toggle("explain")}
          aria-expanded={panel === "explain"}>
          Explain
        </button>
        {target && (
          <button className={`ref-btn${panel === "translate" ? " on" : ""}`} onClick={translate}
            aria-expanded={panel === "translate"}>
            Translate to {langName(target)}
          </button>
        )}
        <ReadAloud id={`passage:${key}`} text={source.text} language={from} className="lg" />
        {source.verified === false && <span className="ref-warn">Quoted words not found verbatim in this decision</span>}
      </div>
      {panel === "explain" && (
        <div className="ref-box" ref={box}>
          <div className="block-label">
            <span className="tag">Why [{source.n}] is cited</span>
            {source.explanation && language && (
              <ReadAloud id={`explain:${key}`} text={source.explanation} language={language} />
            )}
          </div>
          <p>{source.explanation || "The agent gave no explanation for this citation."}</p>
        </div>
      )}
      {panel === "translate" && target && translation && (
        <div className="ref-box" ref={box}>
          <div className="block-label">
            <span className="tag">
              Machine translation · {langName(from)} → {langName(target)}
            </span>
            {translation.state === "done" && (
              <ReadAloud id={`translation:${key}`} text={translation.text} language={target} />
            )}
          </div>
          {translation.state === "loading" && <p className="muted">Translating…</p>}
          {translation.state === "done" && <p>{translation.text}</p>}
          {translation.state === "error" && <p className="ref-err">Translation failed: {translation.text}</p>}
        </div>
      )}
    </div>
  );
}

export default function Preview({ sources, activeN, language, onSelect, onClose }: Props) {
  const active = sources.find((s) => s.n === activeN);
  const decisionId = active?.decisionId;
  const [doc, setDoc] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!decisionId) return;
    let live = true;
    setError(null);
    if (doc?.decisionId !== decisionId) setDoc(null);
    api.getDecision(decisionId).then(
      (d) => live && setDoc(d),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decisionId]);

  const cited = useMemo(() => sources.filter((s) => s.decisionId === decisionId), [sources, decisionId]);
  const parts = useMemo(() => {
    if (!doc) return [];
    const ranges = cited
      .filter((s) => s.charStart !== null && s.charEnd !== null)
      .map((s) => ({ n: s.n, start: s.charStart!, end: s.charEnd! }));
    return segments(joinCitationBreaks(doc.fullText), ranges);
  }, [doc, cited]);

  // Bring the active passage into view: jump when a new text renders, glide within the same text.
  const shownParts = useRef(parts);
  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const el = root.querySelector<HTMLElement>('[data-active="true"]');
    const top = el ? el.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop - 24 : 0;
    root.scrollTo({ top, behavior: shownParts.current === parts ? "smooth" : "auto" });
    shownParts.current = parts;
  }, [parts, activeN]);

  if (!active) return null;
  const d = doc ?? active.decision;
  const regesteActive = active.section === "regeste";
  const lastActive = parts.reduce((last, p, i) => (p.ns.includes(activeN) ? i : last), -1);
  const tools = <RefTools key={activeN} source={active} language={language} />;
  // selected text can be read aloud, translated into the conversation's language, or quoted in a reply
  const selectable = {
    "data-select-lang": d.language,
    "data-select-target": language ?? undefined,
    "data-select-source": `${d.courtLabel} ${d.docket}`,
  };

  return (
    <aside className="preview" aria-label="Source decision">
      <div className="preview-bar">
        Source decision
        <span className="spacer" />
        <button className="icon-btn" onClick={onClose} aria-label="Close preview" title="Close">
          ×
        </button>
      </div>
      <div className="preview-scroll" ref={scroller}>
        <header className="preview-head">
          <span className="corner-square" />
          <div className="badges">
            <span className="badge">{d.courtLabel}</span>
            <span className="badge">{langName(d.language)}</span>
            {d.legalArea && <span className="badge outline">{d.legalArea}</span>}
          </div>
          <h2>{d.docket}</h2>
          {(d.title || d.chamber) && <p className="court">{d.title ?? d.chamber}</p>}
          <dl className="meta-grid">
            <dt>Decided</dt>
            <dd>{formatDate(d.date)}</dd>
            {d.chamber && (
              <>
                <dt>Chamber</dt>
                <dd>{d.chamber}</dd>
              </>
            )}
            <dt>Decision ID</dt>
            <dd>{d.decisionId}</dd>
          </dl>
          <div className="preview-links">
            {d.sourceUrl && (
              <a href={d.sourceUrl} target="_blank" rel="noreferrer">
                Official source ↗
              </a>
            )}
            {d.pdfUrl && (
              <a href={d.pdfUrl} target="_blank" rel="noreferrer">
                PDF ↗
              </a>
            )}
          </div>
        </header>

        <div className="cited-in">
          Cited in this answer:
          {cited.map((s) => (
            <button key={s.n} className={`jump${s.n === activeN ? " on" : ""}`} onClick={() => onSelect(s.n)}>
              [{s.n}] {erwLabel(s.erwaegungen) || s.section}
            </button>
          ))}
        </div>

        {/* no highlight to attach the buttons to (quote not located): show them here */}
        {doc && !regesteActive && lastActive < 0 && <div className="ref-tools-top">{tools}</div>}

        {d.regeste && (
          <div className="regeste" data-active={regesteActive || undefined} {...selectable}>
            <div className="block-label">
              <span className="tag">Regeste</span>
            </div>
            {regesteActive ? (
              <>
                <mark className="hl active">{d.regeste}</mark>
                {tools}
              </>
            ) : (
              d.regeste
            )}
          </div>
        )}

        {error && <p className="preview-empty">Could not load the decision: {error}</p>}
        {!doc && !error && <p className="preview-empty">Loading decision…</p>}
        {doc && (
          <div className="doc-text" {...selectable}>
            {parts.map((p, i) => {
              const tail = i === lastActive && !regesteActive ? tools : null;
              if (!p.ns.length) return <span key={i}>{p.text}</span>;
              const on = p.ns.includes(activeN);
              // mark only the first segment of the active passage as the scroll target
              const first = on && !parts[i - 1]?.ns.includes(activeN);
              return (
                <span key={i}>
                  <mark
                    className={`hl${on ? " active" : ""}`}
                    data-active={first || undefined}
                    title={`Passage ${p.ns.join(", ")}`}
                    onClick={() => onSelect(p.ns[0])}
                  >
                    {p.text}
                  </mark>
                  {tail}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
