import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { langName } from "../format";
import ReadAloud from "./ReadAloud";

/** Text the user selected, for a reply that quotes it. */
export interface Quote {
  text: string;
  language: string;
  /** Where it comes from, e.g. "BGer 4A_19/2016" or "the answer". */
  source: string | null;
}

interface Selected extends Quote {
  /** Language to translate into; null when the text is already in it. */
  target: string | null;
  rect: DOMRect;
}

const MAX_CHARS = 8000; // the translate endpoint's limit
const translations = new Map<string, string>();

/** The current selection, when it lies inside one element marked with data-select-lang. */
function readSelection(): Selected | null {
  const s = window.getSelection();
  if (!s || s.isCollapsed || !s.rangeCount) return null;
  const text = s.toString().replace(/\s+/g, " ").trim();
  if (text.length < 2) return null;
  const range = s.getRangeAt(0);
  const node = range.commonAncestorContainer;
  const box = (node instanceof Element ? node : node.parentElement)?.closest<HTMLElement>("[data-select-lang]");
  if (!box) return null;
  const rect = range.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > window.innerHeight) return null; // scrolled out of view
  const { selectLang: language = "en", selectTarget: target, selectSource: source } = box.dataset;
  return {
    text: text.slice(0, MAX_CHARS),
    language,
    target: target && target !== language ? target : null,
    source: source ?? null,
    rect,
  };
}

/** Floating Read aloud / Translate / Reply buttons over text selected in the answer or the decision.
 *  `replyLabel` names what the reply does where it is shown ("Ask assistant" in Case Prep). */
export default function SelectionTools({ onReply, replyLabel = "Reply" }: {
  onReply: (q: Quote) => void;
  replyLabel?: string;
}) {
  const [sel, setSel] = useState<Selected | null>(null);
  const [translation, setTranslation] = useState<{ of: string; state: "loading" | "done" | "error"; text: string } | null>(null);
  const bar = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let timer = 0;
    const update = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setSel(readSelection()), 120);
    };
    document.addEventListener("selectionchange", update);
    window.addEventListener("scroll", update, true);
    window.addEventListener("resize", update);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("selectionchange", update);
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("resize", update);
    };
  }, []);

  if (!sel) return null;

  const key = `${sel.language}:${sel.target}:${sel.text}`;
  const shown = translation?.of === key ? translation : null;
  // above the selection, unless there is no room (under the nav bar) or a translation box has to open
  // downwards into the page
  const below = sel.rect.top < 110 || !!shown;
  const style = {
    top: below ? sel.rect.bottom + 8 : sel.rect.top - 8,
    left: Math.min(Math.max(sel.rect.left + sel.rect.width / 2, 240), window.innerWidth - 240),
  };

  const translate = () => {
    if (!sel.target || shown) return setTranslation(null);
    const cached = translations.get(key);
    if (cached) return setTranslation({ of: key, state: "done", text: cached });
    setTranslation({ of: key, state: "loading", text: "" });
    api.translate(sel.text, sel.language, sel.target).then(
      (t) => {
        translations.set(key, t);
        setTranslation((cur) => (cur?.of === key ? { of: key, state: "done", text: t } : cur));
      },
      (e: Error) => setTranslation((cur) => (cur?.of === key ? { of: key, state: "error", text: e.message } : cur)),
    );
  };

  const reply = () => {
    onReply({ text: sel.text, language: sel.language, source: sel.source });
    window.getSelection()?.removeAllRanges();
    setSel(null);
  };

  return (
    <div
      ref={bar}
      className={`sel-tools${below ? " below" : ""}`}
      style={style}
      // keep the selection: a mousedown here would otherwise collapse it before the click lands
      onMouseDown={(e) => e.preventDefault()}
    >
      <div className="sel-bar" role="toolbar" aria-label="Selected text">
        <ReadAloud id={`selection:${key}`} text={sel.text} language={sel.language} />
        {sel.target && (
          <button className={shown ? "on" : ""} onClick={translate} aria-expanded={!!shown}>
            Translate to {langName(sel.target)}
          </button>
        )}
        <button onClick={reply}>{replyLabel}</button>
      </div>
      {shown && sel.target && (
        <div className="sel-box">
          <div className="block-label">
            <span className="tag">
              Machine translation · {langName(sel.language)} → {langName(sel.target)}
            </span>
            {shown.state === "done" && <ReadAloud id={`selection-tr:${key}`} text={shown.text} language={sel.target} />}
          </div>
          {shown.state === "loading" && <p className="muted">Translating…</p>}
          {shown.state === "done" && <p>{shown.text}</p>}
          {shown.state === "error" && <p className="ref-err">Translation failed: {shown.text}</p>}
        </div>
      )}
    </div>
  );
}
