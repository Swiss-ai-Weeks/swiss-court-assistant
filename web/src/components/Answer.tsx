import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source, StatuteRef } from "../api";
import { erwLabel } from "../format";
import ReadAloud from "./ReadAloud";

/** Turn [1], [1, 2], [1–3] into markdown links the renderer shows as citation chips. */
export function linkCitations(md: string, max: number): string {
  return md.replace(/\[(\d+(?:\s*[-–,;]\s*\d+)*)\](?!\()/g, (whole, inner: string) => {
    const nums: number[] = [];
    for (const part of inner.split(/\s*[,;]\s*/)) {
      const [a, b] = part.split(/\s*[-–]\s*/).map(Number);
      for (let n = a; n <= (b || a); n++) nums.push(n);
    }
    if (nums.some((n) => n < 1 || n > max)) return whole;
    return nums.map((n) => `[${n}](#cite-${n})`).join("");
  });
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Turn each article the answer names ("Art. 259d CO") into a link to its text, outside existing links. */
function linkStatutes(md: string, statutes: StatuteRef[]): string {
  if (!statutes.length) return md;
  // longest first, so "Art. 56 Abs. 1 OR" is not broken up by a shorter mention inside it
  const order = statutes.map((s, i) => ({ text: s.text, i })).sort((a, b) => b.text.length - a.text.length);
  const pattern = new RegExp(`(?<![\\w.])(${order.map((o) => escape(o.text)).join("|")})(?![\\w])`, "g");
  const index = new Map(order.map((o) => [o.text, o.i]));
  // leave markdown links (the citation chips) alone
  return md
    .split(/(\[[^\]]*\]\([^)]*\))/)
    .map((part, k) => (k % 2 ? part : part.replace(pattern, (m) => `[${m}](#law-${index.get(m)})`)))
    .join("");
}

interface Props {
  id: string;
  text: string;
  sources: Source[];
  /** Articles named in the text, linked to the statute; clicking one calls onStatute. */
  statutes?: StatuteRef[];
  onStatute?: (source: Source) => void;
  /** Language of the question; the answer is read aloud in it. */
  language: string | null;
  streaming: boolean;
  activeN: number | null;
  onCite: (n: number) => void;
}

export default function Answer({ id, text, sources, statutes = [], onStatute, language, streaming, activeN,
  onCite }: Props) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        const law = href?.match(/^#law-(\d+)$/);
        if (law) {
          const ref = statutes[+law[1]];
          return (
            <button className="statute-link" onClick={() => ref && onStatute?.(ref.source)}
              title={ref ? `${ref.source.decision.docket} — open the statute text` : undefined}>
              {children}
            </button>
          );
        }
        const m = href?.match(/^#cite-(\d+)$/);
        if (!m) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
        const n = +m[1];
        const s = sources[n - 1];
        return (
          <button
            className={`cite${activeN === n ? " on" : ""}${s?.supported === false ? " unsupported" : ""}`}
            onClick={() => onCite(n)}
            title={
              s
                ? `${s.decision.docket} ${erwLabel(s.erwaegungen)}`.trim() +
                  (s.explanation ? `\n${s.explanation}` : "") +
                  (s.supported === false ? "\n⚠ Checked: this passage does not state that sentence." : "")
                : undefined
            }
            aria-label={`Open source ${n}`}
          >
            {n}
          </button>
        );
      },
    }),
    [sources, statutes, onStatute, activeN, onCite],
  );

  return (
    <div className="answer">
      <div className="md" data-select-lang={language ?? "en"} data-select-source="the answer">
        <Markdown remarkPlugins={[remarkGfm]} components={components}>
          {linkStatutes(linkCitations(text, sources.length), streaming ? [] : statutes)}
        </Markdown>
        {streaming && <span className="caret" />}
      </div>
      {!streaming && language && (
        <div className="answer-actions">
          <ReadAloud id={`answer:${id}`} text={text} language={language} />
        </div>
      )}
    </div>
  );
}
