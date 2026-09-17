import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../api";
import { erwLabel } from "../format";
import ReadAloud from "./ReadAloud";

/** Turn [1], [1, 2], [1–3] into markdown links the renderer shows as citation chips. */
function linkCitations(md: string, max: number): string {
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

interface Props {
  id: string;
  text: string;
  sources: Source[];
  /** Language of the question; the answer is read aloud in it. */
  language: string | null;
  streaming: boolean;
  activeN: number | null;
  onCite: (n: number) => void;
}

export default function Answer({ id, text, sources, language, streaming, activeN, onCite }: Props) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
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
    [sources, activeN, onCite],
  );

  return (
    <div className="answer">
      <div className="md" data-select-lang={language ?? "en"} data-select-source="the answer">
        <Markdown remarkPlugins={[remarkGfm]} components={components}>
          {linkCitations(text, sources.length)}
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
