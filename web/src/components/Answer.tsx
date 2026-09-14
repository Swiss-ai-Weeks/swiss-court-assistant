import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../api";
import { erwLabel } from "../format";

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
  text: string;
  sources: Source[];
  streaming: boolean;
  activeN: number | null;
  onCite: (n: number) => void;
}

export default function Answer({ text, sources, streaming, activeN, onCite }: Props) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        const m = href?.match(/^#cite-(\d+)$/);
        if (!m) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
        const n = +m[1];
        const s = sources[n - 1];
        return (
          <button
            className={`cite${activeN === n ? " on" : ""}`}
            onClick={() => onCite(n)}
            title={
              s
                ? `${s.decision.docket} ${erwLabel(s.erwaegungen)}`.trim() + (s.explanation ? `\n${s.explanation}` : "")
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
      <div className="md">
        <Markdown remarkPlugins={[remarkGfm]} components={components}>
          {linkCitations(text, sources.length)}
        </Markdown>
        {streaming && <span className="caret" />}
      </div>
    </div>
  );
}
