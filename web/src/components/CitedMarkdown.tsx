import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../api";
import { erwLabel } from "../format";
import { linkCitations } from "./Answer";

interface Props {
  text: string;
  /** The matter's running authorities list — `sources[n - 1]` is what [n] points to. */
  sources: Source[];
  onCite: (n: number) => void;
}

/** Markdown whose [n] markers become clickable citation chips, for text assembled from several
 *  issues (the assessment, the memo) where the markers point into the matter's combined authorities
 *  list rather than a single answer's own sources. */
export default function CitedMarkdown({ text, sources, onCite }: Props) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        const m = href?.match(/^#cite-(\d+)$/);
        if (!m) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
        const n = +m[1];
        const s = sources[n - 1];
        return (
          <button
            className={`cite${s?.supported === false ? " unsupported" : ""}`}
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
    [sources, onCite],
  );

  return (
    <div className="md">
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {linkCitations(text, sources.length)}
      </Markdown>
    </div>
  );
}
