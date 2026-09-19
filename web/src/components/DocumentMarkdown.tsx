import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Element, ElementContent, Root, RootContent } from "hast";

/** A stretch of the document's text and the citations that cover it (Preview's segments). */
export interface Segment {
  text: string;
  ns: number[];
}

interface Props {
  /** The document's text cut at every cited passage, in order; joined, they are the whole text. */
  parts: Segment[];
  activeN: number;
  onSelect: (n: number) => void;
}

// Private-use characters mark where a cited passage opens (with its segment's index) and closes. They
// go into the text before it is parsed, so the offsets the citations carry need no mapping onto the
// rendered Markdown; the rehype step below turns them into <mark> elements.
const OPEN = "\uE000";
const MID = "\uE001";
const CLOSE = "\uE002";
const MARKER = new RegExp(`${OPEN}(\\d+)${MID}|${CLOSE}`, "g");

/** Nemotron Parse writes tables as LaTeX: `\begin{tabular}{ccc} a & b\\ … \end{tabular}`. As a GFM table,
 *  with the first row as its header when every cell of it is bold, else an empty header. */
function tabular(body: string): string {
  const rows = body
    .split(/\\\\/)
    .map((r) => r.replace(/\\hline/g, "").trim())
    .filter(Boolean)
    .map((r) => r.split(/(?<!\\)&/).map((c) => c.trim().replace(/\|/g, "\\|").replace(/\n+/g, " ")));
  if (!rows.length) return "";
  const width = Math.max(...rows.map((r) => r.length));
  const pad = (r: string[]) => [...r, ...Array(width - r.length).fill("")];
  const bold = (c: string) => /^\*\*[^*].*\*\*$/.test(c.replace(MARKER, ""));
  const header = rows[0].every((c) => !c || bold(c)) ? pad(rows.shift()!) : Array(width).fill(" ");
  const line = (r: string[]) => `| ${pad(r).join(" | ")} |`;
  return ["", line(header), `|${" --- |".repeat(width)}`, ...rows.map(line), ""].join("\n");
}

/** The parser's text as Markdown: tables, page labels, and its single line breaks kept (an address is
 *  written one line per line, which Markdown would run together). */
function toMarkdown(text: string): string {
  return text
    .split(/(\\begin\{tabular\}\{[^}]*\}[\s\S]*?\\end\{tabular\})/)
    .map((chunk, i) => {
      if (i % 2) return tabular(chunk.replace(/^\\begin\{tabular\}\{[^}]*\}/, "").replace(/\\end\{tabular\}$/, ""));
      return chunk
        // a passage that starts a line opens after the line's Markdown, or "## Title" is no longer a heading
        .replace(new RegExp(`^((?:${OPEN}\\d+${MID})+)(#{1,6} |[-*+] |\\d+[.)] |> )`, "gm"), "$2$1")
        .replace(/^\[Page (\d+)\]\s*$/gm, "###### Page $1")
        .replace(/([^\n])\n(?=[^\n])/g, "$1  \n");
    })
    .join("\n");
}

/** Turn the markers into <mark> elements: every text node between an opening and a closing marker is
 *  wrapped, so a passage that runs across bold text or table cells is highlighted all along. */
function rehypeMarks() {
  return (tree: Root) => {
    let open: number | null = null;
    const walk = (node: Root | Element) => {
      const out: (RootContent | ElementContent)[] = [];
      for (const child of node.children) {
        if (child.type !== "text") {
          if (child.type === "element") walk(child);
          out.push(child);
          continue;
        }
        let at = 0;
        const emit = (text: string) => {
          if (!text) return;
          // whitespace between table cells or list items is not text to highlight (a <mark> in a <tr>)
          out.push(open === null || !text.trim() ? { type: "text", value: text }
            : { type: "element", tagName: "mark", properties: { dataSeg: open },
                children: [{ type: "text", value: text }] });
        };
        for (const m of child.value.matchAll(MARKER)) {
          emit(child.value.slice(at, m.index));
          open = m[1] === undefined ? null : +m[1];
          at = m.index + m[0].length;
        }
        emit(child.value.slice(at));
      }
      node.children = out as typeof node.children;
    };
    walk(tree);
  };
}

/** An attached document in the preview, rendered as the Markdown the parser wrote, with the cited
 *  passages highlighted like in a decision. */
export default function DocumentMarkdown({ parts, activeN, onSelect }: Props) {
  const source = useMemo(
    () => toMarkdown(parts.map((p, i) => (p.ns.length ? `${OPEN}${i}${MID}${p.text}${CLOSE}` : p.text)).join("")),
    [parts],
  );
  const components = useMemo<Components>(
    () => ({
      mark({ node }) {
        const i = Number(node?.properties?.dataSeg);
        const p = parts[i];
        if (!p) return null;
        const text = node?.children.map((c) => (c.type === "text" ? c.value : "")).join("") ?? "";
        const on = p.ns.includes(activeN);
        // the first piece of the active passage is the scroll target
        const first = on && !parts[i - 1]?.ns.includes(activeN);
        return (
          <mark className={`hl${on ? " active" : ""}`} data-active={first || undefined}
            title={`Passage ${p.ns.join(", ")}`} onClick={() => onSelect(p.ns[0])}>
            {text}
          </mark>
        );
      },
      a({ href, children }) {
        return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
      },
    }),
    [parts, activeN, onSelect],
  );
  return (
    <div className="md doc-md">
      <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeMarks]} components={components}>
        {source}
      </Markdown>
    </div>
  );
}
