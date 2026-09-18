import { useEffect, useRef } from "react";
import type { Message, Source, Stage, StatuteRef, ToolCall } from "../api";
import Answer from "./Answer";
import { QuoteCard } from "./Composer";

export interface PendingTurn {
  status: { stage: Stage; detail: string } | null;
  tools: ToolCall[];
  sources: Source[];
  content: string;
  /** Reasoning streamed since the last tool call. */
  thinking?: string;
  language?: string;
  error?: string;
}

export interface Selection {
  messageId: string;
  n: number;
}

interface Props {
  messages: Message[];
  pending: PendingTurn | null;
  selection: Selection | null;
  onOpenSource: (messageId: string, n: number) => void;
  /** An article named in an answer's text (not a numbered citation). */
  onOpenStatute: (source: Source) => void;
}

const PENDING_ID = "pending";

export default function Thread({ messages, pending, selection, onOpenSource, onOpenStatute }: Props) {
  const end = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const scroller = useRef<HTMLDivElement>(null);

  // Follow the stream only while the user is at the bottom.
  useEffect(() => {
    if (pinned.current) end.current?.scrollIntoView({ block: "end" });
  }, [messages.length, pending?.content, pending?.tools, pending?.status]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  return (
    <div className="thread" ref={scroller} onScroll={onScroll}>
      <div className="thread-inner">
        {messages.map((m) =>
          m.role === "user" ? (
            <UserTurn key={m.id} text={m.content} />
          ) : (
            <AssistantTurn
              key={m.id}
              id={m.id}
              content={m.content}
              sources={m.sources ?? []}
              statutes={m.statutes ?? []}
              onOpenStatute={onOpenStatute}
              tools={m.toolCalls ?? []}
              language={m.language ?? null}
              streaming={false}
              activeN={selection?.messageId === m.id ? selection.n : null}
              onOpenSource={onOpenSource}
            />
          ),
        )}
        {pending && (
          <AssistantTurn
            id={PENDING_ID}
            content={pending.content}
            sources={pending.sources}
            tools={pending.tools}
            status={pending.status}
            thinking={pending.thinking}
            error={pending.error}
            language={pending.language ?? null}
            streaming
            activeN={selection?.messageId === PENDING_ID ? selection.n : null}
            onOpenSource={onOpenSource}
          />
        )}
        <div ref={end} />
      </div>
    </div>
  );
}

function UserTurn({ text }: { text: string }) {
  // "> " lines quote a selection the user replied to (see SelectionTools)
  const lines = text.split("\n");
  const quote = lines.filter((l) => l.startsWith(">")).map((l) => l.replace(/^>\s?/, ""));
  const source = quote.length > 1 && quote[quote.length - 1].startsWith("— ") ? quote.pop()!.slice(2) : null;
  const rest = lines.filter((l) => !l.startsWith(">")).join("\n").trim();
  return (
    <div className="msg msg-user">
      <div className="bubble">
        {quote.length > 0 && <QuoteCard text={quote.join("\n")} source={source} />}
        {rest}
      </div>
    </div>
  );
}

/** What each tool looks at — the court decisions or the statutes — and what it does there. The two
 *  searches work the same way, so the label says how and the tag says where. */
type Domain = "case" | "statute";
const TOOLS: Record<string, { domain: Domain; label: string }> = {
  semantic_search: { domain: "case", label: "Search by meaning" },
  keyword_search: { domain: "case", label: "Search exact words" },
  read_decision: { domain: "case", label: "Read decision" },
  citing_decisions: { domain: "case", label: "Who cites it" },
  search_laws: { domain: "statute", label: "Search by meaning" },
  read_law: { domain: "statute", label: "Read article" },
  search_decisions: { domain: "case", label: "Search exact words" },
  count_decisions: { domain: "case", label: "Count" },
};
const DOMAIN_NAME: Record<Domain, string> = { case: "Case law", statute: "Statutes" };

export const TOOL_LABEL: Record<string, string> = Object.fromEntries(
  Object.entries(TOOLS).map(([name, t]) => [name, t.label]),
);

/** "Statutes · Read article", for places that show a tool call as one line of text. */
export function toolTitle(name: string): string {
  const t = TOOLS[name];
  return t ? `${DOMAIN_NAME[t.domain]} · ${t.label}` : name;
}

function DomainTag({ name }: { name: string }) {
  const t = TOOLS[name];
  return t ? <span className={`src-tag ${t.domain}`}>{DOMAIN_NAME[t.domain]}</span> : null;
}

/** "3 steps in case law · 1 in statutes" — where the research went, at a glance. */
function researchSummary(calls: ToolCall[]): string {
  const count = (d: Domain) => calls.filter((c) => TOOLS[c.name]?.domain === d).length;
  const cases = count("case");
  const statutes = count("statute");
  const other = calls.length - cases - statutes;
  const steps = (n: number) => `${n} step${n === 1 ? "" : "s"}`;
  const parts = [];
  if (cases) parts.push(`${steps(cases)} in case law`);
  if (statutes) parts.push(`${parts.length ? statutes : steps(statutes)} in statutes`);
  if (other) parts.push(`${steps(other)} elsewhere`);
  return parts.join(" · ");
}

function toolArg(c: ToolCall): string {
  const a = c.args;
  if (c.name === "read_law" && a.article)
    return `Art. ${a.article} ${a.code ?? ""}${a.canton && a.canton !== "CH" ? ` (${a.canton})` : ""}`;
  const perLanguage = ["de", "fr", "it"].filter((l) => a[`query_${l}`]).map((l) => `${l.toUpperCase()} ${a[`query_${l}`]}`);
  const main = perLanguage.length
    ? perLanguage.join(" · ")
    : (a.query ?? a.keyword ?? a.decision_id ?? Object.values(a)[0] ?? "");
  const offset = typeof a.offset === "number" && a.offset > 0 ? ` · from character ${a.offset.toLocaleString("en")}` : "";
  const cantonal = c.name === "search_laws" && a.cantonal ? " · incl. cantonal law" : "";
  return `${String(main)}${offset}${cantonal}`;
}

function Activity({ calls, live }: { calls: ToolCall[]; live: boolean }) {
  return (
    <details className="activity" open={live || undefined}>
      <summary>Research · {researchSummary(calls)}</summary>
      <ol>
        {calls.map((c) => {
          const state = c.error ? "error" : c.summary == null ? "running" : "done";
          return (
            <li key={c.id} className={state}>
              {c.thought && (
                <span className="thought" title={c.thought}>
                  {c.thought}
                </span>
              )}
              <span className="dot" />
              <DomainTag name={c.name} />
              <span className="tool">{TOOL_LABEL[c.name] ?? c.name}</span>
              <span className="arg">{toolArg(c)}</span>
              {c.summary != null && <span className="result">{c.summary}</span>}
            </li>
          );
        })}
      </ol>
    </details>
  );
}

interface TurnProps {
  id: string;
  content: string;
  sources: Source[];
  statutes?: StatuteRef[];
  onOpenStatute?: (source: Source) => void;
  tools: ToolCall[];
  status?: PendingTurn["status"];
  thinking?: string;
  error?: string;
  language: string | null;
  streaming: boolean;
  activeN: number | null;
  onOpenSource: (messageId: string, n: number) => void;
}

function AssistantTurn({ id, content, sources, statutes, onOpenStatute, tools, status, thinking, error, language,
  streaming, activeN, onOpenSource }: TurnProps) {
  const open = (n: number) => onOpenSource(id, n);
  const thought = streaming && !content && !error && thinking ? thinking.replace(/\s+/g, " ").trim() : "";
  return (
    <div className="msg msg-assistant">
      {tools.length > 0 && <Activity calls={tools} live={streaming} />}
      {thought && (
        // one grey line; its newest words stay in view while older ones scroll out on the left
        <p className="thinking-line" title={thought}>
          <span className="thinking-label">Thinking</span>
          <span className="thinking-text">
            <span>{thought.slice(-400)}</span>
          </span>
        </p>
      )}
      {streaming && !content && !error && status && !thought && (
        <p className="status-line" aria-live="polite">
          <span className="dot" />
          {status.detail}
        </p>
      )}
      {content && (
        <Answer id={id} text={content} sources={sources} statutes={statutes} onStatute={onOpenStatute}
          language={language} streaming={streaming} activeN={activeN} onCite={open} />
      )}
      {error && <p className="msg-error">{error}</p>}
    </div>
  );
}
