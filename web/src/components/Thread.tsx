import { useEffect, useRef } from "react";
import type { Clarification, DocumentInfo, Message, Source, Stage, StatuteRef, ToolCall } from "../api";
import Answer from "./Answer";
import { AttachmentChip, QuoteCard } from "./Composer";

export interface PendingTurn {
  status: { stage: Stage; detail: string } | null;
  tools: ToolCall[];
  sources: Source[];
  content: string;
  /** Reasoning streamed since the last tool call. */
  thinking?: string;
  language?: string;
  error?: string;
  clarification?: Clarification;
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
  /** Answer a question the assistant asked back (one of its suggested answers). */
  onReply?: (text: string) => void;
}

const PENDING_ID = "pending";

export default function Thread({ messages, pending, selection, onOpenSource, onOpenStatute, onReply }: Props) {
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
        {messages.map((m, i) =>
          m.role === "user" ? (
            <UserTurn key={m.id} text={m.content} attachments={m.attachments ?? []} />
          ) : (
            <AssistantTurn
              key={m.id}
              id={m.id}
              content={m.content}
              sources={m.sources ?? []}
              statutes={m.statutes ?? []}
              onOpenStatute={onOpenStatute}
              tools={m.toolCalls ?? []}
              clarification={m.clarification ?? undefined}
              onReply={i === messages.length - 1 && !pending ? onReply : undefined}
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
            clarification={pending.clarification}
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

function UserTurn({ text, attachments }: { text: string; attachments: DocumentInfo[] }) {
  // "> " lines quote a selection the user replied to (see SelectionTools)
  const lines = text.split("\n");
  const quote = lines.filter((l) => l.startsWith(">")).map((l) => l.replace(/^>\s?/, ""));
  const source = quote.length > 1 && quote[quote.length - 1].startsWith("- ") ? quote.pop()!.slice(2) : null;
  const rest = lines.filter((l) => !l.startsWith(">")).join("\n").trim();
  return (
    <div className="msg msg-user">
      <div className="bubble">
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((d) => <AttachmentChip key={d.id} name={d.name} info={d} />)}
          </div>
        )}
        {quote.length > 0 && <QuoteCard text={quote.join("\n")} source={source} />}
        {rest}
      </div>
    </div>
  );
}

/** What each tool looks at - the court decisions or the statutes - and what it does there. The two
 *  searches work the same way, so the label says how and the tag says where. */
type Domain = "case" | "statute" | "document" | "review";
const TOOLS: Record<string, { domain: Domain; label: string }> = {
  semantic_search: { domain: "case", label: "Search by meaning" },
  keyword_search: { domain: "case", label: "Search exact words" },
  read_decision: { domain: "case", label: "Read decision" },
  citing_decisions: { domain: "case", label: "Who cites it" },
  list_decisions: { domain: "case", label: "List decisions" },
  search_laws: { domain: "statute", label: "Search by meaning" },
  read_law: { domain: "statute", label: "Read article" },
  search_decisions: { domain: "case", label: "Search exact words" },
  count_decisions: { domain: "case", label: "Count" },
  read_document: { domain: "document", label: "Read document" },
  search_document: { domain: "document", label: "Find in document" },
  search_case_file: { domain: "document", label: "Search the case file" },
  // write_answer only shows up as a step when it did not end the research: the agent's notes named
  // something still missing, or the checks rejected most of the draft, and it went back to search
  write_answer: { domain: "review", label: "Sent back to research" },
};
const DOMAIN_NAME: Record<Domain, string> = {
  case: "Case law", statute: "Statutes", document: "Your document", review: "Review",
};

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

/** "3 steps in case law · 1 in statutes" - where the research went, at a glance. */
function researchSummary(calls: ToolCall[]): string {
  const count = (d: Domain) => calls.filter((c) => TOOLS[c.name]?.domain === d).length;
  const cases = count("case");
  const statutes = count("statute");
  const documents = count("document");
  const other = calls.length - cases - statutes - documents - count("review");
  const steps = (n: number) => `${n} step${n === 1 ? "" : "s"}`;
  const parts = [];
  if (cases) parts.push(`${steps(cases)} in case law`);
  if (statutes) parts.push(`${parts.length ? statutes : steps(statutes)} in statutes`);
  if (documents) parts.push(`${parts.length ? documents : steps(documents)} in the document`);
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
    : c.name === "list_decisions" || c.name === "read_document" || c.name === "write_answer"
      ? ""
      : c.name === "search_document"
      ? (a.words ?? "")
      : (a.query ?? a.keyword ?? a.decision_id ?? Object.values(a)[0] ?? "");
  const offset = typeof a.offset === "number" && a.offset > 0 ? ` · from character ${a.offset.toLocaleString("en")}` : "";
  return `${String(main)}${offset}`;
}

const COURT_NAME: Record<string, string> = {
  federal_supreme: "Federal Supreme Court",
  leading_cases: "Leading cases (BGE)",
  federal_administrative: "Federal Administrative Court",
  federal_criminal: "Federal Criminal Court",
  federal_patent: "Federal Patent Court",
  federal_other: "Other federal bodies",
  cantonal: "Cantonal courts",
};

/** The part of the corpus a step was restricted to: "GE", "Federal Supreme Court", "2020–", "OR". */
function toolFilters(c: ToolCall): string[] {
  // the model sometimes writes "None" for a filter it does not use
  const a = Object.fromEntries(
    Object.entries(c.args).filter(([, v]) => v != null && !["none", "null", ""].includes(String(v).toLowerCase())),
  ) as ToolCall["args"];
  const words = (v: unknown) => String(v).replace(/_/g, " ");
  if (c.name === "search_laws") {
    return [a.canton && a.canton !== "CH" ? `${a.canton} law` : "", a.code ?? "", a.cantonal ? "incl. cantonal law" : ""]
      .filter(Boolean)
      .map(String);
  }
  if (c.name === "read_law") return [];
  const years = a.year_from || a.year_to ? `${a.year_from ?? ""}–${a.year_to ?? ""}` : "";
  return [
    a.canton ? String(a.canton) : "",
    a.court ? (COURT_NAME[String(a.court)] ?? words(a.court)) : "",
    a.area ? `${words(a.area)} law` : "",
    a.proceeding ? words(a.proceeding) : "",
    years,
    c.name === "list_decisions" && a.oldest ? "oldest first" : "",
  ].filter(Boolean);
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
              {toolFilters(c).map((f) => (
                <span key={f} className="filter" title="Searched only this part of the corpus">
                  {f}
                </span>
              ))}
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
  clarification?: Clarification;
  /** Only on the last turn: its suggested answers can still be sent. */
  onReply?: (text: string) => void;
}

function AssistantTurn({ id, content, sources, statutes, onOpenStatute, tools, status, thinking, error, language,
  streaming, activeN, onOpenSource, clarification, onReply }: TurnProps) {
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
        // "checking" is the agent verifying its own draft against the passages; it can run for a few
        // rounds, and the detail says which ("Checking 3 citations", "Revising the draft: 2 problems").
        <p className={`status-line${status.stage === "checking" ? " status-checking" : ""}`} aria-live="polite">
          <span className="dot" />
          {status.detail}
        </p>
      )}
      {clarification && <p className="clarify-label">Question for you</p>}
      {content && (
        <Answer id={id} text={content} sources={sources} statutes={statutes} onStatute={onOpenStatute}
          language={language} streaming={streaming} activeN={activeN} onCite={open} />
      )}
      {clarification && onReply && (
        <div className="clarify">
          {clarification.options.map((o) => (
            <button key={o} className="clarify-option" onClick={() => onReply(o)}>
              {o}
            </button>
          ))}
          <span className="clarify-hint">or type your answer below</span>
        </div>
      )}
      {error && <p className="msg-error">{error}</p>}
    </div>
  );
}
