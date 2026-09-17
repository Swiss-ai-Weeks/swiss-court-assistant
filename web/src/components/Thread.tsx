import { useEffect, useRef } from "react";
import type { Message, Source, Stage, ToolCall } from "../api";
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
}

const PENDING_ID = "pending";

export default function Thread({ messages, pending, selection, onOpenSource }: Props) {
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

const TOOL_LABEL: Record<string, string> = {
  semantic_search: "Semantic search",
  keyword_search: "Keyword search",
  read_decision: "Read decision",
  citing_decisions: "Who cites it",
};

function toolArg(c: ToolCall): string {
  const a = c.args;
  const perLanguage = ["de", "fr", "it"].filter((l) => a[`query_${l}`]).map((l) => `${l.toUpperCase()} ${a[`query_${l}`]}`);
  const main = perLanguage.length
    ? perLanguage.join(" · ")
    : (a.query ?? a.keyword ?? a.decision_id ?? Object.values(a)[0] ?? "");
  const offset = typeof a.offset === "number" && a.offset > 0 ? ` · from character ${a.offset.toLocaleString("en")}` : "";
  return `${String(main)}${offset}`;
}

function Activity({ calls, live }: { calls: ToolCall[]; live: boolean }) {
  return (
    <details className="activity" open={live || undefined}>
      <summary>
        Research - {calls.length} searches{calls.length === 1 ? "" : "s"}
      </summary>
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
  tools: ToolCall[];
  status?: PendingTurn["status"];
  thinking?: string;
  error?: string;
  language: string | null;
  streaming: boolean;
  activeN: number | null;
  onOpenSource: (messageId: string, n: number) => void;
}

function AssistantTurn({ id, content, sources, tools, status, thinking, error, language, streaming, activeN,
  onOpenSource }: TurnProps) {
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
        <Answer id={id} text={content} sources={sources} language={language} streaming={streaming} activeN={activeN}
          onCite={open} />
      )}
      {error && <p className="msg-error">{error}</p>}
    </div>
  );
}
