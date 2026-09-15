import { useEffect, useRef } from "react";
import type { Message, Source, Stage, ToolCall } from "../api";
import Answer from "./Answer";

export interface PendingTurn {
  status: { stage: Stage; detail: string } | null;
  tools: ToolCall[];
  sources: Source[];
  content: string;
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
            error={pending.error}
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
  return (
    <div className="msg msg-user">
      <div className="bubble">{text}</div>
    </div>
  );
}

const TOOL_LABEL: Record<string, string> = {
  semantic_search: "Semantic search",
  keyword_search: "Keyword search",
  read_decision: "Read decision",
};

function toolArg(c: ToolCall): string {
  const a = c.args;
  const main = a.query ?? a.keyword ?? a.decision_id ?? Object.values(a)[0] ?? "";
  const offset = typeof a.offset === "number" && a.offset > 0 ? ` · from character ${a.offset.toLocaleString("en")}` : "";
  return `${String(main)}${offset}`;
}

function Activity({ calls, live }: { calls: ToolCall[]; live: boolean }) {
  return (
    <details className="activity" open={live || undefined}>
      <summary>
        Research · {calls.length} tool call{calls.length === 1 ? "" : "s"}
      </summary>
      <ol>
        {calls.map((c) => {
          const state = c.error ? "error" : c.summary == null ? "running" : "done";
          return (
            <li key={c.id} className={state}>
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
  error?: string;
  streaming: boolean;
  activeN: number | null;
  onOpenSource: (messageId: string, n: number) => void;
}

function AssistantTurn({ id, content, sources, tools, status, error, streaming, activeN, onOpenSource }: TurnProps) {
  const open = (n: number) => onOpenSource(id, n);
  return (
    <div className="msg msg-assistant">
      {tools.length > 0 && <Activity calls={tools} live={streaming} />}
      {streaming && !content && !error && status && (
        <p className="status-line" aria-live="polite">
          <span className="dot" />
          {status.detail}
        </p>
      )}
      {content && <Answer text={content} sources={sources} streaming={streaming} activeN={activeN} onCite={open} />}
      {error && <p className="msg-error">{error}</p>}
    </div>
  );
}
