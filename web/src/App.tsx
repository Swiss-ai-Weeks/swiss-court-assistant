import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ConversationSummary, type Health, type Message } from "./api";
import Composer from "./components/Composer";
import Preview from "./components/Preview";
import Sidebar from "./components/Sidebar";
import Thread, { type PendingTurn, type Selection } from "./components/Thread";
import Welcome from "./components/Welcome";

export default function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const [health, setHealth] = useState<Health | "offline" | null>(null);

  const refreshList = useCallback(
    () => api.listConversations().then(setConversations, (e) => console.error("history:", e)),
    [],
  );
  useEffect(() => {
    refreshList();
    api.health().then(setHealth, () => setHealth("offline"));
  }, [refreshList]);

  const openConversation = async (id: string | null) => {
    abort.current?.abort();
    setPending(null);
    setSelection(null);
    setSidebarOpen(false);
    setActiveId(id);
    setMessages(id ? (await api.getConversation(id)).messages : []);
  };

  const deleteConversation = async (id: string) => {
    await api.deleteConversation(id);
    if (id === activeId) await openConversation(null);
    refreshList();
  };

  const send = async (text: string) => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setSelection(null);
    setMessages((m) => [...m, { id: `local-${Date.now()}`, role: "user", content: text, createdAt: new Date().toISOString() }]);
    setPending({ status: null, tools: [], sources: [], content: "" });
    try {
      for await (const ev of api.chat(activeId, text, ctrl.signal)) {
        switch (ev.type) {
          case "conversation":
            setActiveId(ev.conversation.id);
            refreshList();
            break;
          case "status":
            setPending((p) => p && { ...p, status: { stage: ev.stage, detail: ev.detail } });
            break;
          case "tool_start":
            setPending((p) => p && { ...p, tools: [...p.tools, ev.call] });
            break;
          case "tool_end":
            setPending(
              (p) =>
                p && {
                  ...p,
                  tools: p.tools.map((c) => (c.id === ev.id ? { ...c, summary: ev.summary, error: ev.error } : c)),
                },
            );
            break;
          case "delta":
            setPending((p) => p && { ...p, content: p.content + ev.text });
            break;
          case "citation": {
            const src = ev.source;
            setPending(
              (p) =>
                p && {
                  ...p,
                  content: `${p.content}[${src.n}]`,
                  sources: p.sources.some((s) => s.n === src.n) ? p.sources : [...p.sources, src],
                },
            );
            break;
          }
          case "done":
            setMessages((m) => [...m, ev.message]);
            setPending(null);
            setSelection((s) => (s?.messageId === "pending" ? { ...s, messageId: ev.message.id } : s));
            refreshList();
            break;
          case "error":
            setPending((p) => p && { ...p, error: ev.message });
            break;
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setPending((p) => p && { ...p, error: (e as Error).message });
      else setPending(null);
    } finally {
      if (abort.current === ctrl) abort.current = null;
    }
  };

  const selectedSources =
    selection?.messageId === "pending"
      ? pending?.sources ?? []
      : messages.find((m) => m.id === selection?.messageId)?.sources ?? [];
  const showPreview = !!selection && selectedSources.some((s) => s.n === selection.n);
  const busy = pending !== null && !pending.error;

  return (
    <div className="app">
      <nav className="nav">
        <button className="nav-menu" onClick={() => setSidebarOpen((o) => !o)} aria-label="Toggle history">
          ☰
        </button>
        <div className="wordmark">
          <span className="wordmark-square" />
          Swiss Court Assistant
        </div>
        <div className="nav-meta">
          {health === "offline" ? (
            <b>API offline</b>
          ) : (
            health && (
              <>
                {health.decisions.toLocaleString("en")} decisions · <b>{health.agent} agent</b>
              </>
            )
          )}
        </div>
      </nav>

      <div className={`body${showPreview ? " with-preview" : ""}`}>
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          open={sidebarOpen}
          onNew={() => openConversation(null)}
          onSelect={openConversation}
          onDelete={deleteConversation}
        />
        <div className={`backdrop${sidebarOpen ? " show" : ""}`} onClick={() => setSidebarOpen(false)} />

        <main className="chat">
          {messages.length === 0 && !pending ? (
            <div className="thread">
              <Welcome onAsk={send} />
            </div>
          ) : (
            <Thread
              messages={messages}
              pending={pending}
              selection={selection}
              onOpenSource={(messageId, n) => setSelection({ messageId, n })}
            />
          )}
          <Composer busy={busy} demo={health !== null && health !== "offline" && health.agent === "stub"} onSend={send} onStop={() => abort.current?.abort()} />
        </main>

        {showPreview && (
          <Preview
            sources={selectedSources}
            activeN={selection!.n}
            onSelect={(n) => setSelection((s) => s && { ...s, n })}
            onClose={() => setSelection(null)}
          />
        )}
      </div>
    </div>
  );
}
