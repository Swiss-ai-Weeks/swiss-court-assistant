import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ChatEvent, type ConversationSummary, type Health, type MatterSummary, type Message,
  type Source } from "./api";
import { startVoice, type VoiceEvent, type VoiceSession } from "./voice";
import Composer from "./components/Composer";
import Logo from "./components/Logo";
import MatterPage from "./components/MatterPage";
import Preview from "./components/Preview";
import SelectionTools, { type Quote } from "./components/SelectionTools";
import Sidebar from "./components/Sidebar";
import Thread, { type PendingTurn, type Selection } from "./components/Thread";
import Welcome from "./components/Welcome";

type Page = "chat" | "matters";

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [matters, setMatters] = useState<MatterSummary[]>([]);
  const [matterId, setMatterId] = useState<string | null>(null);
  // a cited passage opened from a matter; the chat page opens its own from `selection`
  const [matterPreview, setMatterPreview] = useState<{ sources: Source[]; n: number } | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // selected text the next message replies to (see SelectionTools)
  const [quote, setQuote] = useState<Quote | null>(null);
  // voice mode: an open microphone session, what it hears, and whether the assistant is talking
  const [voice, setVoice] = useState<VoiceSession | null>(null);
  const [heard, setHeard] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const [health, setHealth] = useState<Health | "offline" | null>(null);

  const refreshList = useCallback(
    () => api.listConversations().then(setConversations, (e) => console.error("history:", e)),
    [],
  );
  const refreshMatters = useCallback(
    () => api.listMatters().then(setMatters, (e) => console.error("matters:", e)),
    [],
  );
  useEffect(() => {
    refreshList();
    refreshMatters();
    api.health().then(setHealth, () => setHealth("offline"));
  }, [refreshList, refreshMatters]);

  const openMatter = (id: string | null) => {
    setMatterId(id);
    setMatterPreview(null);
    setSidebarOpen(false);
  };

  const deleteMatter = async (id: string) => {
    await api.deleteMatter(id);
    if (id === matterId) openMatter(null);
    refreshMatters();
  };

  const openConversation = async (id: string | null) => {
    abort.current?.abort();
    setPending(null);
    setSelection(null);
    setQuote(null);
    setSidebarOpen(false);
    setActiveId(id);
    setMessages(id ? (await api.getConversation(id)).messages : []);
  };

  const deleteConversation = async (id: string) => {
    await api.deleteConversation(id);
    if (id === activeId) await openConversation(null);
    refreshList();
  };

  const pendingRef = useRef<PendingTurn | null>(null);
  pendingRef.current = pending;

  /** One turn's events, from the chat stream or from voice mode: both send the same shapes. */
  const applyEvent = (ev: ChatEvent) => {
    switch (ev.type) {
      case "conversation":
        setActiveId(ev.conversation.id);
        refreshList();
        break;
      case "meta":
        setPending((p) => p && { ...p, language: ev.language });
        break;
      case "status":
        setPending((p) => p && { ...p, status: { stage: ev.stage, detail: ev.detail } });
        break;
      case "thinking":
        setPending((p) => p && { ...p, thinking: (p.thinking ?? "") + ev.text });
        break;
      case "tool_start": // the call carries its thought; the next step thinks afresh
        setPending((p) => p && { ...p, tools: [...p.tools, ev.call], thinking: "" });
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
      case "verdict":
        setPending(
          (p) =>
            p && { ...p, sources: p.sources.map((s) => (s.n === ev.n ? { ...s, supported: ev.supported } : s)) },
        );
        break;
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
  };

  const send = async (text: string) => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setSelection(null);
    setMessages((m) => [...m, { id: `local-${Date.now()}`, role: "user", content: text, createdAt: new Date().toISOString() }]);
    setPending({ status: null, tools: [], sources: [], content: "" });
    try {
      for await (const ev of api.chat(activeId, text, ctrl.signal)) applyEvent(ev);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setPending((p) => p && { ...p, error: (e as Error).message });
      else setPending(null);
    } finally {
      if (abort.current === ctrl) abort.current = null;
    }
  };

  // ── voice mode ────────────────────────────────────────────────────────
  const onVoiceEvent = (ev: VoiceEvent) => {
    switch (ev.type) {
      case "partial":
        setHeard(ev.final ? "" : ev.text);
        break;
      case "user": {
        const p = pendingRef.current; // an answer the user talked over keeps what was said so far
        if (p?.content)
          setMessages((m) => [
            ...m,
            { id: `local-a-${Date.now()}`, role: "assistant", content: p.content, sources: p.sources,
              toolCalls: p.tools, language: p.language ?? null, createdAt: new Date().toISOString() },
          ]);
        setMessages((m) => [...m, ev.message]);
        setPending({ status: null, tools: [], sources: [], content: "" });
        break;
      }
      case "speaking": // from playback, not from the socket: the audio arrives long before it is heard
        setSpeaking(ev.on);
        break;
      case "speech_start":
      case "speech_end":
      case "cancel_speech":
        break;
      case "voice_ended":
        setVoice(null);
        setSpeaking(false);
        setHeard("");
        if (ev.reason) setVoiceError(ev.reason);
        break;
      default:
        applyEvent(ev);
    }
  };

  /** The language spoken so far in this conversation, else the browser's (of the four in the corpus). */
  const voiceLanguage = () => {
    const spoken = [...messages].reverse().find((m) => m.language)?.language;
    const browser = navigator.language.slice(0, 2);
    return spoken ?? (["de", "fr", "it", "en"].includes(browser) ? browser : "en");
  };

  const toggleVoice = async () => {
    if (voice) {
      voice.stop();
      setVoice(null);
      setHeard("");
      setSpeaking(false);
      return;
    }
    setVoiceError(null);
    try {
      setVoice(await startVoice({ conversationId: activeId, language: voiceLanguage(), onEvent: onVoiceEvent }));
    } catch (e) {
      setVoiceError((e as Error).message || "The microphone is not available.");
    }
  };

  const selectedSources =
    selection?.messageId === "pending"
      ? pending?.sources ?? []
      : messages.find((m) => m.id === selection?.messageId)?.sources ?? [];
  const showPreview = !!selection && selectedSources.some((s) => s.n === selection.n);
  const selectedLanguage =
    selection?.messageId === "pending"
      ? pending?.language ?? null
      : messages.find((m) => m.id === selection?.messageId)?.language ?? null;
  const busy = pending !== null && !pending.error;
  // the recogniser may understand one language or many; say so only when it is limited to English
  const spoken = health !== null && health !== "offline" ? health.speechLanguages ?? [] : [];
  const voiceHint = spoken.length === 1 && spoken[0].startsWith("en")
    ? "Speak in English — talk over me to interrupt."
    : "Talk over me to interrupt.";


  return (
    <div className="app">
      <nav className="nav">
        <button className="nav-menu" onClick={() => setSidebarOpen((o) => !o)} aria-label="Toggle history">
          ☰
        </button>
        <div className="wordmark">
          <Logo className="wordmark-logo" />
          Swiss Court Assistant
        </div>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={page === "chat"} className={page === "chat" ? "active" : ""}
            onClick={() => setPage("chat")}>
            Assistant
          </button>
          <button role="tab" aria-selected={page === "matters"} className={page === "matters" ? "active" : ""}
            onClick={() => setPage("matters")}>
            Matters
          </button>
        </div>
      </nav>

      <div className={`body${(page === "chat" ? showPreview : !!matterPreview) ? " with-preview" : ""}`}>
        {page === "chat" ? (
          <Sidebar
            items={conversations}
            label="History"
            newLabel="New conversation"
            emptyLabel="No conversations yet."
            activeId={activeId}
            open={sidebarOpen}
            onNew={() => openConversation(null)}
            onSelect={openConversation}
            onDelete={deleteConversation}
          />
        ) : (
          <Sidebar
            items={matters}
            label="Matters"
            newLabel="New matter"
            emptyLabel="No matters yet."
            activeId={matterId}
            open={sidebarOpen}
            onNew={() => openMatter(null)}
            onSelect={openMatter}
            onDelete={deleteMatter}
          />
        )}
        <div className={`backdrop${sidebarOpen ? " show" : ""}`} onClick={() => setSidebarOpen(false)} />

        {page === "chat" ? (
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
            {(voice || voiceError) && (
              <div className={`voice-bar${speaking ? " speaking" : ""}`}>
                <span className="voice-dot" />
                <span className="voice-state">{voiceError ?? (speaking ? "Speaking" : "Listening")}</span>
                {!voiceError && <span className="voice-heard">{heard || voiceHint}</span>}
                <button className="voice-stop" onClick={voice ? toggleVoice : () => setVoiceError(null)}>
                  {voice ? "Stop voice" : "Dismiss"}
                </button>
              </div>
            )}
            <Composer busy={busy} demo={health !== null && health !== "offline" && health.agent === "stub"} quote={quote}
              voiceOn={!!voice} onToggleVoice={toggleVoice}
              onClearQuote={() => setQuote(null)} onSend={send} onStop={() => abort.current?.abort()} />
          </main>
        ) : (
          <MatterPage
            matterId={matterId}
            onOpenMatter={openMatter}
            onChanged={refreshMatters}
            onOpenSource={(sources, n) => setMatterPreview({ sources, n })}
          />
        )}
        <SelectionTools onReply={setQuote} />

        {page === "chat" && showPreview && (
          <Preview
            sources={selectedSources}
            activeN={selection!.n}
            language={selectedLanguage}
            onSelect={(n) => setSelection((s) => s && { ...s, n })}
            onClose={() => setSelection(null)}
          />
        )}
        {page === "matters" && matterPreview && (
          <Preview
            sources={matterPreview.sources}
            activeN={matterPreview.n}
            language={null}
            onSelect={(n) => setMatterPreview((p) => p && { ...p, n })}
            onClose={() => setMatterPreview(null)}
          />
        )}
      </div>
    </div>
  );
}
