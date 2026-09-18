import { useEffect, useRef, useState } from "react";
import { api, type DocumentInfo } from "../api";
import { langName } from "../format";
import type { Quote } from "./SelectionTools";

/** What can be attached: PDFs and scans go through Nemotron Parse, Word and text files are read directly. */
export const DOCUMENT_ACCEPT = ".pdf,.docx,.dotx,.txt,.md,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp";
const MAX_ATTACHMENTS = 5;

/** A file on its way to the server: parsing takes a few seconds a page, so it shows while it runs. */
interface Attachment {
  key: string;
  name: string;
  info?: DocumentInfo;
  error?: string;
  abort: AbortController;
}

interface Props {
  busy: boolean;
  demo: boolean;
  /** Selected text the next message replies to: shown above the input, not editable. */
  quote: Quote | null;
  onClearQuote: () => void;
  /** Voice mode: the microphone is open and the assistant answers out loud. */
  voiceOn: boolean;
  onToggleVoice: () => void;
  onSend: (text: string, attachments: DocumentInfo[]) => void;
  onStop: () => void;
}

const pages = (d: DocumentInfo) => `${d.pages} page${d.pages === 1 ? "" : "s"}`;

/** A document attached to a message: its name, how long it is, and a link to the file as uploaded. */
export function AttachmentChip({ name, info, error, onRemove }: {
  name: string;
  info?: DocumentInfo;
  error?: string;
  onRemove?: () => void;
}) {
  const state = error ? " failed" : info ? "" : " parsing";
  return (
    <span className={`attachment${state}`} title={error ?? (info ? `${info.name} · read with ${info.parser}` : "Reading…")}>
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        <path d="M6 2h8l5 5v15H6z M14 2v5h5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      </svg>
      {info ? (
        <a href={api.documentFileUrl(info.id)} target="_blank" rel="noreferrer">{name}</a>
      ) : (
        <span className="attachment-name">{name}</span>
      )}
      <span className="attachment-meta">{error ? "could not be read" : info ? pages(info) : "reading…"}</span>
      {onRemove && (
        <button onClick={onRemove} aria-label={`Remove ${name}`} title="Remove">
          ×
        </button>
      )}
    </span>
  );
}

/** The message as sent: the quote as "> " lines (the thread and the backend recognise them), then the text. */
function withQuote(text: string, q: Quote | null): string {
  return q ? `> ${q.text}\n${q.source ? `> — ${q.source}\n` : ""}\n${text}` : text;
}

export function QuoteCard({ text, source, language, onRemove }: {
  text: string;
  source: string | null;
  language?: string;
  onRemove?: () => void;
}) {
  return (
    <div className="quote-card">
      <div className="quote-card-head">
        <span className="quote-card-label">
          Replying to {source ?? "the selection"}
          {language && ` · ${langName(language)}`}
        </span>
        {onRemove && (
          <button className="quote-card-close" onClick={onRemove} aria-label="Remove the quoted text" title="Remove">
            ×
          </button>
        )}
      </div>
      <p className="quote-card-text" title={text}>
        {text}
      </p>
    </div>
  );
}

export default function Composer({ busy, demo, quote, onClearQuote, voiceOn, onToggleVoice, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [over, setOver] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const reading = attachments.some((a) => !a.info && !a.error);
  const ready = attachments.filter((a) => a.info).map((a) => a.info!);
  const failed = attachments.find((a) => a.error)?.error;

  // Parsing starts as soon as a file is picked, so it is usually done by the time the question is typed.
  const attach = (files: FileList | File[]) => {
    for (const file of Array.from(files).slice(0, MAX_ATTACHMENTS - attachments.length)) {
      const a: Attachment = { key: `${file.name}-${Date.now()}-${Math.random()}`, name: file.name, abort: new AbortController() };
      setAttachments((all) => [...all, a]);
      api.uploadDocument(file, a.abort.signal).then(
        (info) => setAttachments((all) => all.map((x) => (x.key === a.key ? { ...x, info } : x))),
        (e: Error) => {
          if (e.name !== "AbortError")
            setAttachments((all) => all.map((x) => (x.key === a.key ? { ...x, error: e.message } : x)));
        },
      );
    }
  };
  const remove = (key: string) =>
    setAttachments((all) => {
      all.find((a) => a.key === key)?.abort.abort();
      return all.filter((a) => a.key !== key);
    });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight + 2, 200)}px`;
  }, [text]);

  useEffect(() => {
    if (quote) ref.current?.focus();
  }, [quote]);

  const submit = () => {
    const t = text.trim();
    if (!t || busy || reading) return;
    onSend(withQuote(t, quote), ready);
    setText("");
    setAttachments([]);
    onClearQuote();
  };

  return (
    <div
      className={`composer${over ? " over" : ""}`}
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        setOver(false);
        attach(e.dataTransfer.files);
      }}
    >
      <div className="composer-inner">
        {quote && <QuoteCard text={quote.text} source={quote.source} language={quote.language} onRemove={onClearQuote} />}
        {attachments.length > 0 && (
          <div className="attachments" aria-live="polite">
            {attachments.map((a) => (
              <AttachmentChip key={a.key} name={a.name} info={a.info} error={a.error} onRemove={() => remove(a.key)} />
            ))}
          </div>
        )}
        <div className="composer-row">
          <input
            ref={picker}
            type="file"
            accept={DOCUMENT_ACCEPT}
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) attach(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            className="mic-btn"
            onClick={() => picker.current?.click()}
            disabled={attachments.length >= MAX_ATTACHMENTS}
            title="Attach a document (PDF, Word, scan or photo)"
            aria-label="Attach a document"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M20 11.5l-8.2 8.2a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4L14.8 7"
                fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
          <textarea
            ref={ref}
            rows={1}
            value={text}
            placeholder={quote ? "Ask about the selected text…" : attachments.length ? "Ask about the attached document…"
              : "Ask about Swiss court decisions…"}
            aria-label="Your question"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              } else if (e.key === "Escape" && quote) {
                onClearQuote();
              }
            }}
          />
          <button
            className={`mic-btn${voiceOn ? " on" : ""}`}
            onClick={onToggleVoice}
            aria-pressed={voiceOn}
            title={voiceOn ? "Leave voice mode" : "Talk to the assistant"}
            aria-label={voiceOn ? "Leave voice mode" : "Talk to the assistant"}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z" fill="currentColor" />
              <path d="M5 11a7 7 0 0 0 14 0M12 18v3" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
          </button>
          {busy ? (
            <button className="btn-primary" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button className="btn-primary" onClick={submit} disabled={!text.trim() || reading}
              title={reading ? "Wait until the document is read" : undefined}>
              Ask
            </button>
          )}
        </div>
        {failed && <p className="msg-error composer-error">{failed}</p>}
        <p className="composer-note">
          {demo && <b>Stub agent, canned answers. </b>}
          {reading && <b>Reading the document with Nemotron Parse… </b>}
          Answers are AI-generated from retrieved passages. Check the cited decisions before relying on them.
        </p>
      </div>
    </div>
  );
}
