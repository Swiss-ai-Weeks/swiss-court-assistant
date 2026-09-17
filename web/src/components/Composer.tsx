import { useEffect, useRef, useState } from "react";
import { langName } from "../format";
import type { Quote } from "./SelectionTools";

interface Props {
  busy: boolean;
  demo: boolean;
  /** Selected text the next message replies to: shown above the input, not editable. */
  quote: Quote | null;
  onClearQuote: () => void;
  /** Voice mode: the microphone is open and the assistant answers out loud. */
  voiceOn: boolean;
  onToggleVoice: () => void;
  onSend: (text: string) => void;
  onStop: () => void;
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
  const ref = useRef<HTMLTextAreaElement>(null);

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
    if (!t || busy) return;
    onSend(withQuote(t, quote));
    setText("");
    onClearQuote();
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        {quote && <QuoteCard text={quote.text} source={quote.source} language={quote.language} onRemove={onClearQuote} />}
        <div className="composer-row">
          <textarea
            ref={ref}
            rows={1}
            value={text}
            placeholder={quote ? "Ask about the selected text…" : "Ask about Swiss court decisions…"}
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
            <button className="btn-primary" onClick={submit} disabled={!text.trim()}>
              Ask
            </button>
          )}
        </div>
        <p className="composer-note">
          {demo && <b>Stub agent, canned answers. </b>}
          Answers are AI-generated from retrieved passages. Check the cited decisions before relying on them.
        </p>
      </div>
    </div>
  );
}
