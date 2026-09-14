import { useEffect, useRef, useState } from "react";

interface Props {
  busy: boolean;
  demo: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export default function Composer({ busy, demo, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight + 2, 200)}px`;
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        <div className="composer-row">
          <textarea
            ref={ref}
            rows={1}
            value={text}
            placeholder="Ask about Swiss court decisions…"
            aria-label="Your question"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
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
