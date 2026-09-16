import { useSpeech } from "../speech";

interface Props {
  /** Identifies this text; only one text is read at a time. */
  id: string;
  text: string;
  language: string;
  className?: string;
}

const LABEL = { idle: "Read aloud", loading: "Preparing…", playing: "Stop", error: "Audio unavailable" };

export default function ReadAloud({ id, text, language, className = "" }: Props) {
  const { state, toggle } = useSpeech(id);
  const on = state === "loading" || state === "playing";
  return (
    <button
      className={`speak-btn ${className}${on ? " on" : ""}${state === "error" ? " err" : ""}`}
      onClick={() => toggle(text, language)}
      aria-pressed={on}
      title={state === "error" ? "The speech model is not available right now" : undefined}
    >
      <span className={`speak-icon ${state}`} aria-hidden="true" />
      {LABEL[state]}
    </button>
  );
}
