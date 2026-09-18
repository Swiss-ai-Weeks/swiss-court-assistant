import { useRef, useState } from "react";
import type { MatterInput } from "../api";
import { isAudio, record, toPcm16k, type Recorder } from "../audio";

interface Props {
  /** What the server is doing with the upload, or null when it is idle. */
  busy: string | null;
  error: string | null;
  onStart: (input: MatterInput) => void;
}

const ACCEPT = ".pdf,.docx,.txt,.md,audio/*";

/** The start of a matter: whatever the client handed over — a document, a recording of them telling
 *  the story, or notes typed from the first phone call. */
export default function MatterIntake({ busy, error, onStart }: Props) {
  const [text, setText] = useState("");
  const [over, setOver] = useState(false);
  const [recorder, setRecorder] = useState<Recorder | null>(null);
  const [seconds, setSeconds] = useState(0);
  // decoding happens in this tab and can take a moment for a long recording, so it gets its own line
  const [converting, setConverting] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const tick = useRef<number | null>(null);

  const take = async (file: File) => {
    setFailed(null);
    if (!isAudio(file)) return onStart({ file, filename: file.name });
    setConverting("Converting the recording…");
    try {
      // the browser decodes it; the server only ever sees 16 kHz PCM
      const pcm = await toPcm16k(await file.arrayBuffer());
      onStart({ file: pcm, filename: `${file.name}.pcm`, title: file.name });
    } catch {
      setFailed(`${file.name} could not be decoded as audio.`);
    } finally {
      setConverting(null);
    }
  };

  const toggleRecording = async () => {
    if (recorder) {
      if (tick.current) window.clearInterval(tick.current);
      setConverting("Converting the recording…");
      try {
        const pcm = await recorder.stop();
        setRecorder(null);
        const minutes = Math.floor(seconds / 60);
        onStart({ file: pcm, filename: "client-interview.pcm", title: `Client interview (${minutes || "<1"} min)` });
      } catch {
        setRecorder(null);
        setFailed("The recording could not be read.");
      } finally {
        setConverting(null);
      }
      return;
    }
    setFailed(null);
    try {
      setRecorder(await record());
      setSeconds(0);
      tick.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch {
      setFailed("The microphone is not available.");
    }
  };

  const clock = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  const working = converting ?? busy;

  return (
    <div className="intake">
      <p className="eyebrow">Case Prep · intake to memo</p>
      <h1>Start from what the client gave you.</h1>
      <p className="intake-lead">
        A document, a recording of the first conversation, or your own notes. The assistant reads it, names
        the legal questions in it, researches each one in Swiss case law, and drafts a memo where every
        proposition carries a citation.
      </p>

      <div
        className={`dropzone${over ? " over" : ""}${working ? " busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const file = e.dataTransfer.files[0];
          if (file && !working) take(file);
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) take(file);
          }}
        />
        {working ? (
          <div className="working" aria-live="polite">
            <span className="dot" />
            <p className="working-main">{working}</p>
            <p className="working-hint">
              A recording takes roughly as long to transcribe as it lasted. Nothing leaves this machine.
            </p>
          </div>
        ) : (
          <>
            <p className="dropzone-main">Drop a file here</p>
            <p className="dropzone-hint">PDF, Word, text — or an audio recording (mp3, m4a, wav, ogg)</p>
            <div className="dropzone-actions">
              <button className="btn-outline" onClick={() => input.current?.click()}>
                Choose a file
              </button>
              <button className={`btn-outline${recorder ? " recording" : ""}`} onClick={toggleRecording}>
                {recorder ? `Stop recording · ${clock}` : "Record the client"}
              </button>
            </div>
          </>
        )}
      </div>

      <div className="intake-text">
        <label htmlFor="matter-facts">Or type the facts</label>
        <textarea
          id="matter-facts"
          value={text}
          rows={6}
          placeholder="Our client, a tenant in Zurich, received a termination on 14 March 2024…"
          onChange={(e) => setText(e.target.value)}
        />
        <button className="btn-primary" disabled={!!working || text.trim().length < 20}
          onClick={() => onStart({ text })}>
          {working ? "Reading…" : "Open the matter"}
        </button>
      </div>

      {(failed || error) && <p className="msg-error">{failed ?? error}</p>}
    </div>
  );
}
