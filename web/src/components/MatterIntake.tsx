import { useRef, useState } from "react";
import { api, type DocumentInfo, type MatterInput } from "../api";
import { isAudio, record, toPcm16k, type Recorder } from "../audio";
import { DOCUMENT_ACCEPT } from "./Composer";

interface Props {
  /** What the server is doing with the upload, or null when it is idle. */
  busy: string | null;
  error: string | null;
  onStart: (input: MatterInput) => void;
}

const ACCEPT = `${DOCUMENT_ACCEPT},audio/*`;
const MAX_ITEMS = 12;

/** One piece of the case file on its way in: converted in this tab (audio), then read on the server. */
interface Item {
  key: string;
  name: string;
  audio: boolean;
  state: "converting" | "reading" | "ready" | "failed";
  info?: DocumentInfo;
  error?: string;
  abort: AbortController;
}

const minutes = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

function describe(item: Item): string {
  if (item.state === "converting") return "converting…";
  if (item.state === "reading") return item.audio ? "transcribing…" : "reading…";
  if (item.state === "failed") return item.error ?? "could not be read";
  const d = item.info!;
  return d.kind === "recording" ? `${minutes(d.seconds ?? 0)} min · transcribed` : `${d.pages} page${d.pages === 1 ? "" : "s"}`;
}

/** The start of a matter: whatever the client handed over — documents, recordings of them telling the
 *  story, notes typed from the first phone call — as many as there are. Each is read as soon as it is
 *  added, so the case file is ready by the time the last one is in. */
export default function MatterIntake({ busy, error, onStart }: Props) {
  const [text, setText] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [over, setOver] = useState(false);
  const [recorder, setRecorder] = useState<Recorder | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const tick = useRef<number | null>(null);

  const update = (key: string, patch: Partial<Item>) =>
    setItems((all) => all.map((x) => (x.key === key ? { ...x, ...patch } : x)));

  /** Upload one piece: a document as it is, a recording as 16 kHz PCM (the browser decodes it). */
  const add = async (name: string, audio: boolean, data: () => Promise<{ blob: Blob; filename: string }>) => {
    const item: Item = { key: `${name}-${Date.now()}-${Math.random()}`, name, audio, state: audio ? "converting" : "reading",
      abort: new AbortController() };
    setItems((all) => [...all, item]);
    let body: { blob: Blob; filename: string };
    try {
      body = await data();
    } catch {
      return update(item.key, { state: "failed", error: "could not be decoded as audio" });
    }
    update(item.key, { state: "reading" });
    try {
      const info = await api.uploadDocument(body.blob, item.abort.signal, body.filename);
      update(item.key, { state: "ready", info });
    } catch (e) {
      if ((e as Error).name !== "AbortError") update(item.key, { state: "failed", error: (e as Error).message });
    }
  };

  const take = (files: FileList | File[]) => {
    setFailed(null);
    const room = MAX_ITEMS - items.length;
    if (files.length > room) setFailed(`Up to ${MAX_ITEMS} files per matter.`);
    for (const file of Array.from(files).slice(0, room)) {
      if (isAudio(file))
        add(file.name, true, async () => ({ blob: await toPcm16k(await file.arrayBuffer()), filename: `${file.name}.pcm` }));
      else add(file.name, false, async () => ({ blob: file, filename: file.name }));
    }
  };

  const remove = (key: string) =>
    setItems((all) => {
      all.find((x) => x.key === key)?.abort.abort();
      return all.filter((x) => x.key !== key);
    });

  const toggleRecording = async () => {
    if (recorder) {
      if (tick.current) window.clearInterval(tick.current);
      const r = recorder;
      setRecorder(null);
      const name = `Client interview ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      add(name, true, async () => ({ blob: await r.stop(), filename: `${name}.pcm` }));
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

  const pending = items.some((x) => x.state === "converting" || x.state === "reading");
  const ready = items.filter((x) => x.state === "ready");
  const notes = text.trim();
  const canOpen = !busy && !pending && !recorder && (ready.length > 0 || notes.length >= 20);

  return (
    <div className="intake">
      <p className="eyebrow">Case Prep · intake to memo</p>
      <h1>Start from what the client gave you.</h1>
      <p className="intake-lead">
        Documents, recordings of the first conversation, your own notes
      </p>

      <div
        className={`dropzone${over ? " over" : ""}${busy ? " busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!busy && e.dataTransfer.files.length) take(e.dataTransfer.files);
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) take(e.target.files);
            e.target.value = "";
          }}
        />
        <p className="dropzone-main">Drop files here</p>
        <p className="dropzone-hint">
          PDF, Word, text, scans or photos of letters, audio recordings (mp3, m4a, wav, ogg)
        </p>
        <div className="dropzone-actions">
          <button className="btn-outline" onClick={() => input.current?.click()} disabled={!!busy}>
            Choose files
          </button>
          <button className={`btn-outline${recorder ? " recording" : ""}`} onClick={toggleRecording} disabled={!!busy}>
            {recorder ? `Stop recording · ${minutes(seconds)}` : "Record the client"}
          </button>
        </div>
      </div>

      {items.length > 0 && (
        <ul className="case-items" aria-live="polite">
          {items.map((x) => (
            <li key={x.key} className={`case-item ${x.state}`}>
              <span className="case-kind">{x.audio ? "Recording" : "Document"}</span>
              <span className="case-name" title={x.name}>{x.name}</span>
              <span className="case-meta">{describe(x)}</span>
              <button className="case-remove" onClick={() => remove(x.key)} aria-label={`Remove ${x.name}`} title="Remove">
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      {pending && (
        <p className="working-hint">
          PDFs and scans are read page by page with Nemotron Parse; a recording takes roughly as long to transcribe
          as it lasted. Nothing leaves this machine.
        </p>
      )}

      <div className="intake-text">
        <label htmlFor="matter-facts">{items.length ? "Add your own notes (optional)" : "Or type the facts"}</label>
        <textarea
          id="matter-facts"
          value={text}
          rows={items.length ? 4 : 6}
          placeholder="Our client, a tenant in Zurich, received a termination on 14 March 2024…"
          onChange={(e) => setText(e.target.value)}
        />
        <button
          className="btn-primary"
          disabled={!canOpen}
          onClick={() => onStart({ documentIds: ready.map((x) => x.info!.id), text: notes || undefined })}
        >
          {busy ? "Opening…" : pending ? "Reading the files…" : `Open the matter${
            ready.length + (notes ? 1 : 0) > 1 ? ` · ${ready.length + (notes ? 1 : 0)} items` : ""}`}
        </button>
      </div>

      {(failed || error) && <p className="msg-error">{failed ?? error}</p>}
    </div>
  );
}
