import { useEffect, useRef, useState } from "react";
import { api, type DocumentInfo, type Matter } from "../api";
import { isAudio, toPcm16k } from "../audio";
import { DOCUMENT_ACCEPT } from "./Composer";

interface Props {
  matterId: string;
  /** While the matter is being researched the case file must not change under the run. */
  disabled?: boolean;
  /** The matter as it is once the new pieces belong to it. */
  onAdded: (matter: Matter) => void;
}

const ACCEPT = `${DOCUMENT_ACCEPT},audio/*`;

/** A file on its way into a case file that already exists. */
interface Item {
  key: string;
  name: string;
  audio: boolean;
  state: "converting" | "reading" | "ready" | "failed";
  seconds?: number;
  info?: DocumentInfo;
  error?: string;
  abort: AbortController;
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

/** More documents for a case that is already open - a letter that arrived later, the second interview.
 *  Each is read on the server as at intake; once all of them are read they join the case file together,
 *  which is also when they are indexed, so the assistant can be asked about them straight away. The
 *  research itself is not redone: the matter says how much of the case file its last run has not seen. */
export default function AddToCase({ matterId, disabled, onAdded }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const update = (key: string, patch: Partial<Item>) =>
    setItems((all) => all.map((x) => (x.key === key ? { ...x, ...patch } : x)));

  const add = async (name: string, audio: boolean, data: () => Promise<{ blob: Blob; filename: string }>) => {
    const item: Item = { key: `${name}-${Date.now()}-${Math.random()}`, name, audio,
      state: audio ? "converting" : "reading", abort: new AbortController() };
    setItems((all) => [...all, item]);
    setError(null);
    let body: { blob: Blob; filename: string };
    try {
      body = await data();
    } catch {
      return update(item.key, { state: "failed", error: "could not be decoded as audio" });
    }
    update(item.key, { state: "reading" });
    try {
      const info = await api.uploadDocument(body.blob, item.abort.signal, body.filename,
        (seconds) => update(item.key, { seconds }));
      update(item.key, { state: "ready", info });
    } catch (e) {
      if ((e as Error).name !== "AbortError") update(item.key, { state: "failed", error: (e as Error).message });
    }
  };

  const take = (files: FileList | File[]) => {
    for (const file of Array.from(files)) {
      if (isAudio(file))
        add(file.name, true, async () => ({ blob: await toPcm16k(await file.arrayBuffer()), filename: `${file.name}.pcm` }));
      else add(file.name, false, async () => ({ blob: file, filename: file.name }));
    }
  };

  const remove = (key: string) =>
    setItems((all) => {
      const gone = all.find((x) => x.key === key);
      gone?.abort.abort();
      if (gone?.info) api.deleteDocument(gone.info.id).catch(() => {});
      return all.filter((x) => x.key !== key);
    });

  // They go into the case file in one request, once the last one has been read: two requests at a time
  // would each save the matter as they found it, and the first file added would be lost.
  useEffect(() => {
    const ready = items.filter((x) => x.state === "ready");
    if (adding || !ready.length || items.some((x) => x.state === "converting" || x.state === "reading")) return;
    setAdding(true);
    api.addMatterAssets(matterId, { documentIds: ready.map((x) => x.info!.id) }).then(
      (matter) => {
        setItems((all) => all.filter((x) => x.state === "failed"));
        onAdded(matter);
      },
      (e: Error) => setError(e.message),
    ).finally(() => setAdding(false));
  }, [items, adding, matterId, onAdded]);

  return (
    <div className="case-add">
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
      <button className="link-btn" onClick={() => input.current?.click()} disabled={disabled || adding}
        title={disabled ? "The case file cannot change while the matter is being researched"
          : "Read more documents, recordings or scans into this case file"}>
        {adding ? "Adding…" : "+ Add files"}
      </button>
      {items.length > 0 && (
        <ul className="case-items" aria-live="polite">
          {items.map((x) => (
            <li key={x.key} className={`case-item ${x.state}`}>
              <span className="case-kind">{x.audio ? "Recording" : "Document"}</span>
              <span className="case-name" title={x.name}>{x.name}</span>
              <span className="case-meta">
                {x.state === "converting" ? "converting…"
                  : x.state === "failed" ? x.error ?? "could not be read"
                  : x.state === "ready" ? "read"
                  : `${x.audio ? "transcribing" : "reading"}${x.seconds ? ` · ${clock(x.seconds)}` : "…"}`}
              </span>
              <button className="case-remove" onClick={() => remove(x.key)} aria-label={`Remove ${x.name}`} title="Remove">
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      {error && <p className="msg-error">{error}</p>}
    </div>
  );
}
