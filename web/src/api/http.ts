// Api over the FastAPI backend (src/swiss_court_assistant/server). In dev, Vite proxies /api.
// Paths are relative to the page, so the app also works behind a path prefix (e.g. /coder/proxy/8090/).
import type { Api, ChatEvent, Citations, Conversation, ConversationSummary, Decision, Health, Matter,
  MatterEvent, MatterSummary, UploadStatus } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

const enc = encodeURIComponent;

/** Server-sent events: frames separated by a blank line, payload on "data:" lines. */
async function* events<T>(res: Response): AsyncGenerator<T> {
  if (!res.ok || !res.body) throw new Error(await errorText(res));
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let cut: number;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data) as T;
    }
  }
}

const POLL_MS = 1000;
const POLL_TRIES = 5; // consecutive failed polls before giving up: a blip must not lose a long parse

/** Sleeps, or rejects as soon as the caller aborts. */
function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", stop);
      resolve();
    }, ms);
    const stop = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", stop, { once: true });
  });
}

/** Asks after a document being read until it is there, or until reading it failed. */
async function poll(job: UploadStatus, signal?: AbortSignal, onReading?: (seconds: number) => void) {
  let status = job;
  for (let failures = 0; ; ) {
    if (status.state === "ready" && status.document) return status.document;
    if (status.state === "failed") throw new Error(status.error ?? "That file could not be read.");
    onReading?.(status.seconds);
    await wait(POLL_MS, signal);
    try {
      status = await json<UploadStatus>(await fetch(`api/documents/jobs/${enc(job.id)}`, { signal }));
      failures = 0;
    } catch (e) {
      if ((e as Error).name === "AbortError" || ++failures >= POLL_TRIES) throw e;
    }
  }
}

/** Stop reading a file the user removed, and throw away what was stored of it. */
function forget(jobId: string) {
  fetch(`api/documents/jobs/${enc(jobId)}`, { method: "DELETE", keepalive: true }).catch(() => {});
}

export const httpApi: Api = {
  health: () => fetch("api/health").then((r) => json<Health>(r)),
  listConversations: () => fetch("api/conversations").then((r) => json<ConversationSummary[]>(r)),
  getConversation: (id) => fetch(`api/conversations/${enc(id)}`).then((r) => json<Conversation>(r)),
  getDecision: (id) => fetch(`api/decisions/${enc(id)}`).then((r) => json<Decision>(r)),
  getCitations: (id, limit = 20) =>
    fetch(`api/decisions/${enc(id)}/citations?limit=${limit}`).then((r) => json<Citations>(r)),

  async translate(text, source, target) {
    const res = await fetch("api/translate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, source, target }),
    });
    return (await json<{ translation: string }>(res)).translation;
  },

  async speech(text, language, signal) {
    const res = await fetch("api/speech", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, language }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(await errorText(res));
    return { sampleRate: Number(res.headers.get("x-sample-rate")) || 22050, stream: res.body };
  },

  async deleteConversation(id) {
    const res = await fetch(`api/conversations/${enc(id)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await errorText(res));
  },

  async *chat(conversationId, text, signal, documentIds = [], matterId = null) {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ conversationId, message: text, documentIds, matterId }),
      signal,
    });
    yield* events<ChatEvent>(res);
  },

  async uploadDocument(file, signal, filename, onReading) {
    const form = new FormData();
    form.append("file", file, filename ?? (file instanceof File ? file.name : "upload.bin"));
    // Reading a scanned brief page by page takes minutes - longer than a proxy keeps a request open, which
    // is what the 504s were. So the upload only hands the file over, and we ask how far it got.
    const job = await json<UploadStatus>(await fetch("api/documents/jobs", { method: "POST", body: form, signal }));
    try {
      return await poll(job, signal, onReading);
    } catch (e) {
      if ((e as Error).name === "AbortError") forget(job.id);
      throw e;
    }
  },
  async deleteDocument(id) {
    const res = await fetch(`api/documents/${enc(id)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 404) throw new Error(await errorText(res));
  },
  documentFileUrl: (id) => `api/documents/${enc(id)}/file`,

  listMatters: () => fetch("api/matters").then((r) => json<MatterSummary[]>(r)),
  getMatter: (id) => fetch(`api/matters/${enc(id)}`).then((r) => json<Matter>(r)),
  memoUrl: (id, format = "docx") => `api/matters/${enc(id)}/memo${format === "md" ? "?format=md" : ""}`,

  async createMatter({ documentIds = [], text, title }) {
    const form = new FormData();
    for (const id of documentIds) form.append("document_ids", id);
    if (text) form.append("text", text);
    if (title) form.append("title", title);
    return json<Matter>(await fetch("api/matters", { method: "POST", body: form }));
  },

  async addMatterAssets(id, { documentIds = [], text }) {
    const form = new FormData();
    for (const d of documentIds) form.append("document_ids", d);
    if (text) form.append("text", text);
    return json<Matter>(await fetch(`api/matters/${enc(id)}/assets`, { method: "POST", body: form }));
  },

  async deleteMatter(id) {
    const res = await fetch(`api/matters/${enc(id)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await errorText(res));
  },

  async *runMatter(id, signal) {
    const res = await fetch(`api/matters/${enc(id)}/run`, { method: "POST", signal });
    yield* events<MatterEvent>(res);
  },
};
