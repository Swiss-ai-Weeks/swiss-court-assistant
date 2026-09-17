// Api over the FastAPI backend (src/swiss_court_assistant/server). In dev, Vite proxies /api.
// Paths are relative to the page, so the app also works behind a path prefix (e.g. /coder/proxy/8090/).
import type { Api, ChatEvent, Citations, Conversation, ConversationSummary, Decision, Health, Matter,
  MatterEvent, MatterSummary } from "./types";

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

  async *chat(conversationId, text, signal) {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ conversationId, message: text }),
      signal,
    });
    yield* events<ChatEvent>(res);
  },

  listMatters: () => fetch("api/matters").then((r) => json<MatterSummary[]>(r)),
  getMatter: (id) => fetch(`api/matters/${enc(id)}`).then((r) => json<Matter>(r)),
  memoUrl: (id) => `api/matters/${enc(id)}/memo`,

  async createMatter({ file, filename, text, title }) {
    // multipart either way: a document, a recording already decoded to 16 kHz PCM, or typed facts
    const form = new FormData();
    if (file) form.append("file", file, filename ?? (file instanceof File ? file.name : "upload.bin"));
    if (text) form.append("text", text);
    if (title) form.append("title", title);
    return json<Matter>(await fetch("api/matters", { method: "POST", body: form }));
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
