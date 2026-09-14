// Api over the FastAPI backend (src/swiss_court_assistant/server). In dev, Vite proxies /api.
// Paths are relative to the page, so the app also works behind a path prefix (e.g. /coder/proxy/8090/).
import type { Api, ChatEvent, Conversation, ConversationSummary, Decision, Health } from "./types";

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

export const httpApi: Api = {
  health: () => fetch("api/health").then((r) => json<Health>(r)),
  listConversations: () => fetch("api/conversations").then((r) => json<ConversationSummary[]>(r)),
  getConversation: (id) => fetch(`api/conversations/${enc(id)}`).then((r) => json<Conversation>(r)),
  getDecision: (id) => fetch(`api/decisions/${enc(id)}`).then((r) => json<Decision>(r)),

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
    if (!res.ok || !res.body) throw new Error(await errorText(res));

    // Server-sent events: frames separated by a blank line, payload on "data:" lines.
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
        if (data) yield JSON.parse(data) as ChatEvent;
      }
    }
  },
};
