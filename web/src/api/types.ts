// Shapes shared by the mock API and the future HTTP backend.

export interface DecisionSummary {
  decisionId: string;
  court: string;
  courtLabel: string;
  canton: string | null;
  chamber: string | null;
  docket: string;
  date: string | null;
  language: string;
  title: string | null;
  regeste: string | null;
  legalArea: string | null;
  sourceUrl: string | null;
  pdfUrl: string | null;
}

export interface Decision extends DecisionSummary {
  fullText: string;
}

/** A retrieved passage, numbered as cited in the answer ([n]). Text is verbatim from the decision. */
export interface Source {
  n: number;
  chunkId: string;
  decisionId: string;
  text: string;
  section: "regeste" | "erwaegung" | "body";
  erwaegungen: string[];
  /** Offsets into Decision.fullText; null for Regeste passages. */
  charStart: number | null;
  charEnd: number | null;
  score: number;
  decision: DecisionSummary;
  /** Why the agent cites this passage. */
  explanation?: string | null;
  /** False when the quoted words were not found verbatim in the decision. */
  verified?: boolean;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  /** Null while the tool is running. */
  summary?: string | null;
  error?: boolean;
  /** The agent's reasoning before this call. */
  thought?: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  /** Assistant answers: markdown with [n] citations into `sources`. */
  content: string;
  searchQuery?: string | null;
  sources?: Source[] | null;
  toolCalls?: ToolCall[] | null;
  /** Language of the question (on assistant messages); cited passages can be translated into it. */
  language?: string | null;
  createdAt: string;
}

export interface Health {
  status: string;
  /** "stub" until the real agent lands. */
  agent: string;
  decisions: number;
  /** Language codes the speech recogniser understands; empty when voice mode is unavailable. */
  speechLanguages?: string[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: string;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}

export type Stage = "thinking" | "answer";

export type ChatEvent =
  | { type: "conversation"; conversation: ConversationSummary }
  | { type: "meta"; language: string }
  | { type: "status"; stage: Stage; detail: string }
  /** Reasoning streamed while the agent decides on its next tool call. */
  | { type: "thinking"; text: string }
  | { type: "tool_start"; call: ToolCall }
  | { type: "tool_end"; id: string; summary: string; error: boolean }
  | { type: "delta"; text: string }
  /** Cites `source` right after the text so far; the same source may be cited again. */
  | { type: "citation"; source: Source }
  | { type: "done"; message: Message }
  | { type: "error"; message: string };

export interface Api {
  health(): Promise<Health>;
  listConversations(): Promise<ConversationSummary[]>;
  getConversation(id: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  getDecision(id: string): Promise<Decision>;
  /** Machine translation of a cited passage (language codes: de, fr, it, rm, en). */
  translate(text: string, source: string, target: string): Promise<string>;
  /** Speech for `text` read in `language`: a stream of 16-bit little-endian mono PCM at `sampleRate`. */
  speech(text: string, language: string, signal?: AbortSignal): Promise<{ sampleRate: number; stream: ReadableStream<Uint8Array> }>;
  /** Streams one assistant turn. A null conversationId starts a new conversation. */
  chat(conversationId: string | null, text: string, signal?: AbortSignal): AsyncIterable<ChatEvent>;
}
