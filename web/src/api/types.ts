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
  /** Whether the passage really states the sentence citing it; null while unchecked. */
  supported?: boolean | null;
}

/** A decision at the other end of a citation edge (the corpus citation graph). */
export interface CitingDecision {
  decisionId: string;
  court: string | null;
  docket: string | null;
  date: string | null;
  /** Part of this app's corpus, so it can be opened here. */
  inCorpus: boolean;
}

export interface Citations {
  decisionId: string;
  citedByCount: number;
  citesCount: number;
  citedBy: CitingDecision[];
  cites: CitingDecision[];
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
  /** The grounding check on citation `n`, once the passage has been compared with the sentence. */
  | { type: "verdict"; n: number; supported: boolean }
  | { type: "done"; message: Message }
  | { type: "error"; message: string };

// ── matters: one client case through intake, research, assessment and drafting ──
export type MatterStage = "new" | "intake" | "research" | "assessment" | "drafting" | "done";

export interface Issue {
  n: number;
  question: string;
  /** Why this question decides the case. */
  why: string;
  area: string | null;
  /** Markdown with [n] citations into `sources`; null until the research has run. */
  answer: string | null;
  sources: Source[] | null;
}

export interface Intake {
  summary: string;
  parties: string[];
  timeline: string[];
}

export interface MatterSummary {
  id: string;
  title: string;
  stage: MatterStage;
  sourceName: string | null;
  sourceKind: "document" | "recording" | "text";
  createdAt: string;
  updatedAt: string;
}

export interface Matter extends MatterSummary {
  language: string;
  /** The client's story as text, however it arrived. */
  facts: string;
  intake: Intake | null;
  issues: Issue[];
  assessment: string | null;
  memo: string | null;
}

export type MatterEvent =
  | { type: "stage"; stage: MatterStage; status: "running" | "done" }
  | { type: "intake"; title: string; intake: Intake; issues: Issue[] }
  | { type: "issue_start"; n: number }
  | { type: "issue_tool"; n: number; name: string; arg: string }
  | { type: "issue_delta"; n: number; text: string }
  | { type: "issue_citation"; n: number; source: Source }
  | { type: "issue_verdict"; n: number; source: number; supported: boolean }
  | { type: "issue_done"; n: number; issue: Issue }
  | { type: "assessment_delta"; text: string }
  | { type: "done"; matter: Matter }
  | { type: "error"; stage: MatterStage; message: string };

/** What the client handed over: a file (document or audio), or the facts typed in. */
export interface MatterInput {
  file?: File | Blob;
  filename?: string;
  text?: string;
  title?: string;
}

export interface Api {
  health(): Promise<Health>;
  listConversations(): Promise<ConversationSummary[]>;
  getConversation(id: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  getDecision(id: string): Promise<Decision>;
  /** How often later decisions cite this one, and which. */
  getCitations(id: string, limit?: number): Promise<Citations>;
  /** Machine translation of a cited passage (language codes: de, fr, it, rm, en). */
  translate(text: string, source: string, target: string): Promise<string>;
  /** Speech for `text` read in `language`: a stream of 16-bit little-endian mono PCM at `sampleRate`. */
  speech(text: string, language: string, signal?: AbortSignal): Promise<{ sampleRate: number; stream: ReadableStream<Uint8Array> }>;
  /** Streams one assistant turn. A null conversationId starts a new conversation. */
  chat(conversationId: string | null, text: string, signal?: AbortSignal): AsyncIterable<ChatEvent>;
  listMatters(): Promise<MatterSummary[]>;
  getMatter(id: string): Promise<Matter>;
  /** Reads the document or recording and opens a matter on it; nothing is researched yet. */
  createMatter(input: MatterInput): Promise<Matter>;
  deleteMatter(id: string): Promise<void>;
  /** Runs the matter through intake, research, assessment and drafting. */
  runMatter(id: string, signal?: AbortSignal): AsyncIterable<MatterEvent>;
  /** Where the drafted memo can be downloaded as Markdown. */
  memoUrl(id: string): string;
}
