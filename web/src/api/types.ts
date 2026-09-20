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
  /** "law": a statute article - its law_id is in decisionId, and `decision` describes the article.
   *  "document": a document the user attached - its id (doc_…) is in decisionId. */
  section: "regeste" | "erwaegung" | "body" | "law" | "document";
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

/** An article the answer names in its text ("Art. 259d CO"), linked to the statute - a reference, not a
 *  citation: numbered citations are evidence for a sentence, these only open the article. */
export interface StatuteRef {
  /** The mention exactly as it appears in the answer. */
  text: string;
  source: Source;
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

/** A question the assistant asked back instead of answering. */
export interface Clarification {
  question: string;
  /** Likely answers, offered as buttons. */
  options: string[];
  /** What the research found before asking; the next turn reads it back. */
  notes: string;
}

/** A document the user attached, parsed (Nemotron Parse for PDFs and scans) and stored on the server. */
export interface DocumentInfo {
  id: string;
  name: string;
  /** A file, a recording of the client (kept as WAV; its text is the transcript), or typed notes. */
  kind?: "document" | "recording" | "notes" | "generated";
  pages: number;
  chars: number;
  /** "nemotron-parse", "python-docx", "text", "nemotron-asr" or "typed". */
  parser: string;
  language: string | null;
  /** Length of a recording. */
  seconds?: number | null;
  createdAt: string;
}

/** A file being read on the server, polled until it is done (see uploadDocument). */
export interface UploadStatus {
  id: string;
  name: string;
  state: "reading" | "ready" | "failed";
  /** How long it has been read. */
  seconds: number;
  document?: DocumentInfo | null;
  error?: string | null;
  status?: number | null;
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
  /** Articles the answer names, linked to their text; set once the answer is complete. */
  statutes?: StatuteRef[] | null;
  /** Set when the assistant asked back instead of answering. */
  clarification?: Clarification | null;
  /** Documents attached to a user message. */
  attachments?: DocumentInfo[] | null;
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
  /** Asked from Case Prep about this matter: answered against its case file and research. */
  matterId?: string | null;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}

/** What the agent is doing: researching with its tools ("thinking", the only stage that streams
 *  reasoning), checking the answer it drafted against the passages it found and revising it
 *  ("checking"), or writing out what passed the checks ("answer"). */
export type Stage = "thinking" | "checking" | "answer";

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
  /** The grounding check on citation `n`, once the passage has been compared with the sentence.
   *  Citations the check rejects are removed from the answer before it is sent, so in practice every
   *  verdict that arrives is `true`; `supported` stays null on a citation that could not be checked. */
  | { type: "verdict"; n: number; supported: boolean }
  /** The assistant asks the user something instead of answering; the turn ends after it. */
  | { type: "clarify"; clarification: Clarification }
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
  statutes?: StatuteRef[] | null;
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
  /** "bundle": several files, recordings and notes. */
  sourceKind: "document" | "recording" | "text" | "bundle";
  /** The uploaded document, kept on the server (its original opens at documentFileUrl). */
  documentId?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Matter extends MatterSummary {
  language: string;
  /** The client's story as text, however it arrived. */
  facts: string;
  /** The case file: every document, recording and note the matter was opened on. */
  assets?: DocumentInfo[];
  /** Passages of the case file in its search index (0: not indexed, so not searchable by meaning). */
  indexed?: number;
  /** Items added to the case file since the last run, which therefore has not seen them. */
  addedSinceRun?: number;
  intake: Intake | null;
  issues: Issue[];
  assessment: string | null;
  memo: string | null;
}

export type MatterEvent =
  | { type: "stage"; stage: MatterStage; status: "running" | "done" }
  | { type: "intake"; title: string; intake: Intake; issues: Issue[] }
  | { type: "indexed"; passages: number }
  | { type: "issue_start"; n: number }
  | { type: "issue_tool"; n: number; name: string; arg: string }
  | { type: "issue_delta"; n: number; text: string }
  | { type: "issue_citation"; n: number; source: Source }
  | { type: "issue_verdict"; n: number; source: number; supported: boolean }
  | { type: "issue_done"; n: number; issue: Issue }
  | { type: "assessment_delta"; text: string }
  | { type: "done"; matter: Matter }
  | { type: "error"; stage: MatterStage; message: string };

/** What the client handed over: documents and recordings already uploaded with uploadDocument, and the
 *  facts typed in. */
export interface MatterInput {
  documentIds?: string[];
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
  /** Streams one assistant turn. A null conversationId starts a new conversation. `documentIds`: documents
   *  uploaded with uploadDocument, attached to this message. `matterId`: a new conversation about this matter
   *  (an existing conversation keeps the matter it was started on). */
  chat(conversationId: string | null, text: string, signal?: AbortSignal, documentIds?: string[],
    matterId?: string | null): AsyncIterable<ChatEvent>;
  /** Parses and stores a document to attach to a message or a matter. A recording goes as 16 kHz PCM with
   *  a name ending in ".pcm"; it is transcribed and kept as audio. The server reads it in the background
   *  and this polls until it is done, so a long scan cannot time out on the way; `onReading` is called
   *  with the seconds it has been read so far. Aborting stops the reading on the server too. */
  uploadDocument(file: Blob, signal?: AbortSignal, filename?: string,
    onReading?: (seconds: number) => void): Promise<DocumentInfo>;
  /** Throws away a document the user took back before it was used. */
  deleteDocument(id: string): Promise<void>;
  /** The document as it was uploaded. */
  documentFileUrl(id: string): string;
  listMatters(): Promise<MatterSummary[]>;
  getMatter(id: string): Promise<Matter>;
  /** Reads the document or recording and opens a matter on it; nothing is researched yet. */
  createMatter(input: MatterInput): Promise<Matter>;
  /** Adds documents, recordings or notes to the case file of a matter that already exists. They are
   *  indexed at once; the research is not redone (see Matter.addedSinceRun). */
  addMatterAssets(id: string, input: MatterInput): Promise<Matter>;
  deleteMatter(id: string): Promise<void>;
  /** Runs the matter through intake, research, assessment and drafting. */
  runMatter(id: string, signal?: AbortSignal): AsyncIterable<MatterEvent>;
  /** Where the drafted memo can be downloaded: Word by default, or Markdown. */
  memoUrl(id: string, format?: "docx" | "md"): string;
}
