// Voice mode: microphone → /api/voice (WebSocket) → transcripts, agent events and spoken answer.
// The assistant's speech plays while its text streams; speaking over it interrupts and starts a new turn.
import type { ChatEvent, Message } from "./api";

export type VoiceEvent =
  | ChatEvent
  | { type: "partial"; text: string; final: boolean }
  | { type: "user"; message: Message }
  | { type: "speech_start"; text: string; sampleRate: number }
  | { type: "speech_end" }
  | { type: "cancel_speech" }
  /** Local: the assistant's voice is actually sounding (the socket delivers it far ahead of playback). */
  | { type: "speaking"; on: boolean }
  /** Local: the socket closed or the microphone was refused. */
  | { type: "voice_ended"; reason?: string };

const MIC_RATE = 16000; // what the ASR NIM expects
const FRAME = 512; // samples per message (~32 ms)

// An AudioWorklet cannot be a separate file here (the page may run behind a path prefix), so it is
// compiled from source at runtime: it just forwards raw microphone frames to the main thread.
const WORKLET = `
class MicTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(${FRAME}); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("mic-tap", MicTap);
`;

export interface VoiceSession {
  stop(): void;
}

function socketUrl(): string {
  const url = new URL("api/voice", window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

/** Plays the PCM chunks the server streams, and can drop what is still queued when interrupted. */
class Playback {
  private ctx = new AudioContext();
  private next = 0;
  private sources: AudioBufferSourceNode[] = [];
  private timer = 0;
  private sounding = false;
  rate = 22050;

  constructor(private notify: (on: boolean) => void) {}

  private set(on: boolean) {
    if (on !== this.sounding) {
      this.sounding = on;
      this.notify(on);
    }
  }

  /** Playing until the last scheduled chunk has been heard. */
  private watch() {
    window.clearTimeout(this.timer);
    const left = Math.max(0, this.next - this.ctx.currentTime);
    this.timer = window.setTimeout(() => this.set(false), left * 1000 + 80);
  }

  push(bytes: ArrayBuffer) {
    if (this.ctx.state === "closed") return;
    const pcm = new Int16Array(bytes);
    if (!pcm.length) return;
    const buffer = this.ctx.createBuffer(1, pcm.length, this.rate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);
    this.next = Math.max(this.next, this.ctx.currentTime + 0.08);
    src.start(this.next);
    this.next += buffer.duration;
    this.sources.push(src);
    src.onended = () => (this.sources = this.sources.filter((s) => s !== src));
    this.set(true);
    this.watch();
  }

  /** Stop immediately (the user started talking). */
  cancel() {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already finished */
      }
    }
    this.sources = [];
    this.next = 0;
    window.clearTimeout(this.timer);
    this.set(false);
  }

  close() {
    this.cancel();
    if (this.ctx.state !== "closed") void this.ctx.close();
  }
}

export async function startVoice(opts: {
  conversationId: string | null;
  language: string;
  onEvent: (e: VoiceEvent) => void;
}): Promise<VoiceSession> {
  const stream = await navigator.mediaDevices.getUserMedia({
    // the microphone hears the assistant's own voice; let the browser cancel it
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const mic = new AudioContext({ sampleRate: MIC_RATE });
  const playback = new Playback((on) => opts.onEvent({ type: "speaking", on }));
  const ws = new WebSocket(socketUrl());
  ws.binaryType = "arraybuffer";

  const stop = () => {
    try {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
      ws.close();
    } catch {
      /* already closing */
    }
    stream.getTracks().forEach((t) => t.stop());
    void mic.close();
    playback.close();
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return playback.push(e.data);
    const event = JSON.parse(e.data) as VoiceEvent;
    if (event.type === "speech_start") playback.rate = event.sampleRate || playback.rate;
    if (event.type === "cancel_speech") playback.cancel();
    opts.onEvent(event);
  };
  ws.onerror = () => opts.onEvent({ type: "voice_ended", reason: "The voice connection failed." });
  ws.onclose = () => {
    playback.close();
    opts.onEvent({ type: "voice_ended" });
  };

  await new Promise<void>((resolve, reject) => {
    ws.onopen = () => resolve();
    setTimeout(() => reject(new Error("The voice connection timed out.")), 10000);
  });
  ws.send(JSON.stringify({ type: "start", conversationId: opts.conversationId, language: opts.language }));

  await mic.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  const tap = new AudioWorkletNode(mic, "mic-tap");
  tap.port.onmessage = (e: MessageEvent<Float32Array>) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const floats = e.data;
    const pcm = new Int16Array(floats.length);
    for (let i = 0; i < floats.length; i++) pcm[i] = Math.max(-1, Math.min(1, floats[i])) * 32767;
    ws.send(pcm.buffer);
  };
  mic.createMediaStreamSource(stream).connect(tap);
  // the tap produces no output; connecting it keeps the graph alive in some browsers
  tap.connect(mic.destination);

  return { stop };
}
