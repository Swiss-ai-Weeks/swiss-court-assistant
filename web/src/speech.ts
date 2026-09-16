// Read aloud: one playback at a time, streamed from api/speech as raw 16-bit PCM and played with Web Audio.
import { useSyncExternalStore } from "react";
import { api } from "./api";

export type SpeechState = "idle" | "loading" | "playing" | "error";

interface Playback {
  key: string;
  state: SpeechState;
  stop: () => void;
}

let current: Playback | null = null;
const listeners = new Set<() => void>();

function set(p: Playback | null) {
  current = p;
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function stopSpeech() {
  current?.stop();
  set(null);
}

async function play(key: string, text: string, language: string) {
  stopSpeech();
  const abort = new AbortController();
  const ctx = new AudioContext();
  const me: Playback = {
    key,
    state: "loading",
    stop: () => {
      abort.abort();
      void ctx.close();
    },
  };
  const update = (state: SpeechState) => current === me && set({ ...me, state });
  set(me);

  try {
    const { sampleRate, stream } = await api.speech(text, language, abort.signal);
    const reader = stream.getReader();
    let next = 0;
    let carry: Uint8Array | null = null; // a byte left over when a chunk splits a sample
    let last: AudioBufferSourceNode | null = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      let bytes = value;
      if (carry) {
        bytes = new Uint8Array(carry.length + value.length);
        bytes.set(carry);
        bytes.set(value, carry.length);
        carry = null;
      }
      if (bytes.length % 2) {
        carry = bytes.slice(-1);
        bytes = bytes.slice(0, -1);
      }
      if (!bytes.length) continue;
      const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
      const buf = ctx.createBuffer(1, pcm.length, sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      next = Math.max(next, ctx.currentTime + 0.05);
      src.start(next);
      next += buf.duration;
      last = src;
      update("playing");
    }
    if (!last) throw new Error("No audio");
    last.onended = () => {
      if (current === me || current?.key === key) stopSpeech();
    };
  } catch (e) {
    if (abort.signal.aborted) return;
    console.error(e);
    update("error");
    void ctx.close();
  }
}

/** State of the playback identified by `key`, and a toggle that starts or stops it. */
export function useSpeech(key: string) {
  const p = useSyncExternalStore(subscribe, () => current);
  const state: SpeechState = p?.key === key ? p.state : "idle";
  const toggle = (text: string, language: string) => {
    if (state === "loading" || state === "playing") stopSpeech();
    else void play(key, text, language);
  };
  return { state, toggle };
}
