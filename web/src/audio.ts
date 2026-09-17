/** Recordings for a matter, turned into what the ASR NIM wants: 16 kHz mono 16-bit PCM.
 *
 * The browser already knows how to decode every format it can play, so the conversion happens here
 * rather than with ffmpeg on the server — an uploaded voice memo (m4a, mp3, ogg …) and a recording
 * made in the page both end up as the same bytes. */

const RATE = 16000;

export const AUDIO_TYPES = /\.(mp3|m4a|mp4|wav|ogg|oga|opus|webm|flac|aac|aiff?)$/i;

export function isAudio(file: File): boolean {
  return file.type.startsWith("audio/") || file.type === "video/webm" || AUDIO_TYPES.test(file.name);
}

/** Decode, mix down to mono and resample to 16 kHz; the result is a raw PCM blob. */
export async function toPcm16k(data: ArrayBuffer): Promise<Blob> {
  const ctx = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(data.slice(0));
  } finally {
    await ctx.close();
  }
  const frames = Math.max(1, Math.ceil(decoded.duration * RATE));
  const offline = new OfflineAudioContext(1, frames, RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const mono = (await offline.startRendering()).getChannelData(0);

  const pcm = new Int16Array(mono.length);
  for (let i = 0; i < mono.length; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return new Blob([pcm.buffer], { type: "application/octet-stream" });
}

export interface Recorder {
  stop(): Promise<Blob>;
  cancel(): void;
}

/** Record from the microphone until `stop()`, which hands back the 16 kHz PCM. */
export async function record(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const parts: BlobPart[] = [];
  recorder.ondataavailable = (e) => e.data.size && parts.push(e.data);
  recorder.start(1000);
  const release = () => stream.getTracks().forEach((t) => t.stop());

  return {
    stop: () =>
      new Promise<Blob>((resolve, reject) => {
        recorder.onstop = async () => {
          release();
          try {
            resolve(await toPcm16k(await new Blob(parts, { type: recorder.mimeType }).arrayBuffer()));
          } catch (e) {
            reject(e as Error);
          }
        };
        recorder.stop();
      }),
    cancel: () => {
      if (recorder.state !== "inactive") recorder.stop();
      release();
    },
  };
}
