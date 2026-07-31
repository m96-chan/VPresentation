/**
 * Text -> speech -> pose track.
 *
 * Synthesis goes through `VoxShot.stream()`, which yields one
 * `SynthesizedAudio` per chunk so rendering can start before the whole article
 * has been spoken.
 *
 * `stream()` does not say which text produced each chunk, but it does not have
 * to: internally it calls the exported `splitSentences(text, {maxLength,
 * minLength})` and yields in order, so calling the same function with the same
 * options reproduces the chunk list exactly and the two zip together. That is
 * why {@link SpeakOptions.maxChunkLength} must match whatever the `VoxShot`
 * instance was built with — it is the one place the pairing could silently go
 * wrong.
 *
 * Note `stream()` is sequential internally (synthesize, yield, repeat), so it
 * is no faster overall than one `speak()` per sentence. What it buys is time
 * to first frame.
 */
import { splitSentences, type SynthesizedAudio, type VoxShot } from "voxshot";

import { tagEmotion } from "./emotion/tagger.js";
import type { EmotionSpan } from "./emotion/emotion.js";
import type { LipsyncOptions } from "./lipsync/lipsync.js";

export interface SpeakOptions {
  readonly tts: VoxShot;
  readonly text: string;
  readonly fps?: number;
  readonly seed?: number;
  readonly blendSeconds?: number;
  readonly lipsyncOptions?: LipsyncOptions;
  /**
   * Must match the `maxChunkLength` the `VoxShot` instance was created with,
   * or the text/audio pairing drifts. Defaults to VoxShot's own default.
   */
  readonly maxChunkLength?: number;
  /** Must match the instance's `minChunkLength`. */
  readonly minChunkLength?: number;
  readonly speed?: number;
  readonly signal?: AbortSignal;
}

/** One synthesized chunk, and the pose frames it produced. */
export interface StreamedChunk {
  readonly index: number;
  readonly total: number;
  readonly text: string;
  readonly audio: SynthesizedAudio;
  readonly span: EmotionSpan;
}

/** The chunks `VoxShot.stream()` will produce for this text. */
export function chunkScript(text: string, options: SpeakOptions | { maxChunkLength?: number; minChunkLength?: number } = {}): string[] {
  return splitSentences(text, {
    maxLength: options.maxChunkLength,
    minLength: options.minChunkLength,
  });
}

/**
 * Synthesize chunk by chunk, yielding each with the text and emotion it
 * carries.
 *
 * `stream()` yields `SynthesizedAudio` without saying which text produced it,
 * so the pairing here is positional: it relies on the generator yielding in
 * the same order `splitSentences` produced. See test/voxshot-order.test.ts,
 * which pins that contract.
 *
 * Poses are not built here. They come from `LivePoseEngine`, which is already
 * running: the caller schedules each chunk into it. Keeping synthesis and
 * animation separate is what stops a late chunk from stalling the character —
 * it just keeps breathing until the audio turns up.
 */
export async function* streamSpeech(options: SpeakOptions): AsyncGenerator<StreamedChunk> {
  const { tts, text } = options;
  const texts = chunkScript(text, options);
  if (texts.length === 0) throw new Error("nothing to speak");

  let cursor = 0;
  let index = 0;

  for await (const audio of tts.stream(text, { speed: options.speed ?? 1 })) {
    options.signal?.throwIfAborted();

    const chunkText = texts[index] ?? "";
    const { emotion, intensity } = tagEmotion(chunkText);
    const span: EmotionSpan = {
      start: cursor,
      end: cursor + audio.duration,
      emotion,
      intensity,
    };
    cursor += audio.duration;

    yield { index, total: texts.length, text: chunkText, audio, span };
    index++;
  }
}

/** Wrap raw PCM in an `AudioBuffer` for playback and for muxing into a video. */
export function toAudioBuffer(
  samples: Float32Array,
  sampleRate: number,
  context: BaseAudioContext,
): AudioBuffer {
  const buffer = context.createBuffer(1, samples.length, sampleRate);
  // Copy into a freshly allocated, ArrayBuffer-backed view: `copyToChannel`
  // rejects SharedArrayBuffer-backed arrays, and VoxShot's samples may be
  // either depending on whether synthesis ran in a worker.
  const owned = new Float32Array(samples.length);
  owned.set(samples);
  buffer.copyToChannel(owned, 0);
  return buffer;
}
