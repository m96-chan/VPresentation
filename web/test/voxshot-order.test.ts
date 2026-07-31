import { describe, expect, it } from "vitest";
import { MemoryVoiceStore, VoxShot, splitSentences } from "voxshot";
import type { EmbedResult, SynthesisEngine, SynthesisRequest } from "voxshot";

/**
 * Ordering contract for `VoxShot.stream()`.
 *
 * The pipeline pairs each yielded `SynthesizedAudio` with a text chunk *by
 * position*, because the generator does not say which text produced what. That
 * is only sound if the generator yields in `splitSentences` order. These tests
 * pin it, using a fake engine so no model or browser is involved.
 *
 * The interesting case is a later chunk that synthesizes faster than an
 * earlier one: if anything ever runs two syntheses concurrently, that is when
 * the order would invert.
 */

const SAMPLE_RATE = 24000;

interface Call {
  readonly text: string;
  readonly startedAt: number;
  finishedAt: number;
}

/**
 * An engine whose chunks finish *in reverse order of arrival*: the first
 * request is the slowest, the last is instant.
 */
class ReverseLatencyEngine implements SynthesisEngine {
  readonly name = "reverse-latency";
  readonly sampleRate = SAMPLE_RATE;
  readonly calls: Call[] = [];
  private seen = 0;

  constructor(private readonly totalChunks: number) {}

  async load(): Promise<void> {}

  async embed(): Promise<Float32Array | EmbedResult> {
    return new Float32Array(192);
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    const order = this.seen++;
    const call: Call = { text: request.text, startedAt: now(), finishedAt: 0 };
    this.calls.push(call);

    // Earlier requests wait longest.
    const delay = (this.totalChunks - order) * 12;
    await new Promise((resolve) => setTimeout(resolve, delay));
    call.finishedAt = now();

    // Stamp the identity of the chunk into the audio so the consumer's pairing
    // can be checked rather than trusted.
    const samples = new Float32Array(Math.round(0.2 * SAMPLE_RATE));
    samples[0] = order;
    return samples;
  }

  async dispose(): Promise<void> {}
}

function now(): number {
  return performance.now();
}

const SCRIPT = [
  "The first sentence is deliberately the slowest one to render.",
  "The second follows it.",
  "A third arrives after that.",
  "The fourth is nearly instant.",
  "And a fifth closes the paragraph.",
].join(" ");

async function makeVoxShot(engine: SynthesisEngine): Promise<VoxShot> {
  const tts = await VoxShot.create({
    engine,
    voiceStore: new MemoryVoiceStore(),
    // Disable the cache: a cache hit returns without awaiting the engine, and
    // this test is about what the engine's timing does to the ordering.
    synthesisCache: null,
  });
  // Must be audible: cloneVoice rejects an all-zero reference.
  const reference = new Float32Array(SAMPLE_RATE);
  for (let i = 0; i < reference.length; i++) reference[i] = Math.sin(i * 0.05) * 0.3;
  await tts.cloneVoice({ samples: reference, sampleRate: SAMPLE_RATE });
  return tts;
}

describe("VoxShot.stream() ordering", () => {
  it("chunks the text the same way splitSentences does", async () => {
    const engine = new ReverseLatencyEngine(5);
    const tts = await makeVoxShot(engine);

    for await (const _ of tts.stream(SCRIPT)) void _;

    expect(engine.calls.map((c) => c.text)).toEqual(splitSentences(SCRIPT));
    await tts.dispose();
  });

  it("yields in request order even when later chunks finish sooner", async () => {
    const engine = new ReverseLatencyEngine(5);
    const tts = await makeVoxShot(engine);

    const yielded: number[] = [];
    for await (const audio of tts.stream(SCRIPT)) {
      // The stamp says which synthesize() call produced this audio.
      yielded.push(audio.samples[0] ?? -1);
    }

    expect(yielded).toEqual([0, 1, 2, 3, 4]);
    await tts.dispose();
  });

  it("never has two syntheses in flight at once", async () => {
    // Overlapping calls are the precondition for any reordering, so this is
    // the property that actually matters.
    const engine = new ReverseLatencyEngine(5);
    const tts = await makeVoxShot(engine);

    for await (const _ of tts.stream(SCRIPT)) void _;

    const calls = engine.calls;
    for (let i = 1; i < calls.length; i++) {
      expect(
        calls[i]!.startedAt,
        `call ${i} ("${calls[i]!.text.slice(0, 24)}…") started before call ${i - 1} finished`,
      ).toBeGreaterThanOrEqual(calls[i - 1]!.finishedAt - 1);
    }
    await tts.dispose();
  });

  it("concatenates in order through speak() too", async () => {
    const engine = new ReverseLatencyEngine(5);
    const tts = await makeVoxShot(engine);

    const audio = await tts.speak(SCRIPT);
    const chunkSamples = Math.round(0.2 * SAMPLE_RATE);
    const stamps = Array.from({ length: 5 }, (_, i) => audio.samples[i * chunkSamples]);

    expect(stamps).toEqual([0, 1, 2, 3, 4]);
    await tts.dispose();
  });
});
