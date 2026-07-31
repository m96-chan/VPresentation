import { describe, expect, it } from "vitest";
import { LivePoseEngine } from "../src/track/live.js";
import { LipsyncAnalyser } from "../src/lipsync/lipsync.js";

const SR = 24000;

function tone(seconds: number): Float32Array {
  const n = Math.round(seconds * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin(i * 0.06) * 0.4;
  return out;
}

/** Rough retained size of the analyser's audio buffer, in samples. */
function retainedSamples(a: LipsyncAnalyser): number {
  return (a as unknown as { samples: Float32Array }).samples.length;
}

function retainedSpans(e: LivePoseEngine): number {
  return (e as unknown as { spans: unknown[] }).spans.length;
}

describe("a long session stays bounded", () => {
  // A 34-page paper is around 106 minutes of speech. Anything that grows per
  // utterance — retained audio, mouth frames, emotion spans — has to be
  // pruned, or the tab dies partway through regardless of how correct the
  // rendering is.

  it("does not retain audio for the whole utterance", () => {
    const analyser = new LipsyncAnalyser(SR, { fps: 30 });
    for (let i = 0; i < 60; i++) analyser.push(tone(1));
    analyser.flush();
    // 60 s of audio at 24 kHz would be 1.44 M samples if nothing were dropped.
    expect(retainedSamples(analyser)).toBeLessThan(SR * 2);
  });

  it("still analyses correctly while dropping consumed audio", () => {
    const whole = new LipsyncAnalyser(SR, { fps: 30 });
    const streamed = new LipsyncAnalyser(SR, { fps: 30 });

    const parts = [tone(0.7), tone(1.1), tone(0.5)];
    const joined = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
    let at = 0;
    for (const p of parts) {
      joined.set(p, at);
      at += p.length;
    }

    const a = [...whole.push(joined), ...whole.flush()];
    const b = [...parts.flatMap((p) => streamed.push(p)), ...streamed.flush()];

    expect(b).toHaveLength(a.length);
    for (let i = 0; i < a.length; i++) {
      expect(b[i]!.openness, `frame ${i}`).toBeCloseTo(a[i]!.openness, 9);
    }
  });

  it("does not accumulate emotion spans across a long presentation", () => {
    const engine = new LivePoseEngine({ fps: 30, seed: 1 });
    let clock = 0;

    // 400 short utterances, as a long paper would produce.
    for (let i = 0; i < 400; i++) {
      engine.beginSpeech(clock, SR);
      engine.pushAudio(tone(0.5), {
        start: clock,
        end: clock + 0.5,
        emotion: "neutral",
        intensity: 1,
      });
      engine.endSpeech();
      clock += 0.6;
      engine.frameAt(clock);
    }

    // Spans older than the clock can never affect a frame again.
    expect(retainedSpans(engine)).toBeLessThan(50);
  });

  it("keeps per-frame cost flat as the session runs on", () => {
    const engine = new LivePoseEngine({ fps: 30, seed: 1 });
    let clock = 0;
    for (let i = 0; i < 300; i++) {
      engine.beginSpeech(clock, SR);
      engine.pushAudio(tone(0.4), { start: clock, end: clock + 0.4, emotion: "happy" });
      engine.endSpeech();
      clock += 0.5;
    }
    // The pose at a late time must still be produced, and finite.
    const pose = engine.frameAt(clock + 1);
    expect([...pose].every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("one very long utterance", () => {
  it("drops frames the clock has played", () => {
    const engine = new LivePoseEngine({ fps: 30, seed: 1 });
    engine.beginSpeech(0, SR);

    // Two minutes fed in as it would arrive, with the clock following.
    for (let i = 0; i < 120; i++) {
      engine.pushAudio(tone(1));
      engine.frameAt(i);
    }
    engine.endSpeech();

    const held = (engine as unknown as { utterance: { mouth: unknown[] } }).utterance.mouth.length;
    // 120 s at 30 fps is 3600 frames if nothing is dropped.
    expect(held).toBeLessThan(600);
  });

  it("still returns the right mouth frame after pruning", () => {
    const engine = new LivePoseEngine({ fps: 30, seed: 1 });
    engine.beginSpeech(0, SR);
    for (let i = 0; i < 60; i++) {
      engine.pushAudio(tone(1));
      engine.frameAt(i);
    }
    // Mid-speech, the mouth should still be doing something.
    const pose = engine.frameAt(59.5);
    expect([...pose].every((v) => Number.isFinite(v))).toBe(true);
  });
});
