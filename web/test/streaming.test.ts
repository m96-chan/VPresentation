import { describe, expect, it } from "vitest";
import { analyseLipsync, LipsyncAnalyser } from "../src/lipsync/lipsync.js";
import { buildPoseTrack, PoseTrackBuilder } from "../src/track/posetrack.js";
import { NUM_POSE_PARAMS } from "../src/pose/params.js";
import type { EmotionSpan } from "../src/emotion/emotion.js";

const SR = 24000;

function resonate(x: Float32Array, freq: number, bw: number, sr: number): Float32Array {
  const r = Math.exp((-Math.PI * bw) / sr);
  const a1 = 2 * r * Math.cos((2 * Math.PI * freq) / sr);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) - r * r * (y[n - 2] ?? 0);
  }
  return y;
}

/** Speech-ish audio of the given length. */
function speech(seconds: number, f1 = 730, f2 = 1090): Float32Array {
  const n = Math.round(seconds * SR);
  const src = new Float32Array(n);
  for (let i = 0; i < n; i += Math.round(SR / 120)) {
    if (((i / SR) * 4) % 1 < 0.62) src[i] = 1;
  }
  const y = resonate(resonate(src, f1, 80, SR), f2, 110, SR);
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  return (peak > 0 ? y.map((v) => v / peak) : y) as Float32Array;
}

function concat(parts: readonly Float32Array[]): Float32Array {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Float32Array(total);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

describe("LipsyncAnalyser", () => {
  const parts = [speech(0.9), speech(1.3, 270, 2290), speech(0.7, 300, 870)];
  const whole = concat(parts);

  it("streamed analysis matches batch analysis frame for frame", () => {
    // This is the property that lets playback start before synthesis finishes:
    // if streaming produced even slightly different frames, the recorded video
    // would not match what was watched.
    const batch = analyseLipsync(whole, SR, { fps: 30 });

    const analyser = new LipsyncAnalyser(SR, { fps: 30 });
    const streamed = [...parts.flatMap((p) => analyser.push(p)), ...analyser.flush()];

    expect(streamed).toHaveLength(batch.length);
    for (let i = 0; i < batch.length; i++) {
      expect(streamed[i]!.time, `frame ${i} time`).toBeCloseTo(batch[i]!.time, 9);
      expect(streamed[i]!.openness, `frame ${i} openness`).toBeCloseTo(batch[i]!.openness, 9);
      for (const v of ["aaa", "iii", "uuu", "eee", "ooo"] as const) {
        expect(streamed[i]!.vowels[v], `frame ${i} ${v}`).toBeCloseTo(batch[i]!.vowels[v], 9);
      }
    }
  });

  it("holds back frames whose analysis window is not yet complete", () => {
    // A centred window needs audio from *after* the frame. That only forces a
    // hold-back when the window's half-width exceeds the hop, so this uses a
    // 120 ms window (half-width 60 ms) against a 33 ms hop.
    const analyser = new LipsyncAnalyser(SR, { fps: 30, windowSeconds: 0.12 });
    const emitted = analyser.push(speech(1));
    expect(emitted.length).toBeLessThan(30);
    expect(emitted.length).toBeGreaterThan(25);
    expect(emitted.length + analyser.flush().length).toBe(30);
  });

  it("does not hold anything back needlessly at the default window", () => {
    // 40 ms window, 33 ms hop: every frame's window closes before the next
    // frame starts, so a chunk yields all of its frames immediately.
    const analyser = new LipsyncAnalyser(SR, { fps: 30 });
    expect(analyser.push(speech(1))).toHaveLength(30);
  });

  it("emits nothing extra once flushed", () => {
    const analyser = new LipsyncAnalyser(SR, { fps: 30 });
    analyser.push(speech(0.5));
    analyser.flush();
    expect(analyser.flush()).toEqual([]);
  });

  it("handles being pushed empty chunks", () => {
    const analyser = new LipsyncAnalyser(SR, { fps: 30 });
    expect(analyser.push(new Float32Array(0))).toEqual([]);
  });
});

describe("PoseTrackBuilder", () => {
  const parts = [speech(1.1), speech(1.4, 270, 2290), speech(0.9, 570, 840)];
  const whole = concat(parts);
  const spans: EmotionSpan[] = [];
  {
    let at = 0;
    const names = ["happy", "sad", "surprised"] as const;
    parts.forEach((p, i) => {
      const duration = p.length / SR;
      spans.push({ start: at, end: at + duration, emotion: names[i]!, intensity: 0.8 });
      at += duration;
    });
  }

  const options = { fps: 30, seed: 20260731, blendSeconds: 0.3 } as const;

  it("produces exactly the same track as building it in one go", () => {
    // Every stateful part — lipsync smoothing, the speech envelope, and the
    // springs that give the body inertia — has to carry across chunk
    // boundaries. If any of them reset, the head would snap at every sentence.
    const batch = buildPoseTrack({
      samples: whole,
      sampleRate: SR,
      emotions: spans,
      ...options,
    });

    const builder = new PoseTrackBuilder(SR, { ...options });
    parts.forEach((p, i) => builder.push(p, spans[i]));
    const streamed = builder.finish();

    expect(streamed.frameCount).toBe(batch.frameCount);
    for (let i = 0; i < batch.data.length; i++) {
      expect(streamed.data[i], `slot ${i} (frame ${Math.floor(i / NUM_POSE_PARAMS)})`).toBeCloseTo(
        batch.data[i]!,
        6,
      );
    }
  });

  it("exposes frames as they become available", () => {
    const builder = new PoseTrackBuilder(SR, { ...options });
    expect(builder.frameCount).toBe(0);

    builder.push(parts[0]!, spans[0]);
    const afterFirst = builder.frameCount;
    expect(afterFirst).toBeGreaterThan(20);

    builder.push(parts[1]!, spans[1]);
    expect(builder.frameCount).toBeGreaterThan(afterFirst);
  });

  it("can be read while still being built", () => {
    const builder = new PoseTrackBuilder(SR, { ...options });
    builder.push(parts[0]!, spans[0]);
    const pose = builder.poseAt(5);
    expect(pose).toHaveLength(NUM_POSE_PARAMS);
  });

  it("is a no-op to finish an empty builder", () => {
    const track = new PoseTrackBuilder(SR, { ...options }).finish();
    expect(track.frameCount).toBe(0);
  });
});
