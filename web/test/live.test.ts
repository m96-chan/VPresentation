import { describe, expect, it } from "vitest";
import { LivePoseEngine } from "../src/track/live.js";
import { NUM_POSE_PARAMS, POSE_INDEX } from "../src/pose/params.js";

const SR = 24000;
const SEED = 20260731;

function resonate(x: Float32Array, freq: number, bw: number): Float32Array {
  const r = Math.exp((-Math.PI * bw) / SR);
  const a1 = 2 * r * Math.cos((2 * Math.PI * freq) / SR);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) - r * r * (y[n - 2] ?? 0);
  }
  return y;
}

function speech(seconds: number): Float32Array {
  const n = Math.round(seconds * SR);
  const src = new Float32Array(n);
  for (let i = 0; i < n; i += Math.round(SR / 120)) {
    if (((i / SR) * 4) % 1 < 0.62) src[i] = 1;
  }
  const y = resonate(resonate(src, 730, 80), 1090, 110);
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  return (peak > 0 ? y.map((v) => v / peak) : y) as Float32Array;
}

/** Walk the engine at a fixed rate and collect one channel. */
function walk(engine: LivePoseEngine, seconds: number, slot: number, fps = 30): number[] {
  const out: number[] = [];
  for (let f = 0; f < seconds * fps; f++) out.push(engine.frameAt(f / fps)[slot]!);
  return out;
}

describe("idle without any audio", () => {
  // The character is a character before it is a speaker: it breathes and
  // blinks whether or not anything is being said. Speech is layered into that,
  // not the thing that starts it.
  it("produces frames with no utterance at all", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    const pose = engine.frameAt(0);
    expect(pose).toHaveLength(NUM_POSE_PARAMS);
  });

  it("breathes forever", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    const breath = walk(engine, 10, POSE_INDEX.breathing);
    expect(Math.max(...breath) - Math.min(...breath)).toBeGreaterThan(0.8);
  });

  it("blinks forever", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    const blink = walk(engine, 30, POSE_INDEX.eye_wink_left);
    expect(Math.max(...blink)).toBeGreaterThan(0.9);
  });

  it("keeps turning to look around", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    const yaw = walk(engine, 15, POSE_INDEX.head_y);
    expect(Math.max(...yaw) - Math.min(...yaw)).toBeGreaterThan(0.3);
  });

  it("keeps the mouth shut", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    expect(walk(engine, 5, POSE_INDEX.mouth_aaa).every((v) => v === 0)).toBe(true);
  });

  it("drifts into the thinking gaze, since silence is what it is doing", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    const gazePitch = walk(engine, 6, POSE_INDEX.iris_rotation_x);
    // iris_rotation_x is the gaze's *pitch*, positive up.
    expect(Math.max(...gazePitch)).toBeGreaterThan(0.2);
  });
});

describe("speech layered into the idle", () => {
  function engineWithSpeech(startTime: number) {
    const engine = new LivePoseEngine({ seed: SEED });
    engine.beginSpeech(startTime, SR);
    engine.pushAudio(speech(2), {
      start: startTime,
      end: startTime + 2,
      emotion: "happy",
      intensity: 0.9,
    });
    engine.endSpeech();
    return engine;
  }

  it("leaves the mouth shut before the utterance starts", () => {
    const engine = engineWithSpeech(3);
    for (let f = 0; f < 60; f++) {
      expect(engine.frameAt(f / 30)[POSE_INDEX.mouth_aaa]).toBe(0);
    }
  });

  it("opens the mouth during the utterance", () => {
    const engine = engineWithSpeech(3);
    const mouth = walk(engine, 6, POSE_INDEX.mouth_aaa);
    const during = mouth.slice(3 * 30 + 15, 5 * 30);
    expect(Math.max(...during)).toBeGreaterThan(0.3);
  });

  it("closes the mouth again afterwards", () => {
    const engine = engineWithSpeech(1);
    const mouth = walk(engine, 6, POSE_INDEX.mouth_aaa);
    expect(mouth.slice(4 * 30).every((v) => v < 0.05)).toBe(true);
  });

  it("applies the emotion only across its span", () => {
    const engine = engineWithSpeech(2);
    const brow = walk(engine, 8, POSE_INDEX.eyebrow_happy_left);
    expect(brow[30]!).toBe(0); // t = 1s, before
    expect(Math.max(...brow.slice(2 * 30 + 10, 4 * 30))).toBeGreaterThan(0.3);
    expect(brow[brow.length - 1]!).toBe(0); // well after
  });

  it("keeps breathing right through the utterance", () => {
    const engine = engineWithSpeech(1);
    // A full 4 s breath cycle starting inside the utterance.
    const breath = walk(engine, 6, POSE_INDEX.breathing).slice(30, 30 + 4 * 30);
    expect(Math.max(...breath) - Math.min(...breath)).toBeGreaterThan(0.9);
  });
});

describe("clock handling", () => {
  it("is deterministic when stepped at a fixed rate", () => {
    const a = walk(new LivePoseEngine({ seed: 5 }), 6, POSE_INDEX.head_y);
    const b = walk(new LivePoseEngine({ seed: 5 }), 6, POSE_INDEX.head_y);
    expect(a).toEqual(b);
  });

  it("stays stable when frames are dropped", () => {
    // The realtime renderer skips frames when inference lags, so the engine is
    // stepped with a variable dt. The springs are implicit, so this must not
    // blow up or snap.
    const engine = new LivePoseEngine({ seed: SEED });
    let prev = engine.frameAt(0)[POSE_INDEX.head_y]!;
    for (const t of [0.3, 0.35, 0.9, 0.95, 2.4, 2.45, 5.0]) {
      const v = engine.frameAt(t)[POSE_INDEX.head_y]!;
      expect(Number.isFinite(v)).toBe(true);
      expect(Math.abs(v)).toBeLessThanOrEqual(1);
      prev = v;
    }
    expect(Number.isFinite(prev)).toBe(true);
  });

  it("tolerates being asked for the same time twice", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    engine.frameAt(1);
    const a = [...engine.frameAt(1)];
    expect(a.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("ignores time going backwards rather than exploding", () => {
    const engine = new LivePoseEngine({ seed: SEED });
    engine.frameAt(5);
    const pose = engine.frameAt(1);
    expect([...pose].every((v) => Number.isFinite(v))).toBe(true);
  });
});
