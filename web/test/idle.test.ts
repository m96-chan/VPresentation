import { describe, expect, it } from "vitest";
import { POSE_INDEX, zeroPose } from "../src/pose/params.js";
import { Idle, applyIdle, blinkAmount, breathingAmount } from "../src/idle/idle.js";

const SEED = 20260731;

describe("determinism", () => {
  // Realtime and offline rendering must agree frame for frame, so every idle
  // signal is a pure function of (time, seed) — no Math.random anywhere.
  it("gives identical results for the same time and seed", () => {
    for (const t of [0, 0.37, 1.5, 9.25, 60.125]) {
      expect(blinkAmount(t, SEED)).toBe(blinkAmount(t, SEED));
      expect(breathingAmount(t)).toBe(breathingAmount(t));
    }
  });

  it("gives different blink schedules for different seeds", () => {
    const a = Array.from({ length: 600 }, (_, i) => blinkAmount(i / 30, 1));
    const b = Array.from({ length: 600 }, (_, i) => blinkAmount(i / 30, 2));
    expect(a).not.toEqual(b);
  });

  it("an Idle instance matches the free functions", () => {
    const idle = new Idle({ seed: SEED });
    const pose = zeroPose();
    idle.apply(pose, 3.5);
    expect(pose[POSE_INDEX.eye_wink_left]).toBeCloseTo(blinkAmount(3.5, SEED), 6);
  });
});

describe("blink", () => {
  it("stays within 0..1", () => {
    for (let i = 0; i < 3000; i++) {
      const v = blinkAmount(i / 30, SEED);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("keeps the eyes open the vast majority of the time", () => {
    const samples = Array.from({ length: 3000 }, (_, i) => blinkAmount(i / 30, SEED));
    const closedish = samples.filter((v) => v > 0.5).length;
    expect(closedish / samples.length).toBeLessThan(0.15);
  });

  it("actually closes the eyes sometimes", () => {
    const samples = Array.from({ length: 3000 }, (_, i) => blinkAmount(i / 30, SEED));
    expect(Math.max(...samples)).toBeGreaterThan(0.9);
  });

  it("blinks at a plausible rate — several times per 30 seconds", () => {
    let count = 0;
    let closed = false;
    for (let i = 0; i < 900; i++) {
      const v = blinkAmount(i / 30, SEED);
      if (!closed && v > 0.5) {
        count++;
        closed = true;
      } else if (closed && v < 0.2) {
        closed = false;
      }
    }
    expect(count).toBeGreaterThanOrEqual(4);
    expect(count).toBeLessThanOrEqual(30);
  });

  it("is smooth — no instant open-to-shut jump between frames", () => {
    let prev = blinkAmount(0, SEED);
    for (let i = 1; i < 3000; i++) {
      const v = blinkAmount(i / 30, SEED);
      expect(Math.abs(v - prev)).toBeLessThan(0.75);
      prev = v;
    }
  });
});

describe("breathing", () => {
  it("stays within 0..1", () => {
    for (let i = 0; i < 1000; i++) {
      const v = breathingAmount(i / 30);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("is periodic", () => {
    const period = 4;
    expect(breathingAmount(1.234)).toBeCloseTo(breathingAmount(1.234 + period), 5);
  });

  it("actually varies", () => {
    const samples = Array.from({ length: 240 }, (_, i) => breathingAmount(i / 30));
    expect(Math.max(...samples) - Math.min(...samples)).toBeGreaterThan(0.5);
  });
});

describe("applyIdle", () => {
  it("writes only blink and breathing slots", () => {
    const pose = zeroPose();
    pose[POSE_INDEX.mouth_aaa] = 0.9;
    pose[POSE_INDEX.eyebrow_happy_left] = 0.5;
    pose[POSE_INDEX.head_x] = 0.3; // owned by motion/body.ts
    applyIdle(pose, 2.0, SEED);

    expect(pose[POSE_INDEX.mouth_aaa]).toBeCloseTo(0.9, 6);
    expect(pose[POSE_INDEX.eyebrow_happy_left]).toBeCloseTo(0.5, 6);
    expect(pose[POSE_INDEX.head_x]).toBeCloseTo(0.3, 6);
    expect(pose[POSE_INDEX.breathing]).toBeGreaterThan(0);
  });

  it("blinks both eyes together", () => {
    // Find a frame where the eyes are actually closing.
    let t = 0;
    for (let i = 0; i < 3000; i++) {
      if (blinkAmount(i / 30, SEED) > 0.5) {
        t = i / 30;
        break;
      }
    }
    const pose = zeroPose();
    applyIdle(pose, t, SEED);
    expect(pose[POSE_INDEX.eye_wink_left]).toBeGreaterThan(0.5);
    expect(pose[POSE_INDEX.eye_wink_left]).toBe(pose[POSE_INDEX.eye_wink_right]);
  });

  it("does not fight an emotion that already set eye_wink", () => {
    // Idle takes the max, so a deliberate wink is never undone by idle.
    const pose = zeroPose();
    pose[POSE_INDEX.eye_wink_left] = 1;
    pose[POSE_INDEX.eye_wink_right] = 1;
    applyIdle(pose, 0.05, SEED);
    expect(pose[POSE_INDEX.eye_wink_left]).toBe(1);
  });
});
