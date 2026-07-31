import { describe, expect, it } from "vitest";
import { Spring, bodyMotionTrack, type BodyMotionInput } from "../src/motion/body.js";

const dt = 1 / 30;

describe("Spring", () => {
  it("settles on a constant target", () => {
    const s = new Spring(1.2, 0.9);
    let x = 0;
    for (let i = 0; i < 300; i++) x = s.step(1, dt);
    expect(x).toBeCloseTo(1, 3);
  });

  it("does not jump to a step change — that is the whole point", () => {
    const s = new Spring(1.2, 0.9);
    const first = s.step(1, dt);
    expect(first).toBeLessThan(0.15);
  });

  it("is stable under a violently alternating target", () => {
    const s = new Spring(1.2, 0.9);
    let x = 0;
    for (let i = 0; i < 600; i++) x = s.step(i % 2 === 0 ? 1 : -1, dt);
    expect(Number.isFinite(x)).toBe(true);
    expect(Math.abs(x)).toBeLessThan(2);
  });

  it("stays stable at a large timestep, where explicit integration explodes", () => {
    const s = new Spring(4, 0.9);
    let x = 0;
    for (let i = 0; i < 200; i++) x = s.step(1, 0.5);
    expect(Number.isFinite(x)).toBe(true);
    expect(x).toBeCloseTo(1, 2);
  });

  it("a lower frequency follows more slowly", () => {
    const slow = new Spring(0.4, 0.9);
    const fast = new Spring(3, 0.9);
    let a = 0;
    let b = 0;
    for (let i = 0; i < 10; i++) {
      a = slow.step(1, dt);
      b = fast.step(1, dt);
    }
    expect(a).toBeLessThan(b);
  });
});

describe("bodyMotionTrack", () => {
  /** A speech envelope with hard on/off bursts — the worst case for jerk. */
  function bursty(frames: number): BodyMotionInput[] {
    return Array.from({ length: frames }, (_, f) => {
      const speaking = Math.floor(f / 4) % 2 === 0;
      return {
        time: f / 30,
        seed: 20260731,
        speech: speaking ? 1 : 0,
        accent: f % 8 === 0 ? 1 : 0,
      };
    });
  }

  const channels = ["yaw", "pitch", "roll", "bodyYaw", "bodyRoll"] as const;

  function stats(track: ReturnType<typeof bodyMotionTrack>, key: (typeof channels)[number]) {
    const v = track.map((m) => m[key]);
    const d1 = v.slice(1).map((x, i) => x - v[i]!);
    const d2 = d1.slice(1).map((x, i) => x - d1[i]!);
    return {
      range: Math.max(...v) - Math.min(...v),
      maxAcc: Math.max(...d2.map(Math.abs)),
    };
  }

  it("removes the snap: acceleration stays an order of magnitude below the range", () => {
    // Before the spring, head_y had max|acc| 0.145 against a range of 0.320 —
    // nearly half the travel of the whole clip inside a single frame.
    const track = bodyMotionTrack(bursty(90), 30);
    for (const key of channels) {
      const { range, maxAcc } = stats(track, key);
      expect(maxAcc, `${key} acceleration`).toBeLessThan(0.012);
      if (range > 0.05) expect(maxAcc / range, `${key} acc/range`).toBeLessThan(0.1);
    }
  });

  it("still moves — smoothing must not flatten the performance", () => {
    const track = bodyMotionTrack(bursty(90), 30);
    // Horizontal is the expressive axis and has to carry the performance.
    expect(stats(track, "yaw").range).toBeGreaterThan(0.08);
    // Vertical deliberately stays quiet while speaking: looking up and down a
    // lot reads as restless, and the nods alone should not swing the head.
    expect(stats(track, "pitch").range).toBeGreaterThan(0.02);
    expect(stats(track, "pitch").range).toBeLessThan(0.12);
  });

  it("keeps everything in range", () => {
    const track = bodyMotionTrack(
      Array.from({ length: 300 }, (_, f) => ({
        time: f / 30,
        seed: 1,
        speech: 1,
        accent: 1,
        emotions: [{ emotion: "surprised" as const, weight: 1 }],
      })),
      30,
    );
    for (const m of track) {
      for (const key of channels) {
        expect(m[key]).toBeGreaterThanOrEqual(-1);
        expect(m[key]).toBeLessThanOrEqual(1);
      }
    }
  });

  it("is deterministic", () => {
    const input = bursty(60);
    expect(bodyMotionTrack(input, 30)).toEqual(bodyMotionTrack(input, 30));
  });

  it("returns nothing for no input", () => {
    expect(bodyMotionTrack([], 30)).toEqual([]);
  });

  it("responds to accents with a lag, not instantly", () => {
    // One accent starting at frame 10. It is *held and decayed* rather than a
    // single-frame spike, because that is what `speechDynamics` produces — a
    // bare one-frame impulse carries almost no energy and would be testing an
    // input the pipeline never generates.
    const input: BodyMotionInput[] = Array.from({ length: 60 }, (_, f) => ({
      time: f / 30,
      seed: 5,
      speech: 0,
      accent: f < 10 ? 0 : Math.max(0, 0.88 ** (f - 10)),
      swayScale: 0,
    }));
    const y = bodyMotionTrack(input, 30).map((m) => m.pitch);
    // Barely moved on the accent frame itself...
    expect(Math.abs(y[10]!)).toBeLessThan(0.02);
    // ...but clearly moved a few frames later.
    expect(Math.max(...y.slice(11, 25).map(Math.abs))).toBeGreaterThan(0.03);
  });
});
