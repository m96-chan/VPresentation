import { describe, expect, it } from "vitest";
import { POSE_INDEX, zeroPose } from "../src/pose/params.js";
import {
  POSTURES,
  thinkingGlanceAt,
  applyBodyMotion,
  bodyMotionAt,
  bodyMotionTrack,
  orientationAt,
  speechDynamics,
  swayAt,
} from "../src/motion/body.js";

const SEED = 20260731;
const still = { time: 0, seed: SEED, speech: 0, accent: 0 };

describe("swayAt", () => {
  it("stays inside the bipolar range", () => {
    for (let i = 0; i < 3000; i++) {
      const s = swayAt(i / 30, SEED);
      for (const v of [s.yaw, s.pitch, s.roll, s.bodyYaw, s.bodyRoll]) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("stays small — the large movement is orientation's job, not noise's", () => {
    // Big continuous noise reads as aimless floating. Sway is only the fine
    // wander layered on top of a held heading.
    for (let i = 0; i < 900; i++) {
      const s = swayAt(i / 30, SEED);
      expect(Math.abs(s.yaw)).toBeLessThan(0.2);
      expect(Math.abs(s.bodyYaw)).toBeLessThan(0.12);
    }
  });

  it("is still smooth frame to frame", () => {
    let prev = swayAt(0, SEED);
    for (let i = 1; i < 3000; i++) {
      const cur = swayAt(i / 30, SEED);
      expect(Math.abs(cur.yaw - prev.yaw)).toBeLessThan(0.06);
      expect(Math.abs(cur.bodyRoll - prev.bodyRoll)).toBeLessThan(0.06);
      prev = cur;
    }
  });

  it("is deterministic and seed-dependent", () => {
    expect(swayAt(2.5, SEED)).toEqual(swayAt(2.5, SEED));
    expect(swayAt(2.5, 1)).not.toEqual(swayAt(2.5, 2));
  });
});

describe("speechDynamics", () => {
  it("returns a level and an accent per frame", () => {
    const openness = [0, 0, 1, 1, 0, 0];
    const d = speechDynamics(openness, 30);
    expect(d).toHaveLength(6);
    expect(d[0]!.level).toBe(0);
    expect(d[2]!.accent).toBeGreaterThan(0);
  });

  it("has no accent during sustained loudness — only on the rise", () => {
    const sustained = speechDynamics(new Array(120).fill(1), 30);
    expect(sustained[2]!.accent).toBeGreaterThan(sustained[119]!.accent);
  });

  it("is silent for silence", () => {
    for (const d of speechDynamics(new Array(60).fill(0), 30)) {
      expect(d.level).toBe(0);
      expect(d.accent).toBe(0);
    }
  });

  it("handles an empty track", () => {
    expect(speechDynamics([], 30)).toEqual([]);
  });
});

describe("bodyMotionAt", () => {
  it("moves the head when speech accents, and not when silent", () => {
    const silent = bodyMotionAt({ ...still, time: 1 });
    const accented = bodyMotionAt({ ...still, time: 1, speech: 0.9, accent: 0.8 });
    expect(Math.abs(accented.pitch - silent.pitch)).toBeGreaterThan(0.05);
  });

  it("leans with sustained speech level", () => {
    const quiet = bodyMotionAt({ ...still, time: 1, speech: 0 });
    const loud = bodyMotionAt({ ...still, time: 1, speech: 1 });
    expect(Math.abs(loud.bodyYaw - quiet.bodyYaw)).toBeGreaterThan(0.02);
  });

  it("keeps everything inside the bipolar range even at full drive", () => {
    for (let i = 0; i < 900; i++) {
      const m = bodyMotionAt({
        time: i / 30,
        seed: SEED,
        speech: 1,
        accent: 1,
        emotions: [{ emotion: "surprised", weight: 1 }],
      });
      for (const v of [m.yaw, m.pitch, m.roll, m.bodyYaw, m.bodyRoll]) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("applies emotion posture", () => {
    const neutral = bodyMotionAt({ ...still, time: 1 });
    const surprised = bodyMotionAt({
      ...still,
      time: 1,
      emotions: [{ emotion: "surprised", weight: 1 }],
    });
    expect(surprised.pitch).not.toBeCloseTo(neutral.pitch, 3);
  });

  it("scales posture by emotion weight", () => {
    const full = bodyMotionAt({ ...still, emotions: [{ emotion: "sad", weight: 1 }] });
    const half = bodyMotionAt({ ...still, emotions: [{ emotion: "sad", weight: 0.5 }] });
    expect(Math.abs(half.pitch - still.speech)).toBeLessThan(Math.abs(full.pitch));
  });

  it("can have its layers disabled independently", () => {
    const none = bodyMotionAt({
      ...still,
      time: 3,
      speech: 1,
      accent: 1,
      swayScale: 0,
      gestureScale: 0,
      turnScale: 0,
    });
    for (const v of [none.yaw, none.pitch, none.roll, none.bodyYaw, none.bodyRoll]) {
      expect(v).toBeCloseTo(0, 10); // may be -0, which is still zero
    }
  });

  it("defines a posture for every emotion it claims to support", () => {
    for (const posture of Object.values(POSTURES)) {
      for (const v of Object.values(posture)) {
        expect(Math.abs(v as number)).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("applyBodyMotion", () => {
  it("writes only the five body slots", () => {
    const pose = zeroPose();
    pose[POSE_INDEX.mouth_aaa] = 0.7;
    pose[POSE_INDEX.breathing] = 0.5;
    applyBodyMotion(pose, { ...still, time: 2, speech: 0.8, accent: 0.5 });

    expect(pose[POSE_INDEX.mouth_aaa]).toBeCloseTo(0.7, 6);
    expect(pose[POSE_INDEX.breathing]).toBeCloseTo(0.5, 6);
    expect(pose[POSE_INDEX.head_y]).not.toBe(0);
  });
});

describe("orientationAt", () => {
  it("holds a heading rather than drifting continuously", () => {
    // Sampled densely, the value should be piecewise constant: a real head
    // turn is "move, then stay", not a permanent wander.
    const values = Array.from({ length: 300 }, (_, i) => orientationAt(i / 30, SEED));
    const changes = values.filter((v, i) => i > 0 && v !== values[i - 1]).length;
    expect(changes).toBeGreaterThan(1); // it does change
    expect(changes).toBeLessThan(10); // but only a handful of times in 10s
  });

  it("stays within -1..1", () => {
    for (let i = 0; i < 3000; i++) {
      const v = orientationAt(i / 30, SEED);
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("looks both ways over time", () => {
    const values = Array.from({ length: 1800 }, (_, i) => orientationAt(i / 30, SEED));
    expect(Math.min(...values)).toBeLessThan(-0.15);
    expect(Math.max(...values)).toBeGreaterThan(0.15);
  });

  it("is deterministic and seed-dependent", () => {
    expect(orientationAt(4.2, SEED)).toBe(orientationAt(4.2, SEED));
    const a = Array.from({ length: 300 }, (_, i) => orientationAt(i / 30, 1));
    const b = Array.from({ length: 300 }, (_, i) => orientationAt(i / 30, 2));
    expect(a).not.toEqual(b);
  });
});

describe("turning left and right", () => {
  function track(seconds: number, seed = SEED) {
    return bodyMotionTrack(
      Array.from({ length: seconds * 30 }, (_, f) => ({
        time: f / 30,
        seed,
        speech: 0.4,
        accent: 0,
      })),
      30,
    );
  }

  it("uses a serious share of head_x's range, not 15% of it", () => {
    const xs = track(12).map((m) => m.yaw);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(0.45);
  });

  it("turns the torso with the head", () => {
    const t = track(12);
    const ys = t.map((m) => m.bodyYaw);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(0.2);
  });

  it("keeps the turn smooth despite the heading being a step function", () => {
    const t = track(12);
    const xs = t.map((m) => m.yaw);
    const d1 = xs.slice(1).map((x, i) => x - xs[i]!);
    const d2 = d1.slice(1).map((x, i) => x - d1[i]!);
    expect(Math.max(...d2.map(Math.abs))).toBeLessThan(0.02);
  });

  it("lets the eyes arrive before the head", () => {
    // On a heading change the iris, being far lighter, should lead.
    const t = track(12);
    let leadFrames = 0;
    for (let i = 1; i < t.length; i++) {
      if (Math.abs(t[i]!.gazeYaw) > Math.abs(t[i]!.yaw)) leadFrames++;
    }
    expect(leadFrames).toBeGreaterThan(t.length * 0.5);
  });
});

describe("vertical balance", () => {
  /** Continuous speech, the case where a downward DC offset accumulates. */
  const talking = Array.from({ length: 360 }, (_, f) => ({
    time: f / 30,
    seed: SEED,
    speech: 0.7,
    accent: Math.sin(f * 0.8) * 0.5,
    silence: 0,
  }));

  it("does not hold the head down through an utterance", () => {
    // The first version clamped `accent` to >= 0, so every syllable pushed the
    // head down and nothing ever pushed it back up.
    const ys = bodyMotionTrack(talking, 30).map((m) => m.pitch);
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
  });

  it("looks down no more than it looks up", () => {
    const ys = bodyMotionTrack(talking, 30).map((m) => m.pitch);
    const down = Math.abs(Math.min(...ys));
    const up = Math.max(...ys);
    expect(down).toBeLessThan(up * 2.2);
  });

  it("keeps vertical travel well under horizontal", () => {
    const t = bodyMotionTrack(
      Array.from({ length: 360 }, (_, f) => ({
        time: f / 30,
        seed: SEED,
        speech: 0.7,
        accent: 0,
        silence: 0,
      })),
      30,
    );
    const span = (k: "yaw" | "pitch") => {
      const v = t.map((m) => m[k]);
      return Math.max(...v) - Math.min(...v);
    };
    expect(span("pitch")).toBeLessThan(span("yaw") * 0.5);
  });
});

describe("thinking gaze", () => {
  /**
   * Walk a long silence and keep the frames where a glance is engaged.
   *
   * The schedule is a function of *how long the silence has run*, not of the
   * clock, so this advances `silence` — holding it fixed and advancing `time`
   * samples the same instant of the same glance over and over.
   */
  function glances(bias = 0) {
    const out: ReturnType<typeof bodyMotionAt>[] = [];
    for (let f = 0; f < 180 * 30; f++) {
      const m = bodyMotionAt({
        time: f / 30,
        seed: SEED,
        speech: 0,
        accent: 0,
        silence: f / 30,
        swayScale: 0,
        headingBias: bias,
      });
      if (Math.abs(m.gazePitch) > 0.25) out.push(m);
    }
    return out;
  }

  it("does nothing during a short gap between words", () => {
    const m = bodyMotionAt({
      time: 4,
      seed: SEED,
      speech: 0,
      accent: 0,
      silence: 0.2,
      swayScale: 0,
      turnScale: 0,
    });
    expect(Math.abs(m.gazePitch)).toBeLessThan(0.02);
  });

  it("drifts the gaze up during a pause", () => {
    const engaged = glances();
    expect(engaged.length).toBeGreaterThan(0);
    // Chin up and gaze up: same sign, both positive.
    expect(Math.max(...engaged.map((m) => m.pitch))).toBeGreaterThan(0.1);
    expect(Math.max(...engaged.map((m) => m.gazePitch))).toBeGreaterThan(0.25);
  });

  it("moves the eyes further than the head, as a real glance does", () => {
    // Isolated from the facing term: with a heading in play the two are
    // independent contributions whose sum can land either way, and the
    // comparison stops meaning anything.
    let checked = 0;
    for (let f = 0; f < 180 * 30; f++) {
      const m = bodyMotionAt({
        time: f / 30,
        seed: SEED,
        speech: 0,
        accent: 0,
        silence: f / 30,
        swayScale: 0,
        turnScale: 0,
      });
      if (Math.abs(m.gazePitch) < 0.25) continue;
      checked++;
      expect(Math.abs(m.gazeYaw)).toBeGreaterThan(Math.abs(m.yaw));
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("keeps one direction for a whole glance, mostly to the left", () => {
    // Counted once per glance, not per frame: the 72/28 split is a property of
    // the schedule, and glances vary in length, so frame-weighting measures
    // duration rather than the choice.
    const sides: number[] = [];
    let current: number | null = null;
    for (let f = 0; f < 600 * 30; f++) {
      const g = thinkingGlanceAt(SEED, f / 30, 1);
      if (g.amount <= 0) {
        current = null;
        continue;
      }
      if (current === null) {
        current = g.side;
        sides.push(g.side);
      } else {
        // A glance that flipped halfway through would read as a flinch.
        expect(g.side, `glance ${sides.length} changed direction`).toBe(current);
      }
    }
    expect(sides.length).toBeGreaterThan(30);
    // `preferred` is +1 here; positive is the viewer's right, so the intent is
    // simply that it follows the preference most of the time.
    expect(sides.filter((s) => s === 1).length / sides.length).toBeGreaterThan(0.6);
  });
});

describe("thinking gaze and heading bias together", () => {
  function idle(bias: number, seconds = 30) {
    return Array.from({ length: seconds * 30 }, (_, f) =>
      bodyMotionAt({
        time: f / 30,
        seed: SEED,
        speech: 0,
        accent: 0,
        // Nothing is being said, so the thinking gaze is fully engaged.
        silence: 5,
        swayScale: 0,
        headingBias: bias,
      }).yaw,
    );
  }

  it("does not let the thinking gaze cancel the lean towards the deck", () => {
    // With a fixed thinking direction, a left-corner presenter got +0.26 of
    // "thinking left" against -0.28 of bias and stared straight ahead.
    const right = idle(0.45);
    const left = idle(-0.45);
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    expect(mean(right)).toBeGreaterThan(0.15);
    expect(mean(left)).toBeLessThan(-0.15);
  });

  it("keeps both corners symmetric", () => {
    // Averaged over four minutes: a 30 s window only contains about ten held
    // headings, so the generator's own finite-sample skew dominates and this
    // measures noise rather than symmetry.
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    expect(Math.abs(mean(idle(0.45, 240)) + mean(idle(-0.45, 240)))).toBeLessThan(0.1);
  });
});

describe("thinking glances", () => {
  function gaze(seconds: number, seed = SEED) {
    return Array.from({ length: seconds * 30 }, (_, f) =>
      bodyMotionAt({
        time: f / 30,
        seed,
        speech: 0,
        accent: 0,
        // Nothing is ever said: the worst case for a latching thinking pose.
        silence: f / 30,
        swayScale: 0,
        headingBias: -0.45,
      }),
    );
  }

  it("does not hold the glance for the whole silence", () => {
    // Ramping in with no decay left the character staring up and away with its
    // eyes pegged for as long as nothing was said, which reads as vacant
    // rather than thoughtful.
    const engaged = gaze(120).filter((m) => Math.abs(m.gazePitch) > 0.25).length;
    expect(engaged / (120 * 30)).toBeLessThan(0.5);
  });

  it("still glances sometimes", () => {
    const engaged = gaze(120).filter((m) => Math.abs(m.gazePitch) > 0.25).length;
    expect(engaged / (120 * 30)).toBeGreaterThan(0.05);
  });

  it("never pegs the eyes at the limit", () => {
    // A gaze sitting on ±1 looks strained, and it also means the value has
    // saturated, so nothing downstream can read it.
    const pegged = gaze(120).filter((m) => Math.abs(m.gazeYaw) > 0.97).length;
    expect(pegged).toBe(0);
  });

  it("eases in and out rather than switching", () => {
    const g = gaze(120).map((m) => m.gazePitch);
    for (let i = 1; i < g.length; i++) {
      expect(Math.abs(g[i]! - g[i - 1]!), `frame ${i}`).toBeLessThan(0.05);
    }
  });

  it("is deterministic", () => {
    expect(gaze(20).map((m) => m.gazeYaw)).toEqual(gaze(20).map((m) => m.gazeYaw));
  });
});
