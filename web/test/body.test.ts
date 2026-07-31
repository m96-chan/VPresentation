import { describe, expect, it } from "vitest";
import { POSE_INDEX, zeroPose } from "../src/pose/params.js";
import {
  POSTURES,
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
  function paused(silenceSeconds: number) {
    return bodyMotionAt({
      time: 4,
      seed: SEED,
      speech: 0,
      accent: 0,
      silence: silenceSeconds,
      swayScale: 0,
      turnScale: 0,
    });
  }

  it("does nothing during a short gap between words", () => {
    expect(Math.abs(paused(0.2).gazePitch)).toBeLessThan(0.02);
  });

  it("drifts the gaze up during a real pause", () => {
    const m = paused(2);
    expect(m.pitch).toBeGreaterThan(0.1); // chin up
    expect(m.gazePitch).toBeGreaterThan(0.2); // gaze up, same sign as pitch
  });

  it("puts the eyes ahead of the head, as a real glance does", () => {
    const m = paused(2);
    expect(Math.abs(m.gazeYaw)).toBeGreaterThan(Math.abs(m.yaw));
  });

  it("mostly goes to the character's upper-left", () => {
    // Sampled wide on purpose. At the intended 72/28 split, 40 draws have a
    // standard deviation of ~4.5 points, which is enough for an unlucky window
    // to dip under 60% and make this flaky.
    const SAMPLES = 400;
    let left = 0;
    for (let t = 0; t < SAMPLES; t++) {
      const m = bodyMotionAt({
        time: t * 3.5,
        seed: SEED,
        speech: 0,
        accent: 0,
        silence: 2,
        swayScale: 0,
        turnScale: 0,
      });
      // Gaze yaw shares the head's sign: positive is the viewer's left.
      if (m.gazeYaw > 0) left++;
    }
    expect(left / SAMPLES).toBeGreaterThan(0.6);
    expect(left / SAMPLES).toBeLessThan(0.85);
  });
});

describe("heading bias", () => {
  // A presenter standing in the bottom-right corner has the deck on their
  // left, and POSITIVE head_x is the viewer's left. Turning the other way
  // during idle points them out of frame, away from the content.
  function headings(bias: number, seconds = 40) {
    return Array.from({ length: seconds * 30 }, (_, f) =>
      bodyMotionAt({
        time: f / 30,
        seed: SEED,
        speech: 0,
        accent: 0,
        swayScale: 0,
        headingBias: bias,
      }).yaw,
    );
  }

  it("leans the whole idle range towards the bias", () => {
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    expect(mean(headings(-0.4))).toBeLessThan(mean(headings(0)));
    expect(mean(headings(0.4))).toBeGreaterThan(mean(headings(0)));
  });

  it("never crosses to the far side of the bias", () => {
    // Constructive, not statistical: the wander is scaled to fit inside the
    // biased half, because adding a bias to a zero-centred range still crossed
    // over whenever the generator's own sample skewed that way.
    expect(headings(0.45).every((v) => v >= -0.02)).toBe(true);
    expect(headings(-0.45).every((v) => v <= 0.02)).toBe(true);
  });

  it("still looks around rather than staring one way", () => {
    const v = headings(-0.4);
    expect(Math.max(...v) - Math.min(...v)).toBeGreaterThan(0.25);
  });

  it("stays in range at full bias", () => {
    for (const v of headings(1)) {
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("points the eyes the same way the head is turned", () => {
    // THA4's mocap converters drive head_y and iris_rotation_y from the same
    // physical direction, so gaze yaw and head yaw share a sign. The long
    // detour here came from reading `head_x` as "horizontal" when it is the
    // rotation *axis* — pitch — so the turn was being driven into the nod.
    for (const bias of [-0.5, 0.5]) {
      const m = bodyMotionAt({
        time: 5,
        seed: SEED,
        speech: 0,
        accent: 0,
        swayScale: 0,
        headingBias: bias,
      });
      expect(Math.sign(m.gazeYaw), `bias ${bias}`).toBe(Math.sign(m.yaw));
    }
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
