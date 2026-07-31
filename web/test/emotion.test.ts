import { describe, expect, it } from "vitest";
import { POSE_INDEX, zeroPose } from "../src/pose/params.js";
import {
  EMOTION_NAMES,
  EMOTION_PRESETS,
  applyEmotion,
  blendEmotions,
  emotionAt,
} from "../src/emotion/emotion.js";

describe("emotion presets", () => {
  it("includes a neutral baseline plus the expressive set", () => {
    expect(EMOTION_NAMES).toContain("neutral");
    expect(EMOTION_NAMES).toContain("happy");
    expect(EMOTION_NAMES).toContain("sad");
    expect(EMOTION_NAMES).toContain("angry");
    expect(EMOTION_NAMES).toContain("surprised");
  });

  it("leaves neutral empty so it is a true no-op", () => {
    expect(Object.keys(EMOTION_PRESETS.neutral)).toHaveLength(0);
  });

  it("never touches mouth vowels or idle slots — those belong to lipsync/idle", () => {
    const forbidden = [
      "mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo", "mouth_delta",
      "head_x", "head_y", "neck_z", "body_y", "body_z", "breathing",
    ] as const;
    for (const name of EMOTION_NAMES) {
      for (const slot of forbidden) {
        expect(EMOTION_PRESETS[name], `${name} must not set ${slot}`).not.toHaveProperty(slot);
      }
    }
  });

  it("keeps every preset value inside 0..1", () => {
    for (const name of EMOTION_NAMES) {
      for (const [slot, value] of Object.entries(EMOTION_PRESETS[name])) {
        expect(value, `${name}.${slot}`).toBeGreaterThanOrEqual(0);
        expect(value, `${name}.${slot}`).toBeLessThanOrEqual(1);
      }
    }
  });

  it("uses the eyebrow group that matches each emotion's name", () => {
    expect(EMOTION_PRESETS.happy.eyebrow_happy_left).toBeGreaterThan(0);
    expect(EMOTION_PRESETS.angry.eyebrow_angry_left).toBeGreaterThan(0);
    expect(EMOTION_PRESETS.sad.eyebrow_troubled_left).toBeGreaterThan(0);
    expect(EMOTION_PRESETS.surprised.eyebrow_raised_left).toBeGreaterThan(0);
  });

  it("keeps left and right symmetric", () => {
    for (const name of EMOTION_NAMES) {
      for (const [slot, value] of Object.entries(EMOTION_PRESETS[name])) {
        if (!slot.endsWith("_left")) continue;
        const right = `${slot.slice(0, -5)}_right`;
        expect(EMOTION_PRESETS[name][right as never], `${name}.${right}`).toBe(value);
      }
    }
  });
});

describe("blendEmotions", () => {
  it("returns the single input unchanged at full weight", () => {
    const blended = blendEmotions([{ emotion: "happy", weight: 1 }]);
    expect(blended.eyebrow_happy_left).toBeCloseTo(EMOTION_PRESETS.happy.eyebrow_happy_left!, 6);
  });

  it("scales by weight", () => {
    const blended = blendEmotions([{ emotion: "happy", weight: 0.5 }]);
    expect(blended.eyebrow_happy_left).toBeCloseTo(EMOTION_PRESETS.happy.eyebrow_happy_left! / 2, 6);
  });

  it("sums overlapping emotions slot by slot", () => {
    const blended = blendEmotions([
      { emotion: "happy", weight: 0.5 },
      { emotion: "surprised", weight: 0.5 },
    ]);
    expect(blended.eyebrow_happy_left).toBeGreaterThan(0);
    expect(blended.eyebrow_raised_left).toBeGreaterThan(0);
  });

  it("is empty with no inputs", () => {
    expect(Object.keys(blendEmotions([]))).toHaveLength(0);
  });
});

describe("emotionAt", () => {
  const spans = [
    { start: 0, end: 2, emotion: "happy" as const },
    { start: 2, end: 4, emotion: "sad" as const },
  ];

  it("is fully inside the first span at its midpoint", () => {
    const p = emotionAt(spans, 1, { blendSeconds: 0.3 });
    expect(p.eyebrow_happy_left).toBeCloseTo(EMOTION_PRESETS.happy.eyebrow_happy_left!, 5);
    expect(p.eyebrow_troubled_left ?? 0).toBe(0);
  });

  it("crossfades across the boundary instead of cutting", () => {
    const p = emotionAt(spans, 2, { blendSeconds: 0.6 });
    expect(p.eyebrow_happy_left ?? 0).toBeGreaterThan(0);
    expect(p.eyebrow_troubled_left ?? 0).toBeGreaterThan(0);
  });

  it("is neutral before and after every span", () => {
    expect(Object.keys(emotionAt(spans, -1, { blendSeconds: 0.3 }))).toHaveLength(0);
    expect(Object.keys(emotionAt(spans, 99, { blendSeconds: 0.3 }))).toHaveLength(0);
  });

  it("honours per-span intensity", () => {
    const p = emotionAt([{ start: 0, end: 2, emotion: "happy", intensity: 0.25 }], 1, {
      blendSeconds: 0.1,
    });
    expect(p.eyebrow_happy_left).toBeCloseTo(EMOTION_PRESETS.happy.eyebrow_happy_left! * 0.25, 5);
  });
});

describe("applyEmotion", () => {
  it("writes preset slots without clobbering the mouth", () => {
    const pose = zeroPose();
    pose[POSE_INDEX.mouth_aaa] = 0.8;
    applyEmotion(pose, EMOTION_PRESETS.happy);
    expect(pose[POSE_INDEX.eyebrow_happy_left]).toBeGreaterThan(0);
    expect(pose[POSE_INDEX.mouth_aaa]).toBeCloseTo(0.8, 6);
  });

  it("clamps into the parameter's declared range", () => {
    const pose = zeroPose();
    applyEmotion(pose, { eyebrow_happy_left: 5 });
    expect(pose[POSE_INDEX.eyebrow_happy_left]).toBe(1);
  });
});
