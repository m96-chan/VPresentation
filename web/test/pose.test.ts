import { describe, expect, it } from "vitest";
import {
  FACE_POSE_LENGTH,
  NUM_POSE_PARAMS,
  POSE_GROUPS,
  POSE_INDEX,
  POSE_PARAM_NAMES,
  facePose,
  restPose,
  zeroPose,
} from "../src/pose/params.js";

// The layout is the contract with THA4. Ground truth:
//   third_party/tha4_src/src/tha4/poser/modes/pose_parameters.py
//   crates/tha4/src/poser.rs        (NUM_POSE_PARAMS = 45, face = pose[0..39])
describe("pose parameter layout", () => {
  it("has 45 parameters, matching NUM_POSE_PARAMS in poser.rs", () => {
    expect(NUM_POSE_PARAMS).toBe(45);
    expect(POSE_PARAM_NAMES).toHaveLength(45);
  });

  it("splits face/body at 39, matching StudentPoser::pose", () => {
    expect(FACE_POSE_LENGTH).toBe(39);
  });

  it("orders groups exactly as pose_parameters.py builds them", () => {
    expect(POSE_GROUPS.map((g) => g.name)).toEqual([
      "eyebrow_troubled",
      "eyebrow_angry",
      "eyebrow_lowered",
      "eyebrow_raised",
      "eyebrow_happy",
      "eyebrow_serious",
      "eye_wink",
      "eye_happy_wink",
      "eye_surprised",
      "eye_relaxed",
      "eye_unimpressed",
      "eye_raised_lower_eyelid",
      "iris_small",
      "mouth_aaa",
      "mouth_iii",
      "mouth_uuu",
      "mouth_eee",
      "mouth_ooo",
      "mouth_delta",
      "mouth_lowered_corner",
      "mouth_raised_corner",
      "mouth_smirk",
      "iris_rotation_x",
      "iris_rotation_y",
      "head_x",
      "head_y",
      "neck_z",
      "body_y",
      "body_z",
      "breathing",
    ]);
  });

  it("places the arity-2 groups as <name>_left then <name>_right", () => {
    expect(POSE_INDEX.eyebrow_troubled_left).toBe(0);
    expect(POSE_INDEX.eyebrow_troubled_right).toBe(1);
    expect(POSE_INDEX.eye_wink_left).toBe(12);
    expect(POSE_INDEX.eye_wink_right).toBe(13);
    expect(POSE_INDEX.iris_small_left).toBe(24);
    expect(POSE_INDEX.iris_small_right).toBe(25);
  });

  it("places the five vowels contiguously at 26..30", () => {
    expect(POSE_INDEX.mouth_aaa).toBe(26);
    expect(POSE_INDEX.mouth_iii).toBe(27);
    expect(POSE_INDEX.mouth_uuu).toBe(28);
    expect(POSE_INDEX.mouth_eee).toBe(29);
    expect(POSE_INDEX.mouth_ooo).toBe(30);
    expect(POSE_INDEX.mouth_delta).toBe(31);
  });

  it("puts the last face parameter at 38 and the first body one at 39", () => {
    expect(POSE_INDEX.iris_rotation_y).toBe(38);
    expect(POSE_INDEX.head_x).toBe(39);
    expect(POSE_INDEX.breathing).toBe(44);
  });

  it("keeps every index unique and gap-free", () => {
    const indices = POSE_PARAM_NAMES.map((n) => POSE_INDEX[n]);
    expect(new Set(indices).size).toBe(45);
    expect([...indices].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 45 }, (_, i) => i),
    );
  });
});

describe("pose construction", () => {
  it("zeroPose is all zeros", () => {
    const p = zeroPose();
    expect(p).toHaveLength(45);
    expect([...p].every((v) => v === 0)).toBe(true);
  });

  it("restPose closes the mouth, unlike the THA4 slider default", () => {
    // pose_parameters.py gives mouth_aaa default_value=1.0, which is a *GUI
    // slider* default. A resting character must have a closed mouth.
    expect(restPose()[POSE_INDEX.mouth_aaa]).toBe(0);
  });

  it("facePose is the first 39 entries", () => {
    const p = zeroPose();
    p[POSE_INDEX.mouth_aaa] = 0.5;
    p[POSE_INDEX.head_x] = 0.9;
    const f = facePose(p);
    expect(f).toHaveLength(39);
    expect(f[POSE_INDEX.mouth_aaa]).toBe(0.5);
    expect([...f]).not.toContain(0.9);
  });
});
