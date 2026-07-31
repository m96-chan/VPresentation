/**
 * THA4 pose vector layout.
 *
 * This is a direct transcription of THA4's own parameter builder:
 *   third_party/tha4_src/src/tha4/poser/modes/pose_parameters.py
 *
 * Groups are laid out in declaration order; a group with `arity: 2` occupies
 * two slots, left then right. The result is 45 floats, of which the first 39
 * feed the face morpher and all 45 feed the body morpher — matching
 * `StudentPoser::pose` in crates/tha4/src/poser.rs.
 */

export type PoseRange = readonly [min: number, max: number];

export interface PoseGroup {
  readonly name: string;
  readonly arity: 1 | 2;
  /** Valid range for each slot in the group. */
  readonly range: PoseRange;
}

const UNIT: PoseRange = [0, 1];
const BIPOLAR: PoseRange = [-1, 1];

export const POSE_GROUPS = [
  { name: "eyebrow_troubled", arity: 2, range: UNIT },
  { name: "eyebrow_angry", arity: 2, range: UNIT },
  { name: "eyebrow_lowered", arity: 2, range: UNIT },
  { name: "eyebrow_raised", arity: 2, range: UNIT },
  { name: "eyebrow_happy", arity: 2, range: UNIT },
  { name: "eyebrow_serious", arity: 2, range: UNIT },
  { name: "eye_wink", arity: 2, range: UNIT },
  { name: "eye_happy_wink", arity: 2, range: UNIT },
  { name: "eye_surprised", arity: 2, range: UNIT },
  { name: "eye_relaxed", arity: 2, range: UNIT },
  { name: "eye_unimpressed", arity: 2, range: UNIT },
  { name: "eye_raised_lower_eyelid", arity: 2, range: UNIT },
  { name: "iris_small", arity: 2, range: UNIT },
  { name: "mouth_aaa", arity: 1, range: UNIT },
  { name: "mouth_iii", arity: 1, range: UNIT },
  { name: "mouth_uuu", arity: 1, range: UNIT },
  { name: "mouth_eee", arity: 1, range: UNIT },
  { name: "mouth_ooo", arity: 1, range: UNIT },
  { name: "mouth_delta", arity: 1, range: UNIT },
  { name: "mouth_lowered_corner", arity: 2, range: UNIT },
  { name: "mouth_raised_corner", arity: 2, range: UNIT },
  { name: "mouth_smirk", arity: 1, range: UNIT },
  { name: "iris_rotation_x", arity: 1, range: BIPOLAR },
  { name: "iris_rotation_y", arity: 1, range: BIPOLAR },
  { name: "head_x", arity: 1, range: BIPOLAR },
  { name: "head_y", arity: 1, range: BIPOLAR },
  { name: "neck_z", arity: 1, range: BIPOLAR },
  { name: "body_y", arity: 1, range: BIPOLAR },
  { name: "body_z", arity: 1, range: BIPOLAR },
  { name: "breathing", arity: 1, range: UNIT },
] as const satisfies readonly PoseGroup[];

type Group = (typeof POSE_GROUPS)[number];

type ExpandGroup<G extends PoseGroup> = G extends { name: infer N extends string }
  ? G extends { arity: 2 }
    ? `${N}_left` | `${N}_right`
    : N
  : never;

/** Every individual pose slot name, e.g. `"eye_wink_left"` or `"mouth_aaa"`. */
export type PoseParamName = ExpandGroup<Group>;

function expand(group: PoseGroup): string[] {
  return group.arity === 2
    ? [`${group.name}_left`, `${group.name}_right`]
    : [group.name];
}

export const POSE_PARAM_NAMES = POSE_GROUPS.flatMap(expand) as PoseParamName[];

export const NUM_POSE_PARAMS = POSE_PARAM_NAMES.length;

/** The face morpher takes `pose[0..39]`; the body morpher takes all 45. */
export const FACE_POSE_LENGTH = 39;

export const POSE_INDEX = Object.freeze(
  Object.fromEntries(POSE_PARAM_NAMES.map((name, i) => [name, i])),
) as Readonly<Record<PoseParamName, number>>;

/** Per-slot valid range, in the same order as {@link POSE_PARAM_NAMES}. */
export const POSE_RANGES: readonly PoseRange[] = POSE_GROUPS.flatMap((g) =>
  expand(g).map(() => g.range),
);

/** A 45-float pose vector. */
export type Pose = Float32Array;

export function zeroPose(): Pose {
  return new Float32Array(NUM_POSE_PARAMS);
}

/**
 * A neutral resting pose.
 *
 * Note this is *not* THA4's slider defaults: `pose_parameters.py` gives
 * `mouth_aaa` a `default_value` of 1.0, which is a GUI convenience (so the
 * mouth slider starts open) and would leave an idle character gaping.
 */
export function restPose(): Pose {
  return zeroPose();
}

/** The first 39 entries — the face morpher's input. */
export function facePose(pose: Pose): Float32Array {
  return pose.subarray(0, FACE_POSE_LENGTH);
}

/** Clamp every slot into its declared range, in place. */
export function clampPose(pose: Pose): Pose {
  for (let i = 0; i < pose.length; i++) {
    const range = POSE_RANGES[i];
    if (!range) continue;
    const v = pose[i] ?? 0;
    pose[i] = v < range[0] ? range[0] : v > range[1] ? range[1] : v;
  }
  return pose;
}
