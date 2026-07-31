/**
 * Emotion -> pose parameters.
 *
 * Emotion owns the eyebrow (0..11), eye (12..23), iris_small (24..25) and
 * mouth-corner (32..36) slots. It deliberately never writes the mouth vowels
 * or the idle slots, so lipsync and idle motion can be layered on top without
 * fighting over the same floats.
 */
import { POSE_INDEX, POSE_RANGES, type Pose, type PoseParamName } from "../pose/params.js";

export type PosePartial = Partial<Record<PoseParamName, number>>;

/** Expand a symmetric pair so presets stay readable. */
function pair(base: string, value: number): PosePartial {
  return { [`${base}_left`]: value, [`${base}_right`]: value } as PosePartial;
}

export const EMOTION_PRESETS = {
  neutral: {},

  happy: {
    ...pair("eyebrow_happy", 0.75),
    ...pair("eye_happy_wink", 0.35),
    ...pair("mouth_raised_corner", 0.7),
  },

  sad: {
    ...pair("eyebrow_troubled", 0.85),
    ...pair("eye_relaxed", 0.4),
    ...pair("mouth_lowered_corner", 0.6),
  },

  angry: {
    ...pair("eyebrow_angry", 0.9),
    ...pair("eye_raised_lower_eyelid", 0.45),
    ...pair("mouth_lowered_corner", 0.35),
  },

  surprised: {
    ...pair("eyebrow_raised", 0.9),
    ...pair("eye_surprised", 0.8),
    ...pair("iris_small", 0.5),
  },

  serious: {
    ...pair("eyebrow_serious", 0.7),
    ...pair("eye_raised_lower_eyelid", 0.25),
  },

  troubled: {
    ...pair("eyebrow_troubled", 0.6),
    ...pair("eye_unimpressed", 0.4),
    ...pair("mouth_lowered_corner", 0.25),
  },
} as const satisfies Record<string, PosePartial>;

export type EmotionName = keyof typeof EMOTION_PRESETS;

export const EMOTION_NAMES = Object.keys(EMOTION_PRESETS) as EmotionName[];

export interface WeightedEmotion {
  readonly emotion: EmotionName;
  readonly weight: number;
}

/** Sum a set of weighted presets slot by slot. */
export function blendEmotions(inputs: readonly WeightedEmotion[]): PosePartial {
  const out: PosePartial = {};
  for (const { emotion, weight } of inputs) {
    if (weight <= 0) continue;
    for (const [slot, value] of Object.entries(EMOTION_PRESETS[emotion])) {
      const key = slot as PoseParamName;
      out[key] = (out[key] ?? 0) + (value as number) * weight;
    }
  }
  return out;
}

export interface EmotionSpan {
  /** Seconds. */
  readonly start: number;
  readonly end: number;
  readonly emotion: EmotionName;
  /** 0..1 scale on the preset. Defaults to 1. */
  readonly intensity?: number;
}

export interface EmotionTimingOptions {
  /** Crossfade length at each span edge, in seconds. */
  readonly blendSeconds?: number;
}

/**
 * Weight of `span` at `time`: 1 in the middle, ramping linearly to 0 across
 * `blendSeconds` at each edge. Outside the span (plus its fade) it is 0.
 */
function spanWeight(span: EmotionSpan, time: number, blend: number): number {
  const half = blend / 2;
  const fadeIn = span.start - half;
  const fadeOut = span.end + half;
  if (time <= fadeIn || time >= fadeOut) return 0;
  if (blend <= 0) return 1;

  const rising = (time - fadeIn) / blend;
  const falling = (fadeOut - time) / blend;
  return Math.max(0, Math.min(1, Math.min(rising, falling)));
}

/**
 * Which emotions are in effect at `time`, and how strongly.
 *
 * Exposed separately from {@link emotionAt} because head/body posture needs
 * the emotion *names*, which a blended pose has already thrown away.
 */
export function activeEmotions(
  spans: readonly EmotionSpan[],
  time: number,
  options: EmotionTimingOptions = {},
): WeightedEmotion[] {
  const blend = options.blendSeconds ?? 0.35;
  const active: WeightedEmotion[] = [];
  for (const span of spans) {
    const w = spanWeight(span, time, blend) * (span.intensity ?? 1);
    if (w > 0) active.push({ emotion: span.emotion, weight: w });
  }
  return active;
}

/** The blended emotion pose at `time`. */
export function emotionAt(
  spans: readonly EmotionSpan[],
  time: number,
  options: EmotionTimingOptions = {},
): PosePartial {
  return blendEmotions(activeEmotions(spans, time, options));
}

/** Write a partial pose into `pose`, clamped to each slot's declared range. */
export function applyEmotion(pose: Pose, partial: PosePartial): Pose {
  for (const [slot, value] of Object.entries(partial)) {
    const index = POSE_INDEX[slot as PoseParamName];
    if (index === undefined) continue;
    const range = POSE_RANGES[index];
    const v = value as number;
    pose[index] = range ? Math.min(range[1], Math.max(range[0], v)) : v;
  }
  return pose;
}
