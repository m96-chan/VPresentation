/**
 * Idle motion — the signals that keep a character alive while nothing else is
 * driving it: blinking, breathing, and slow head/body sway.
 *
 * Everything here is a pure function of `(time, seed)`. That is deliberate:
 * the realtime renderer and the offline renderer must produce identical
 * frames, which rules out `Math.random()` and any accumulated internal state.
 */
import { POSE_INDEX, type Pose } from "../pose/params.js";

/** Integer hash -> [0, 1). Deterministic across platforms. */
function hash01(n: number, seed: number): number {
  let h = (n | 0) ^ Math.imul(seed | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Smooth value noise in [-1, 1] with lattice points once per `1/frequency`. */
function valueNoise(time: number, frequency: number, seed: number): number {
  const x = time * frequency;
  const i = Math.floor(x);
  const f = x - i;
  const a = hash01(i, seed) * 2 - 1;
  const b = hash01(i + 1, seed) * 2 - 1;
  return a + (b - a) * smoothstep(f);
}

// --- blink -----------------------------------------------------------------

const BLINK_MIN_GAP = 1.8;
const BLINK_MAX_GAP = 5.5;
/** A human blink is ~100-150 ms closed, plus the ramps either side. */
const BLINK_DURATION = 0.16;

/**
 * Blink openness at `time`: 0 = eyes open, 1 = fully shut.
 *
 * The schedule is a deterministic walk — blink `k` starts at the sum of the
 * first `k` hashed intervals — so any time can be evaluated without replaying
 * animation state.
 */
export function blinkAmount(time: number, seed: number): number {
  if (time < 0) return 0;

  let start = hash01(0, seed) * BLINK_MAX_GAP;
  for (let k = 1; start <= time + BLINK_DURATION; k++) {
    if (start + BLINK_DURATION > time && start <= time) {
      // Inside this blink: a raised-cosine close/open cycle.
      const phase = (time - start) / BLINK_DURATION;
      return 0.5 - 0.5 * Math.cos(2 * Math.PI * phase);
    }
    const gap = BLINK_MIN_GAP + hash01(k, seed) * (BLINK_MAX_GAP - BLINK_MIN_GAP);
    start += BLINK_DURATION + gap;
  }
  return 0;
}

// --- breathing -------------------------------------------------------------

/** Seconds per breath cycle. */
export const BREATH_PERIOD = 4;

/** Breathing in 0..1, one full cycle every {@link BREATH_PERIOD} seconds. */
export function breathingAmount(time: number): number {
  return 0.5 - 0.5 * Math.cos((2 * Math.PI * time) / BREATH_PERIOD);
}

// --- application -----------------------------------------------------------

export interface IdleOptions {
  readonly seed?: number;
  readonly blink?: boolean;
  readonly breathing?: boolean;
}

/**
 * Layer idle motion onto `pose`.
 *
 * Blink and breathing only — head and body live in `motion/body.ts`, which
 * needs the speech envelope and so cannot be a function of time alone.
 *
 * Blink is applied with `max` rather than assignment so a deliberate wink
 * coming from an emotion preset is never reopened by the idle loop.
 */
export function applyIdle(
  pose: Pose,
  time: number,
  seed = 0,
  options: IdleOptions = {},
): Pose {
  if (options.blink !== false) {
    const blink = blinkAmount(time, seed);
    pose[POSE_INDEX.eye_wink_left] = Math.max(pose[POSE_INDEX.eye_wink_left] ?? 0, blink);
    pose[POSE_INDEX.eye_wink_right] = Math.max(pose[POSE_INDEX.eye_wink_right] ?? 0, blink);
  }

  if (options.breathing !== false) {
    pose[POSE_INDEX.breathing] = breathingAmount(time);
  }

  return pose;
}

/** Convenience wrapper holding the seed and options. */
export class Idle {
  readonly seed: number;
  readonly options: IdleOptions;

  constructor(options: IdleOptions = {}) {
    this.seed = options.seed ?? 0;
    this.options = options;
  }

  apply(pose: Pose, time: number): Pose {
    return applyIdle(pose, time, this.seed, this.options);
  }
}
