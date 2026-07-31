/**
 * Head, body and gaze motion — pose slots 37..38 and 39..43.
 *
 * Five layers, summed, then integrated through per-channel springs:
 *
 *   orientation  a held heading — the large left/right turns
 *   sway         fine procedural drift, so the character is never quite still
 *   gesture      reaction to the speech envelope: nods and lean
 *   posture      a per-emotion bias
 *   thinking     during a pause, the gaze drifts up and to one side
 *
 * Horizontal is the expressive axis and carries most of the performance.
 * Vertical deliberately stays near neutral while speaking — a head that keeps
 * looking up and down reads as restless — and only lifts during pauses, where
 * it reads as thought.
 *
 * Sign conventions, measured from rendered sweeps rather than assumed
 * (`scripts/sweep.ts`). The rule is simple once measured, and was got wrong
 * twice by reading the sweeps by eye:
 *
 *   head_x            positive -> viewer's LEFT
 *   head_y            positive -> chin UP
 *   iris_rotation_x   positive -> viewer's RIGHT
 *   iris_rotation_y   positive -> DOWN
 *
 * **Both iris axes are inverted relative to their head counterpart.** So a
 * gaze that follows the head has to negate it; using the same sign points the
 * eyes the opposite way from the face.
 */
import { POSE_INDEX, POSE_RANGES, type Pose } from "../pose/params.js";
import type { EmotionName, WeightedEmotion } from "../emotion/emotion.js";

export interface BodyMotion {
  readonly headX: number;
  readonly headY: number;
  readonly neckZ: number;
  readonly bodyY: number;
  readonly bodyZ: number;
  /** Eye direction, slots 37..38. Part of "where the character is looking". */
  readonly irisX: number;
  readonly irisY: number;
}

// --- noise -----------------------------------------------------------------

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

function valueNoise(time: number, frequency: number, seed: number): number {
  const x = time * frequency;
  const i = Math.floor(x);
  const f = x - i;
  const a = hash01(i, seed) * 2 - 1;
  const b = hash01(i + 1, seed) * 2 - 1;
  return a + (b - a) * smoothstep(f);
}

/**
 * Two octaves of value noise.
 *
 * A single slow octave only covers a fraction of one cycle in a short clip, so
 * the motion reads as a static offset. The faster octave gives movement you
 * can see within a couple of seconds while the slow one keeps it wandering —
 * but it is kept quiet and only ~2x the base, because a loud fast octave is
 * high-frequency wobble, not life.
 */
function fractalNoise(time: number, frequency: number, seed: number): number {
  return (
    valueNoise(time, frequency, seed) * 0.78 +
    valueNoise(time, frequency * 2.1, seed + 7919) * 0.22
  );
}

// --- inertia ---------------------------------------------------------------

/**
 * A damped spring, integrated implicitly.
 *
 * This is what stops the motion looking stepped. Speech accents and emotion
 * changes are *targets*, not positions: a real head has mass, so it cannot
 * reach a new pose within one frame, and it eases in and out of every move.
 * Driving the pose directly from the envelope — as the first version did —
 * gave `head_y` an acceleration of 0.145 per frame against a total travel of
 * 0.320, i.e. the head teleported on every syllable.
 *
 * Implicit (backward) Euler is used rather than the explicit form because it
 * is unconditionally stable: no combination of stiffness and frame rate can
 * make it blow up, which matters when frames are dropped in realtime.
 */
export class Spring {
  private position: number;
  private velocity = 0;

  constructor(
    /** Natural frequency in Hz — how eagerly it chases the target. */
    private readonly frequency: number,
    /** Damping ratio. 1 is critical; below 1 overshoots and settles. */
    private readonly damping: number,
    initial = 0,
  ) {
    this.position = initial;
  }

  step(target: number, dt: number): number {
    const omega = 2 * Math.PI * this.frequency;
    const f = 1 + 2 * dt * this.damping * omega;
    const hoo = dt * omega * omega;
    const hhoo = dt * hoo;
    const detInv = 1 / (f + hhoo);

    const nextPosition = (f * this.position + dt * this.velocity + hhoo * target) * detInv;
    const nextVelocity = (this.velocity + hoo * (target - this.position)) * detInv;

    this.position = nextPosition;
    this.velocity = nextVelocity;
    return this.position;
  }
}

// --- sway ------------------------------------------------------------------

/**
 * Fine idle drift — the small, continuous wander on top of everything else.
 *
 * Kept deliberately small. The large left/right movement comes from
 * {@link orientationAt}, because continuous noise at a large amplitude reads as
 * aimless floating rather than a character that means to look somewhere.
 */
export function swayAt(time: number, seed: number): BodyMotion {
  return {
    headX: fractalNoise(time, 0.45, seed + 101) * 0.14,
    // Vertical is kept much smaller than horizontal: left/right is the
    // expressive axis, up/down mostly wants to sit near neutral.
    headY: fractalNoise(time, 0.37, seed + 202) * 0.06,
    neckZ: fractalNoise(time, 0.29, seed + 303) * 0.14,
    bodyY: fractalNoise(time, 0.23, seed + 404) * 0.08,
    bodyZ: fractalNoise(time, 0.19, seed + 505) * 0.1,
    irisX: 0,
    irisY: 0,
  };
}

// --- orientation -----------------------------------------------------------

const ORIENT_MIN_HOLD = 1.6;
const ORIENT_MAX_HOLD = 4.2;

/**
 * Which way the character is facing, as a held target in -1..1.
 *
 * Positive faces the viewer's left. A sweep of the student model shows
 * `head_x` and `body_y` stay clean all the way to ±1, so there is far more
 * range available here than idle noise was using — the first version covered
 * about 15% of it.
 *
 * This is a *step* function: a direction is chosen and held for a couple of
 * seconds, then a new one is chosen. Turning happens because the spring
 * downstream cannot follow a step instantly, which is exactly how a real head
 * turn looks — a deliberate move to a new heading, then stillness. Continuous
 * noise at this amplitude would instead look like the character is adrift.
 */
export function orientationAt(time: number, seed: number): number {
  if (time < 0) return 0;

  let start = 0;
  for (let k = 0; k < 4096; k++) {
    const hold = ORIENT_MIN_HOLD + hash01(k, seed + 9187) * (ORIENT_MAX_HOLD - ORIENT_MIN_HOLD);
    if (time < start + hold) {
      // Three kinds of segment, mixed deliberately. A plain
      // `uniform * |uniform|` centre-bias was tried first and clustered almost
      // every heading near zero, so some seeds never turned at all.
      const kind = hash01(k, seed + 8191);
      const magnitude = hash01(k, seed + 7351);

      // Roughly a third of the time, face front — a presenter does come back
      // to centre, and it makes the turns either side of it read as choices.
      if (kind < 0.32) return (magnitude - 0.5) * 0.24;

      return (kind < 0.66 ? -1 : 1) * (0.3 + magnitude * 0.6);
    }
    start += hold;
  }
  return 0;
}

// --- speech dynamics -------------------------------------------------------

export interface SpeechDynamic {
  /** Smoothed loudness, 0..1 — drives sustained lean. */
  readonly level: number;
  /**
   * Loudness relative to its own running average, **signed**, -1..1.
   *
   * Signed on purpose. Clamping this to >= 0 gave nods a permanent downward
   * DC offset: during continuous speech the accent never returned to zero, so
   * the head was held down for the whole utterance. Letting it go negative
   * means the head comes back up between syllables and the nod is centred on
   * the resting pose.
   */
  readonly accent: number;
  /** Seconds of continuous near-silence up to this frame. */
  readonly silence: number;
}

/**
 * Reduce a mouth-openness track to a level and an accent per frame.
 *
 * The accent is openness minus its own slow average, so a sustained vowel
 * stops producing nods after the onset — otherwise the head bobs continuously
 * through every long syllable.
 */
/**
 * Stateful form of {@link speechDynamics}, for feeding frames as they arrive.
 */
export class SpeechDynamicsTracker {
  private slow = 0;
  private level = 0;
  private accent = 0;
  private silenceSeconds = 0;
  private readonly alpha: number;
  private readonly release: number;

  constructor(private readonly fps: number) {
    this.alpha = Math.min(1, 4 / Math.max(1, fps));
    this.release = Math.min(1, 3.5 / Math.max(1, fps));
  }

  step(openness: number): SpeechDynamic {
    this.slow += (openness - this.slow) * this.alpha;
    this.level += (openness - this.level) * this.alpha * 0.5;

    const onset = Math.max(-1, Math.min(1, openness - this.slow));
    this.accent =
      Math.abs(onset) > Math.abs(this.accent)
        ? onset
        : this.accent + (onset - this.accent) * this.release;

    this.silenceSeconds =
      openness < 0.05 ? this.silenceSeconds + 1 / Math.max(1, this.fps) : 0;

    return {
      level: Math.max(0, Math.min(1, this.level)),
      accent: this.accent,
      silence: this.silenceSeconds,
    };
  }
}

export function speechDynamics(openness: readonly number[], fps: number): SpeechDynamic[] {
  const tracker = new SpeechDynamicsTracker(fps);
  return openness.map((v) => tracker.step(v));
}

// --- posture ---------------------------------------------------------------

/**
 * Per-emotion bias. Signs follow THA4's convention where positive `head_y`
 * raises the chin; they were chosen to read correctly on the sample character
 * and are the first thing to flip if an emotion looks inverted.
 */
export const POSTURES: Readonly<Record<EmotionName, Partial<BodyMotion>>> = Object.freeze({
  neutral: {},
  happy: { headY: 0.12, neckZ: 0.1, bodyZ: 0.06 },
  sad: { headY: -0.24, neckZ: -0.12, bodyY: -0.08 },
  angry: { headY: -0.14, bodyZ: -0.06 },
  surprised: { headY: 0.26, bodyY: 0.1 },
  serious: { headY: -0.06, neckZ: -0.04 },
  troubled: { headY: -0.12, neckZ: 0.14 },
});

// --- combination -----------------------------------------------------------

export interface BodyMotionInput {
  readonly time: number;
  readonly seed: number;
  /** Smoothed speech loudness at this time, 0..1. */
  readonly speech: number;
  /** Signed speech accent at this time, -1..1. */
  readonly accent: number;
  /** Seconds of continuous silence up to this frame; drives the thinking gaze. */
  readonly silence?: number;
  readonly emotions?: readonly WeightedEmotion[];
  readonly swayScale?: number;
  readonly gestureScale?: number;
  readonly postureScale?: number;
  /** Scales how far the character turns left/right. */
  readonly turnScale?: number;
  /**
   * Centre of the idle facing range, -1..1.
   *
   * A presenter standing in the bottom-right corner has the deck to their
   * left, so idling towards the right points them out of frame and away from
   * what they are talking about. The compositor sets this from which corner
   * the character is standing in.
   */
  readonly headingBias?: number;
  /**
   * Override the facing target, -1..1 (positive is the viewer's left).
   *
   * Defaults to the procedural {@link orientationAt}. A compositor can drive
   * this instead to make the character look at the slide.
   */
  readonly heading?: number;
}

const clamp1 = (v: number) => (v < -1 ? -1 : v > 1 ? 1 : v);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export function bodyMotionAt(input: BodyMotionInput): BodyMotion {
  const swayScale = input.swayScale ?? 1;
  const gestureScale = input.gestureScale ?? 1;
  const postureScale = input.postureScale ?? 1;
  const turnScale = input.turnScale ?? 1;

  const sway = swayAt(input.time, input.seed);

  let headX = sway.headX * swayScale;
  let headY = sway.headY * swayScale;
  let neckZ = sway.neckZ * swayScale;
  let bodyY = sway.bodyY * swayScale;
  let bodyZ = sway.bodyZ * swayScale;

  // Facing. The head does most of the turn, the torso follows part way, and
  // the neck adds a little counter-tilt so it is not a rigid pivot.
  const bias = clamp1(input.headingBias ?? 0);
  // The wander is scaled to fit *within* the biased side rather than added on
  // top of it. Adding a bias to a zero-centred range still crossed to the far
  // side whenever the generator's own sample happened to skew that way — a
  // presenter in the right-hand corner would turn away from the deck for a
  // stretch, which is exactly what looked wrong.
  const raw = input.heading ?? orientationAt(input.time, input.seed);
  const spread = (1 - Math.abs(bias)) * 0.9;
  const heading = clamp1(bias + raw * spread) * turnScale;
  headX += heading * 0.62;
  bodyY += heading * 0.34;
  neckZ += heading * 0.1;

  // Eyes lead the head. They are sprung faster downstream, so they arrive at
  // the new heading first and the head follows — the standard way to make a
  // turn read as intentional rather than mechanical.
  //
  // Negated: iris_rotation_x runs the opposite way to head_x, so matching the
  // signs made the character look away from wherever its head was turning.
  let irisX = -heading * 0.85;

  if (gestureScale !== 0) {
    // A nod on each accent, with a little lateral scatter so repeated accents
    // do not look like a metronome.
    // Amplitudes are pre-spring targets; the spring attenuates them, so these
    // are larger than the travel you actually see.
    const scatter = valueNoise(input.time, 0.9, input.seed + 606);
    headY -= input.accent * 0.24 * gestureScale;
    headX += input.accent * 0.22 * scatter * gestureScale;
    neckZ += input.accent * 0.16 * scatter * gestureScale;

    // Sustained speech leans the body slightly forward and steadies it.
    bodyY += input.speech * 0.12 * gestureScale;
    bodyZ += input.speech * 0.07 * scatter * gestureScale;
  }

  if (postureScale !== 0 && input.emotions) {
    for (const { emotion, weight } of input.emotions) {
      const posture = POSTURES[emotion];
      const w = weight * postureScale;
      headX += (posture.headX ?? 0) * w;
      headY += (posture.headY ?? 0) * w;
      neckZ += (posture.neckZ ?? 0) * w;
      bodyY += (posture.bodyY ?? 0) * w;
      bodyZ += (posture.bodyZ ?? 0) * w;
    }
  }

  // Thinking gaze — applied last, on top of posture rather than instead of it,
  // so a sad character still holds its posture while it pauses. During a pause
  // the gaze drifts up and to one side, the universal "working it out" tell;
  // while actually speaking the vertical axis stays near neutral.
  const thinking = smoothstep(clamp01(((input.silence ?? 0) - 0.3) / 0.7));
  let irisY = 0;
  if (thinking > 0) {
    // Mostly one way, but not always, so a long pause is not a fixed stare.
    //
    // It follows the heading bias when there is one. Left as a fixed
    // preference it pulled against the bias instead: a presenter in the
    // left-hand corner had a constant +0.26 of "thinking to the left" cancelling
    // its lean towards the deck, and ended up staring straight ahead.
    const preferred = bias === 0 ? 1 : Math.sign(bias);
    const side =
      hash01(Math.floor(input.time / 3.5), input.seed + 4441) < 0.72 ? preferred : -preferred;
    headY += thinking * 0.2;
    headX += thinking * 0.26 * side;
    neckZ += thinking * 0.1 * side;
    // Negated for the same reason as above.
    irisX -= thinking * 0.55 * side;
    irisY -= thinking * 0.5;
  }

  // Same inversion on the vertical axis: negative iris_rotation_y looks up,
  // while positive head_y raises the chin.
  irisY -= headY * 0.6;

  return {
    headX: clamp1(headX),
    headY: clamp1(headY),
    neckZ: clamp1(neckZ),
    bodyY: clamp1(bodyY),
    bodyZ: clamp1(bodyZ),
    irisX: clamp1(irisX),
    irisY: clamp1(irisY),
  };
}

const SLOTS = [
  [POSE_INDEX.head_x, "headX"],
  [POSE_INDEX.head_y, "headY"],
  [POSE_INDEX.neck_z, "neckZ"],
  [POSE_INDEX.body_y, "bodyY"],
  [POSE_INDEX.body_z, "bodyZ"],
  [POSE_INDEX.iris_rotation_x, "irisX"],
  [POSE_INDEX.iris_rotation_y, "irisY"],
] as const;

/**
 * Per-channel inertia.
 *
 * The head is light and answers quickly; the torso is heavy and lags behind
 * it. That difference is most of what makes the motion read as a body rather
 * than five independent sliders. Damping sits just under 1 so moves settle
 * with a hint of overshoot instead of arriving dead.
 */
const SPRINGS: Readonly<Record<keyof BodyMotion, { frequency: number; damping: number }>> = {
  // Turns are the largest moves head_x makes, so it is sprung softly:
  // a bigger step through a stiffer spring means a harder start.
  headX: { frequency: 0.85, damping: 0.9 },
  // head_y takes the nod impulses, so it is the channel that snaps first.
  // It is deliberately the heaviest of the three head axes.
  headY: { frequency: 0.85, damping: 0.92 },
  neckZ: { frequency: 0.9, damping: 0.88 },
  bodyY: { frequency: 0.6, damping: 0.95 },
  bodyZ: { frequency: 0.5, damping: 0.95 },
  // Eyes are near weightless: they snap to a new heading well ahead of the
  // head, which is what sells the turn as deliberate.
  irisX: { frequency: 3.2, damping: 0.9 },
  irisY: { frequency: 2.6, damping: 0.9 },
};

const CHANNELS = ["headX", "headY", "neckZ", "bodyY", "bodyZ", "irisX", "irisY"] as const;

export interface SpringScale {
  /** Multiplies every channel's natural frequency; lower = heavier, smoother. */
  readonly stiffness?: number;
}

/**
 * Turn a sequence of per-frame targets into actual motion.
 *
 * Springs carry state across frames, so this works on the whole track at once
 * rather than frame by frame. It stays deterministic — the same input sequence
 * always integrates to the same output — which is what keeps the realtime and
 * offline renderers in agreement.
 */
/** Holds the springs so motion can be integrated one frame at a time. */
export class BodyMotionIntegrator {
  private readonly dt: number;
  private readonly springs: Record<keyof BodyMotion, Spring>;

  constructor(fps: number, options: SpringScale = {}) {
    const stiffness = options.stiffness ?? 1;
    this.dt = 1 / Math.max(1, fps);
    this.springs = Object.fromEntries(
      CHANNELS.map((key) => [
        key,
        new Spring(SPRINGS[key].frequency * stiffness, SPRINGS[key].damping),
      ]),
    ) as Record<keyof BodyMotion, Spring>;
  }

  /**
   * Integrate one step.
   *
   * `dt` defaults to the nominal frame time, but the live engine passes the
   * real elapsed time so that dropping frames slows the integration rather
   * than changing how the motion reads. The springs are implicit, so any `dt`
   * is stable.
   */
  step(input: BodyMotionInput, dt: number = this.dt): BodyMotion {
    const target = bodyMotionAt(input);
    const springs = this.springs;
    return {
      headX: clamp1(springs.headX.step(target.headX, dt)),
      headY: clamp1(springs.headY.step(target.headY, dt)),
      neckZ: clamp1(springs.neckZ.step(target.neckZ, dt)),
      bodyY: clamp1(springs.bodyY.step(target.bodyY, dt)),
      bodyZ: clamp1(springs.bodyZ.step(target.bodyZ, dt)),
      irisX: clamp1(springs.irisX.step(target.irisX, dt)),
      irisY: clamp1(springs.irisY.step(target.irisY, dt)),
    };
  }
}

export function bodyMotionTrack(
  inputs: readonly BodyMotionInput[],
  fps: number,
  options: SpringScale = {},
): BodyMotion[] {
  if (inputs.length === 0) return [];
  const integrator = new BodyMotionIntegrator(fps, options);
  return inputs.map((input) => integrator.step(input));
}


/** Write one frame of head/body motion into `pose`, touching nothing else. */
export function writeBodyMotion(pose: Pose, motion: BodyMotion): Pose {
  for (const [index, key] of SLOTS) {
    const range = POSE_RANGES[index];
    const v = motion[key];
    pose[index] = range ? Math.min(range[1], Math.max(range[0], v)) : v;
  }
  return pose;
}

/**
 * Un-sprung single-frame application.
 *
 * Kept for callers that drive the pose themselves; the pose-track builder uses
 * {@link bodyMotionTrack} instead, because inertia needs frame-to-frame state.
 */
export function applyBodyMotion(pose: Pose, input: BodyMotionInput): Pose {
  return writeBodyMotion(pose, bodyMotionAt(input));
}
