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
 * **THA4's pose names are rotation axes, not directions.** `head_x` is
 * rotation *about* x, which is pitch; `head_y` is yaw. Reading them as "the x
 * position of the head" gets every axis wrong, which is what happened here —
 * turns were driven into the pitch axis, so the character nodded up and down
 * instead of looking left and right, and flipping the sign of the "turn"
 * changed nothing about where it faced.
 *
 * The authority is THA4's own mocap converters, which map a tracked face onto
 * the pose vector (`tha4/mocap/mediapipe_face_pose_converter_00.py`):
 *
 *   head_x  <- euler[0] about X  -> PITCH     positive = chin up
 *   head_y  <- euler[1] about Y  -> YAW       positive = viewer's left
 *   neck_z  <- euler[2] about Z  -> ROLL
 *   body_y  <- the same value as head_y  -> yaw
 *   body_z  <- the same value as neck_z  -> roll
 *
 *   iris_rotation_x <- EYE_LOOK_UP - EYE_LOOK_DOWN   -> gaze PITCH, positive = up
 *   iris_rotation_y <- EYE_LOOK_IN / OUT             -> gaze YAW,   positive = viewer's left
 *
 * Head and gaze therefore share their signs. Fields here are named for what
 * they *do* (`yaw`, `pitch`, `roll`) so the confusion cannot recur; the mapping
 * onto axis-named slots happens once, in `SLOTS`.
 */
import { POSE_INDEX, POSE_RANGES, type Pose } from "../pose/params.js";
import type { EmotionName, WeightedEmotion } from "../emotion/emotion.js";

export interface BodyMotion {
  /** Head turn. Positive is the viewer's left. Maps to `head_y`. */
  readonly yaw: number;
  /** Head nod. Positive raises the chin. Maps to `head_x`. */
  readonly pitch: number;
  /** Head tilt. Maps to `neck_z`. */
  readonly roll: number;
  /** Torso turn, following the head. Maps to `body_y`. */
  readonly bodyYaw: number;
  /** Torso tilt. Maps to `body_z`. */
  readonly bodyRoll: number;
  /** Gaze turn, same sign as {@link yaw}. Maps to `iris_rotation_y`. */
  readonly gazeYaw: number;
  /** Gaze nod, same sign as {@link pitch}. Maps to `iris_rotation_x`. */
  readonly gazePitch: number;
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
    yaw: fractalNoise(time, 0.45, seed + 101) * 0.14,
    // Pitch is kept much smaller than yaw: left/right is the expressive axis,
    // up/down mostly wants to sit near neutral.
    pitch: fractalNoise(time, 0.37, seed + 202) * 0.06,
    roll: fractalNoise(time, 0.29, seed + 303) * 0.14,
    bodyYaw: fractalNoise(time, 0.23, seed + 404) * 0.08,
    bodyRoll: fractalNoise(time, 0.19, seed + 505) * 0.1,
    gazeYaw: 0,
    gazePitch: 0,
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
  happy: { pitch: 0.1, roll: 0.1, bodyRoll: 0.06 },
  sad: { pitch: -0.12, roll: -0.12, bodyYaw: -0.06 },
  angry: { pitch: -0.07, bodyRoll: -0.06 },
  surprised: { pitch: 0.2, bodyYaw: 0.1 },
  serious: { pitch: -0.03, roll: -0.04 },
  troubled: { pitch: -0.06, roll: 0.14 },
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
  let yaw = sway.yaw * swayScale;
  let pitch = sway.pitch * swayScale;
  let roll = sway.roll * swayScale;
  let bodyYaw = sway.bodyYaw * swayScale;
  let bodyRoll = sway.bodyRoll * swayScale;

  // 1. Facing. The head does most of the turn, the torso follows part way, and
  //    the neck adds a little counter-tilt so it is not a rigid pivot.
  const bias = clamp1(input.headingBias ?? 0);
  // The wander is scaled to fit *within* the biased side rather than added on
  // top of it: adding a bias to a zero-centred range still crossed to the far
  // side whenever the generator's own sample skewed that way.
  const raw = input.heading ?? orientationAt(input.time, input.seed);
  const heading = clamp1(bias + raw * ((1 - Math.abs(bias)) * 0.9)) * turnScale;

  yaw += heading * 0.62;
  bodyYaw += heading * 0.34;
  roll += heading * 0.1;
  // Eyes lead the head — sprung faster downstream, so they arrive first and the
  // head follows. Same sign: gaze yaw and head yaw agree.
  let gazeYaw = heading * 0.85;

  // 2. Speech gesture. Amplitudes are pre-spring targets, so they are larger
  //    than the travel you actually see.
  if (gestureScale !== 0) {
    const scatter = valueNoise(input.time, 0.9, input.seed + 606);
    // `accent` is signed, so this nods down on an onset and recovers upward
    // instead of holding the head down for the whole utterance.
    pitch -= input.accent * 0.24 * gestureScale;
    yaw += input.accent * 0.22 * scatter * gestureScale;
    roll += input.accent * 0.16 * scatter * gestureScale;

    bodyYaw += input.speech * 0.12 * gestureScale;
    bodyRoll += input.speech * 0.07 * scatter * gestureScale;
  }

  // 3. Emotion posture.
  if (postureScale !== 0 && input.emotions) {
    for (const { emotion, weight } of input.emotions) {
      const posture = POSTURES[emotion];
      const w = weight * postureScale;
      yaw += (posture.yaw ?? 0) * w;
      pitch += (posture.pitch ?? 0) * w;
      roll += (posture.roll ?? 0) * w;
      bodyYaw += (posture.bodyYaw ?? 0) * w;
      bodyRoll += (posture.bodyRoll ?? 0) * w;
    }
  }

  // 4. Thinking gaze — pauses only. While speaking, pitch sits near neutral.
  const thinking = smoothstep(clamp01(((input.silence ?? 0) - 0.3) / 0.7));
  let gazePitch = 0;
  if (thinking > 0) {
    // Follows the heading bias; a fixed preference pulled against it and left a
    // corner-placed presenter staring straight ahead.
    const preferred = bias === 0 ? 1 : Math.sign(bias);
    const side =
      hash01(Math.floor(input.time / 3.5), input.seed + 4441) < 0.72 ? preferred : -preferred;
    pitch += thinking * 0.2;
    yaw += thinking * 0.26 * side;
    roll += thinking * 0.1 * side;
    gazeYaw += thinking * 0.55 * side;
    gazePitch += thinking * 0.5;
  }

  // Gaze follows the head's nod, same sign.
  gazePitch += pitch * 0.6;

  return {
    yaw: clamp1(yaw),
    pitch: clamp1(pitch),
    roll: clamp1(roll),
    bodyYaw: clamp1(bodyYaw),
    bodyRoll: clamp1(bodyRoll),
    gazeYaw: clamp1(gazeYaw),
    gazePitch: clamp1(gazePitch),
  };
}

/**
 * The one place semantic names meet THA4's axis-named slots.
 *
 * `head_x` is rotation about x — pitch — and `head_y` is yaw. Keeping the
 * mapping here, rather than spread through the motion code, is what stops the
 * two being confused again.
 */
const SLOTS = [
  [POSE_INDEX.head_y, "yaw"],
  [POSE_INDEX.head_x, "pitch"],
  [POSE_INDEX.neck_z, "roll"],
  [POSE_INDEX.body_y, "bodyYaw"],
  [POSE_INDEX.body_z, "bodyRoll"],
  [POSE_INDEX.iris_rotation_y, "gazeYaw"],
  [POSE_INDEX.iris_rotation_x, "gazePitch"],
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
  // Turns are the largest moves the head makes, so yaw is sprung softly: a
  // bigger step through a stiffer spring means a harder start.
  yaw: { frequency: 0.85, damping: 0.9 },
  // Pitch takes the nod impulses, so it is the channel that snaps first.
  pitch: { frequency: 0.85, damping: 0.92 },
  roll: { frequency: 0.9, damping: 0.88 },
  bodyYaw: { frequency: 0.6, damping: 0.95 },
  bodyRoll: { frequency: 0.5, damping: 0.95 },
  // Eyes are near weightless: they reach a new heading well ahead of the head,
  // which is what sells the turn as deliberate.
  gazeYaw: { frequency: 3.2, damping: 0.9 },
  gazePitch: { frequency: 2.6, damping: 0.9 },
};

const CHANNELS = ["yaw", "pitch", "roll", "bodyYaw", "bodyRoll", "gazeYaw", "gazePitch"] as const;

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
      yaw: clamp1(springs.yaw.step(target.yaw, dt)),
      pitch: clamp1(springs.pitch.step(target.pitch, dt)),
      roll: clamp1(springs.roll.step(target.roll, dt)),
      bodyYaw: clamp1(springs.bodyYaw.step(target.bodyYaw, dt)),
      bodyRoll: clamp1(springs.bodyRoll.step(target.bodyRoll, dt)),
      gazeYaw: clamp1(springs.gazeYaw.step(target.gazeYaw, dt)),
      gazePitch: clamp1(springs.gazePitch.step(target.gazePitch, dt)),
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
