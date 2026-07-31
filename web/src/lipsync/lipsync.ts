/**
 * Audio-driven lip sync: waveform -> the five THA4 mouth vowels.
 *
 * THA4's mouth parameters are `mouth_aaa / iii / uuu / eee / ooo` (indices
 * 26..30) — five cardinal vowel shapes. Each analysis frame is reduced to an
 * openness (how far the jaw drops) and a distribution over those five shapes,
 * obtained by matching the frame's estimated F1/F2 against canonical formant
 * targets.
 *
 * The exported {@link LipsyncSource} seam exists so a forced aligner can
 * replace this estimator later without touching the pose-track builder.
 */
import { findFormants, rms } from "./lpc.js";
import { POSE_INDEX, type Pose } from "../pose/params.js";

export const VOWEL_NAMES = ["aaa", "iii", "uuu", "eee", "ooo"] as const;
export type VowelName = (typeof VOWEL_NAMES)[number];
export type VowelWeights = Record<VowelName, number>;

export interface Formants {
  readonly f1: number;
  readonly f2: number;
}

/**
 * Canonical formant targets, following the Peterson & Barney (1952) vowel
 * space for American English. The five THA4 shapes are close enough to the
 * cardinal vowels that this maps directly.
 */
export const VOWEL_TARGETS: Readonly<Record<VowelName, Formants>> = Object.freeze({
  aaa: { f1: 730, f2: 1090 },
  iii: { f1: 270, f2: 2290 },
  uuu: { f1: 300, f2: 870 },
  eee: { f1: 530, f2: 1840 },
  ooo: { f1: 570, f2: 840 },
});

export interface MouthFrame {
  /** Frame start, in seconds from the beginning of the audio. */
  readonly time: number;
  /** Jaw opening, 0..1. */
  readonly openness: number;
  /** Distribution over the five vowel shapes; sums to 1. */
  readonly vowels: VowelWeights;
}

/** Anything that can turn audio (plus optional text) into a mouth track. */
export interface LipsyncSource {
  readonly name: string;
  mouthTrack(samples: Float32Array, sampleRate: number, text?: string): MouthFrame[];
}

/** Perceptual distance behaves far better in mel space than in raw Hz. */
function toMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/**
 * Softmax over the negative mel-space distance to each vowel target.
 *
 * `sharpness` controls how decisively a frame commits to one shape; the
 * default keeps some blending so transitions between vowels stay smooth.
 */
export function classifyVowel(f1: number, f2: number, sharpness = 0.012): VowelWeights {
  const m1 = toMel(f1);
  const m2 = toMel(f2);

  const scores = VOWEL_NAMES.map((name) => {
    const target = VOWEL_TARGETS[name];
    const d1 = m1 - toMel(target.f1);
    // F2 carries most of the front/back distinction, so weight it a little
    // less than F1 per-mel but let its much larger spread do the work.
    const d2 = m2 - toMel(target.f2);
    return -Math.sqrt(d1 * d1 + d2 * d2);
  });

  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - max) / (sharpness * 1000)));
  const total = exps.reduce((a, b) => a + b, 0);

  const out = {} as VowelWeights;
  VOWEL_NAMES.forEach((name, i) => {
    out[name] = (exps[i] ?? 0) / total;
  });
  return out;
}

export interface LipsyncOptions {
  /** Output frames per second. */
  readonly fps?: number;
  /** Analysis window length in seconds. */
  readonly windowSeconds?: number;
  /**
   * RMS mapped to fully-open. Chatterbox output is roughly peak-normalised,
   * so a fixed reference works better than per-clip normalisation, which
   * would make quiet sentences gape.
   */
  readonly loudRms?: number;
  /** RMS below which the mouth is treated as shut. */
  readonly silenceRms?: number;
  /** Per-frame smoothing coefficients, 0..1 (higher = snappier). */
  readonly attack?: number;
  readonly release?: number;
}

const DEFAULTS = {
  fps: 30,
  windowSeconds: 0.04,
  loudRms: 0.12,
  silenceRms: 0.006,
  attack: 0.55,
  release: 0.28,
} as const;

const CLOSED: VowelWeights = Object.freeze({ aaa: 0, iii: 0, uuu: 0, eee: 0, ooo: 0 });

function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/**
 * Analyse `samples` into a mouth track at `fps`.
 *
 * Frames are windowed and pre-emphasised (the standard +6 dB/oct tilt that
 * flattens the glottal source before LPC), then reduced to openness + vowel
 * weights, then smoothed with an asymmetric attack/release so the jaw ramps
 * instead of snapping.
 */
/**
 * Incremental analyser.
 *
 * Audio arrives chunk by chunk from `VoxShot.stream()`, but the smoothing is
 * sequential and the analysis window is *centred* — a frame needs audio from
 * after it — so frames cannot simply be computed per chunk and concatenated.
 * This holds the state across chunks and only releases a frame once its window
 * is complete, which makes streamed output identical to batch output. Anything
 * less and the recorded video would not match what was watched.
 */
export class LipsyncAnalyser {
  private readonly opts: Required<LipsyncOptions>;
  private readonly hop: number;
  private readonly windowSize: number;
  private readonly window: Float64Array;
  private readonly buf: Float32Array;

  /**
   * A sliding window of received audio.
   *
   * Only the tail still needed by unemitted frames is kept: frames are
   * consumed in order, so once a frame's window has closed its samples can
   * never be read again. Retaining the whole utterance instead grew without
   * bound — 60 s alone held 2.1 M samples, and a paper-length reading would
   * have been hundreds of megabytes.
   */
  private samples: Float32Array = new Float32Array(0);
  /** Absolute count of samples received, including discarded ones. */
  private length = 0;
  /** Absolute index of `samples[0]`. */
  private dropped = 0;
  private nextFrame = 0;
  private openness = 0;
  private smoothed: VowelWeights = { ...CLOSED };

  constructor(
    private readonly sampleRate: number,
    options: LipsyncOptions = {},
  ) {
    this.opts = { ...DEFAULTS, ...options };
    this.hop = sampleRate / this.opts.fps;
    this.windowSize = Math.max(64, Math.round(this.opts.windowSeconds * sampleRate));
    this.window = hann(this.windowSize);
    this.buf = new Float32Array(this.windowSize);
  }

  /** Seconds of audio received so far. */
  get duration(): number {
    return this.sampleRate > 0 ? this.length / this.sampleRate : 0;
  }

  private get held(): number {
    return this.length - this.dropped;
  }

  private grow(extra: number): void {
    if (this.held + extra <= this.samples.length) return;
    const next = new Float32Array(Math.max(this.held + extra, this.samples.length * 2, 1 << 15));
    next.set(this.samples.subarray(0, this.held));
    this.samples = next;
  }

  /** Discard audio no unemitted frame can reach any more. */
  private compact(): void {
    const nextFrom = Math.max(
      0,
      Math.round(this.nextFrame * this.hop) - (this.windowSize >> 1),
    );
    const discard = nextFrom - this.dropped;
    // Only bother once a worthwhile amount has gone stale.
    if (discard < this.windowSize * 8) return;
    this.samples.copyWithin(0, discard, this.held);
    this.dropped += discard;
  }

  /** Append audio and return the frames that became final because of it. */
  push(chunk: Float32Array): MouthFrame[] {
    if (chunk.length === 0) return [];
    this.grow(chunk.length);
    this.samples.set(chunk, this.held);
    this.length += chunk.length;
    const frames = this.drain(false);
    this.compact();
    return frames;
  }

  /** Emit the remaining frames, whose windows run past the end of the audio. */
  flush(): MouthFrame[] {
    return this.drain(true);
  }

  private drain(final: boolean): MouthFrame[] {
    if (this.length === 0) return [];

    const frames: MouthFrame[] = [];
    const limit = final
      ? Math.max(1, Math.round(this.length / this.hop))
      : Math.round(this.length / this.hop);

    while (this.nextFrame < limit) {
      const start = Math.round(this.nextFrame * this.hop);
      const from = Math.max(0, Math.min(this.length - 1, start - (this.windowSize >> 1)));

      // Not final yet: the tail of this frame's window is still in the future.
      if (!final && from + this.windowSize > this.length) break;

      // A frame whose window has already been discarded cannot be recovered;
      // that would be a bug in the compaction bound, not something to paper
      // over silently.
      if (from < this.dropped) throw new Error("lipsync window was discarded too early");

      frames.push(this.analyseFrame(start, from));
      this.nextFrame++;
    }
    return frames;
  }

  private analyseFrame(start: number, from: number): MouthFrame {
    const { buf, window, windowSize, opts } = this;
    const available = Math.min(windowSize, this.length - from);

    buf.fill(0);
    const base = from - this.dropped;
    for (let i = 0; i < available; i++) buf[i] = this.samples[base + i] ?? 0;

    const level = rms(buf.subarray(0, available));

    // Pre-emphasis, then window. Done after the level measurement so loudness
    // is not tilted by the filter.
    for (let i = windowSize - 1; i > 0; i--) {
      buf[i] = ((buf[i] ?? 0) - 0.97 * (buf[i - 1] ?? 0)) * (window[i] ?? 0);
    }
    buf[0] = (buf[0] ?? 0) * (window[0] ?? 0);

    let targetOpen = 0;
    let targetVowels: VowelWeights = CLOSED;
    if (level >= opts.silenceRms) {
      const formants = findFormants(buf, this.sampleRate);
      if (formants) {
        const norm = (level - opts.silenceRms) / Math.max(1e-9, opts.loudRms - opts.silenceRms);
        targetOpen = Math.min(1, Math.max(0, norm));
        targetVowels = classifyVowel(formants.f1, formants.f2);
      }
    }

    const rate = targetOpen > this.openness ? opts.attack : opts.release;
    this.openness += (targetOpen - this.openness) * rate;
    if (this.openness < 1e-4) this.openness = 0;

    const next = {} as VowelWeights;
    for (const name of VOWEL_NAMES) {
      next[name] = this.smoothed[name] + (targetVowels[name] - this.smoothed[name]) * opts.attack;
    }
    // Renormalise so the weights keep summing to 1 through the smoothing.
    const total = VOWEL_NAMES.reduce((s, n) => s + next[n], 0);
    if (total > 1e-6) for (const name of VOWEL_NAMES) next[name] /= total;
    this.smoothed = next;

    return {
      time: start / this.sampleRate,
      openness: this.openness,
      vowels: { ...this.smoothed },
    };
  }
}

/**
 * Analyse a complete utterance.
 *
 * Implemented on top of {@link LipsyncAnalyser} rather than beside it, so the
 * batch and streaming paths cannot drift apart.
 */
export function analyseLipsync(
  samples: Float32Array,
  sampleRate: number,
  options: LipsyncOptions = {},
): MouthFrame[] {
  if (samples.length === 0 || sampleRate <= 0) return [];
  const analyser = new LipsyncAnalyser(sampleRate, options);
  return [...analyser.push(samples), ...analyser.flush()];
}

/** The default source: LPC formant estimation straight off the waveform. */
export const formantLipsync: LipsyncSource = {
  name: "formant-lpc",
  mouthTrack: (samples, sampleRate) => analyseLipsync(samples, sampleRate),
};

const VOWEL_SLOT: Readonly<Record<VowelName, number>> = {
  aaa: POSE_INDEX.mouth_aaa,
  iii: POSE_INDEX.mouth_iii,
  uuu: POSE_INDEX.mouth_uuu,
  eee: POSE_INDEX.mouth_eee,
  ooo: POSE_INDEX.mouth_ooo,
};

/**
 * Write a mouth frame into a pose vector (slots 26..31), leaving every other
 * slot alone so emotion and idle can own theirs.
 */
export function applyMouth(pose: Pose, frame: MouthFrame): Pose {
  for (const name of VOWEL_NAMES) {
    pose[VOWEL_SLOT[name]] = frame.vowels[name] * frame.openness;
  }
  // mouth_delta widens the whole mouth; tie it to openness so louder speech
  // reads as more articulated rather than only more open.
  pose[POSE_INDEX.mouth_delta] = frame.openness * 0.35;
  return pose;
}
