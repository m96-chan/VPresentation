/**
 * Linear-predictive formant estimation.
 *
 * VoxShot returns audio only — `synthesize()` resolves to a `Float32Array`,
 * and `SynthesizedAudio` exposes nothing but samples, sampleRate and duration.
 * There are no phoneme timings to read, and Chatterbox (autoregressive) has no
 * duration predictor to borrow an alignment from, so the mouth shape has to be
 * recovered from the waveform.
 *
 * LPC is the right tool: the vocal tract *is* an all-pole filter, so fitting
 * one to each short frame and reading off its resonances gives F1/F2 directly.
 * Working in the time domain (autocorrelation + Levinson-Durbin, then
 * evaluating the envelope on a frequency grid) avoids needing an FFT at all.
 */

/** Autocorrelation lags `0..order` of `x`. */
export function autocorrelate(x: Float32Array, order: number): Float64Array {
  const r = new Float64Array(order + 1);
  for (let lag = 0; lag <= order; lag++) {
    let sum = 0;
    for (let i = lag; i < x.length; i++) sum += (x[i] ?? 0) * (x[i - lag] ?? 0);
    r[lag] = sum;
  }
  return r;
}

export interface LpcFit {
  /** Prediction-filter coefficients `A(z)`, with `a[0] === 1`. */
  readonly a: Float64Array;
  /** Residual (prediction error) energy. */
  readonly error: number;
}

/**
 * Solve the Yule-Walker equations for the all-pole filter of the given order.
 *
 * A silent (or degenerate) frame yields the identity filter and zero error,
 * which callers use as the "unvoiced, no formants" signal.
 */
export function levinsonDurbin(r: Float64Array, order: number): LpcFit {
  const a = new Float64Array(order + 1);
  a[0] = 1;
  let error = r[0] ?? 0;
  if (error <= 0) return { a, error: 0 };

  const tmp = new Float64Array(order + 1);
  for (let i = 1; i <= order; i++) {
    let acc = r[i] ?? 0;
    for (let j = 1; j < i; j++) acc += (a[j] ?? 0) * (r[i - j] ?? 0);
    const k = -acc / error;
    if (!Number.isFinite(k)) return { a, error: 0 };

    tmp.set(a);
    for (let j = 1; j < i; j++) a[j] = (tmp[j] ?? 0) + k * (tmp[i - j] ?? 0);
    a[i] = k;

    error *= 1 - k * k;
    // Numerically the recursion can drive the error non-positive; stop there
    // rather than emitting an unstable filter.
    if (error <= 0) return { a, error: 0 };
  }
  return { a, error };
}

/**
 * Magnitude of `1 / A(e^{jw})` sampled on a linear frequency grid.
 *
 * Returned in dB so peak comparisons behave sensibly across loudness.
 */
export function lpcEnvelope(
  a: Float64Array,
  sampleRate: number,
  minHz: number,
  maxHz: number,
  points: number,
): Float64Array {
  const env = new Float64Array(points);
  const step = points > 1 ? (maxHz - minHz) / (points - 1) : 0;
  for (let p = 0; p < points; p++) {
    const w = (2 * Math.PI * (minHz + step * p)) / sampleRate;
    let re = 0;
    let im = 0;
    for (let k = 0; k < a.length; k++) {
      const ak = a[k] ?? 0;
      re += ak * Math.cos(w * k);
      im -= ak * Math.sin(w * k);
    }
    const mag2 = re * re + im * im;
    env[p] = mag2 > 1e-20 ? -10 * Math.log10(mag2) : 100;
  }
  return env;
}

export interface Formants {
  readonly f1: number;
  readonly f2: number;
}

const MIN_HZ = 150;
const MAX_HZ = 4000;
const POINTS = 512;
/** Frames quieter than this (RMS) carry no usable vowel information. */
const SILENCE_RMS = 1e-4;

export function rms(x: Float32Array): number {
  let sum = 0;
  for (const v of x) sum += v * v;
  return x.length > 0 ? Math.sqrt(sum / x.length) : 0;
}

/** LPC order rule of thumb: two poles per kHz, plus a few for the source. */
export function lpcOrder(sampleRate: number): number {
  return Math.min(30, Math.round(sampleRate / 1000) + 4);
}

/**
 * Estimate the first two formants of a single (already windowed) frame.
 *
 * Returns `null` when the frame is silent or the fit degenerates — callers
 * treat that as "mouth closed" rather than guessing a vowel.
 */
export function findFormants(frame: Float32Array, sampleRate: number): Formants | null {
  if (rms(frame) < SILENCE_RMS) return null;

  const order = lpcOrder(sampleRate);
  const { a, error } = levinsonDurbin(autocorrelate(frame, order), order);
  if (error <= 0) return null;

  const env = lpcEnvelope(a, sampleRate, MIN_HZ, MAX_HZ, POINTS);
  const step = (MAX_HZ - MIN_HZ) / (POINTS - 1);

  // Collect interior local maxima, refining each with a parabolic fit so the
  // estimate is not quantised to the grid.
  const peaks: Array<{ hz: number; db: number }> = [];
  for (let i = 1; i < env.length - 1; i++) {
    const prev = env[i - 1] ?? 0;
    const cur = env[i] ?? 0;
    const next = env[i + 1] ?? 0;
    if (cur <= prev || cur < next) continue;
    const denom = prev - 2 * cur + next;
    const shift = denom !== 0 ? (0.5 * (prev - next)) / denom : 0;
    peaks.push({ hz: MIN_HZ + (i + shift) * step, db: cur });
  }
  if (peaks.length === 0) return null;

  if (peaks.length === 1) {
    const only = peaks[0]!;
    // A single resonance: treat it as F1 and put F2 just above it so the
    // classifier still sees an ordered pair.
    return { f1: only.hz, f2: Math.min(MAX_HZ, only.hz * 1.6) };
  }

  // Take the two strongest peaks, then order them by frequency.
  const strongest = [...peaks].sort((x, y) => y.db - x.db).slice(0, 2);
  const [lo, hi] = strongest.sort((x, y) => x.hz - y.hz) as [
    { hz: number; db: number },
    { hz: number; db: number },
  ];
  return { f1: lo.hz, f2: hi.hz };
}
