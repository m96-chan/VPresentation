import { describe, expect, it } from "vitest";
import { autocorrelate, findFormants, levinsonDurbin, lpcEnvelope } from "../src/lipsync/lpc.js";

const SR = 16000;

/**
 * Two-pole resonator, the standard source-filter building block:
 *   y[n] = x[n] + 2 r cos(theta) y[n-1] - r^2 y[n-2]
 */
function resonate(x: Float32Array, freq: number, bandwidth: number, sr: number): Float32Array {
  const r = Math.exp((-Math.PI * bandwidth) / sr);
  const theta = (2 * Math.PI * freq) / sr;
  const a1 = 2 * r * Math.cos(theta);
  const a2 = -(r * r);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) + a2 * (y[n - 2] ?? 0);
  }
  return y;
}

/** A synthetic vowel: glottal impulse train at `f0` through two formant resonators. */
function synthVowel(f1: number, f2: number, seconds = 0.05, f0 = 120, sr = SR): Float32Array {
  const n = Math.round(seconds * sr);
  const src = new Float32Array(n);
  const period = Math.round(sr / f0);
  for (let i = 0; i < n; i += period) src[i] = 1;
  const y = resonate(resonate(src, f1, 80, sr), f2, 110, sr);
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  if (peak > 0) for (let i = 0; i < y.length; i++) y[i] = (y[i] ?? 0) / peak;
  return y;
}

describe("autocorrelation + Levinson-Durbin", () => {
  it("produces order+1 autocorrelation lags with r[0] as the energy", () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const r = autocorrelate(x, 2);
    expect(r).toHaveLength(3);
    expect(r[0]).toBeCloseTo(1 + 4 + 9 + 16, 5);
    expect(r[1]).toBeCloseTo(1 * 2 + 2 * 3 + 3 * 4, 5);
    expect(r[2]).toBeCloseTo(1 * 3 + 2 * 4, 5);
  });

  it("recovers the coefficients of a known AR(2) process", () => {
    // x[n] = 0.5 x[n-1] - 0.3 x[n-2] + noise  ->  A(z) = 1 - 0.5 z^-1 + 0.3 z^-2
    let s = 12345;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x3fffffff - 1;
    };
    const n = 20000;
    const x = new Float32Array(n);
    for (let i = 2; i < n; i++) {
      x[i] = 0.5 * (x[i - 1] ?? 0) - 0.3 * (x[i - 2] ?? 0) + rnd() * 0.1;
    }
    const { a } = levinsonDurbin(autocorrelate(x, 2), 2);
    expect(a[0]).toBeCloseTo(1, 6);
    expect(a[1]).toBeCloseTo(-0.5, 1);
    expect(a[2]).toBeCloseTo(0.3, 1);
  });

  it("returns a stable all-zero filter for a silent frame", () => {
    const { a, error } = levinsonDurbin(autocorrelate(new Float32Array(512), 12), 12);
    expect(a[0]).toBe(1);
    expect(error).toBe(0);
    expect([...a].slice(1).every((v) => v === 0)).toBe(true);
  });
});

describe("LPC spectral envelope", () => {
  it("peaks near the resonances that generated the signal", () => {
    const x = synthVowel(730, 1090);
    const { a } = levinsonDurbin(autocorrelate(x, 14), 14);
    const env = lpcEnvelope(a, SR, 100, 4000, 400);
    const at = (hz: number) => {
      const i = Math.round(((hz - 100) / (4000 - 100)) * (env.length - 1));
      return env[Math.min(Math.max(i, 0), env.length - 1)] ?? 0;
    };
    expect(at(730)).toBeGreaterThan(at(400));
    expect(at(730)).toBeGreaterThan(at(2500));
    expect(at(1090)).toBeGreaterThan(at(2500));
  });
});

describe("formant estimation", () => {
  // Peterson & Barney (1952) style targets for the five cardinal vowels.
  const cases: Array<[string, number, number]> = [
    ["aaa", 730, 1090],
    ["iii", 270, 2290],
    ["uuu", 300, 870],
    ["eee", 530, 1840],
    ["ooo", 570, 840],
  ];

  for (const [name, f1, f2] of cases) {
    it(`recovers F1/F2 of a synthetic /${name}/`, () => {
      const est = findFormants(synthVowel(f1, f2), SR);
      expect(est).not.toBeNull();
      expect(est!.f1).toBeGreaterThan(f1 - 120);
      expect(est!.f1).toBeLessThan(f1 + 120);
      expect(est!.f2).toBeGreaterThan(f2 - 200);
      expect(est!.f2).toBeLessThan(f2 + 200);
    });
  }

  it("keeps F1 below F2", () => {
    for (const [, f1, f2] of cases) {
      const est = findFormants(synthVowel(f1, f2), SR)!;
      expect(est.f1).toBeLessThan(est.f2);
    }
  });

  it("returns null for silence", () => {
    expect(findFormants(new Float32Array(640), SR)).toBeNull();
  });
});
