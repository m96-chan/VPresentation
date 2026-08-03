import { describe, expect, it } from "vitest";
import { POSE_INDEX, zeroPose } from "../src/pose/params.js";
import {
  VOWEL_NAMES,
  VOWEL_TARGETS,
  analyseLipsync,
  applyMouth,
  classifyVowel,
} from "../src/lipsync/lipsync.js";

const SR = 24000; // CHATTERBOX_SAMPLE_RATE-ish; the analyser must not assume 16k.

function resonate(x: Float32Array, freq: number, bandwidth: number, sr: number): Float32Array {
  const r = Math.exp((-Math.PI * bandwidth) / sr);
  const a1 = 2 * r * Math.cos((2 * Math.PI * freq) / sr);
  const a2 = -(r * r);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) + a2 * (y[n - 2] ?? 0);
  }
  return y;
}

function synthVowel(f1: number, f2: number, seconds: number, sr = SR, f0 = 120): Float32Array {
  const n = Math.round(seconds * sr);
  const src = new Float32Array(n);
  for (let i = 0; i < n; i += Math.round(sr / f0)) src[i] = 1;
  const y = resonate(resonate(src, f1, 80, sr), f2, 110, sr);
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  if (peak > 0) for (let i = 0; i < y.length; i++) y[i] = (y[i] ?? 0) / peak;
  return y;
}

describe("vowel classification", () => {
  it("covers exactly the five THA4 mouth vowels", () => {
    expect(VOWEL_NAMES).toEqual(["aaa", "iii", "uuu", "eee", "ooo"]);
  });

  it("gives each canonical target its own vowel as the strongest weight", () => {
    for (const name of VOWEL_NAMES) {
      const target = VOWEL_TARGETS[name];
      const w = classifyVowel(target.f1, target.f2);
      const winner = VOWEL_NAMES.reduce((best, n) => (w[n] > w[best] ? n : best), VOWEL_NAMES[0]);
      expect(winner, `expected /${name}/ to win at its own formants`).toBe(name);
    }
  });

  it("produces weights that sum to one and are non-negative", () => {
    const w = classifyVowel(600, 1200);
    const total = VOWEL_NAMES.reduce((s, n) => s + w[n], 0);
    expect(total).toBeCloseTo(1, 6);
    expect(VOWEL_NAMES.every((n) => w[n] >= 0)).toBe(true);
  });

  it("separates /i/ from /u/, which share a low F1 but differ hugely in F2", () => {
    const i = classifyVowel(270, 2290);
    const u = classifyVowel(300, 870);
    expect(i.iii).toBeGreaterThan(i.uuu);
    expect(u.uuu).toBeGreaterThan(u.iii);
  });
});

describe("analyseLipsync", () => {
  it("emits one frame per output frame interval", () => {
    const track = analyseLipsync(synthVowel(730, 1090, 1.0), SR, { fps: 30 });
    expect(track.length).toBeGreaterThanOrEqual(29);
    expect(track.length).toBeLessThanOrEqual(31);
    expect(track[0]!.time).toBeCloseTo(0, 5);
    expect(track[track.length - 1]!.time).toBeGreaterThan(0.9);
  });

  it("keeps the mouth shut through silence", () => {
    const track = analyseLipsync(new Float32Array(SR), SR, { fps: 30 });
    expect(track.every((f) => f.openness === 0)).toBe(true);
  });

  it("opens the mouth on a loud vowel and picks the right shape", () => {
    const track = analyseLipsync(synthVowel(730, 1090, 0.5), SR, { fps: 30 });
    const mid = track[Math.floor(track.length / 2)]!;
    expect(mid.openness).toBeGreaterThan(0.4);
    expect(mid.vowels.aaa).toBeGreaterThan(mid.vowels.iii);
    expect(mid.vowels.aaa).toBeGreaterThan(mid.vowels.uuu);
  });

  it("distinguishes /i/ from /a/ on real framing, not just on ideal formants", () => {
    const a = analyseLipsync(synthVowel(730, 1090, 0.5), SR, { fps: 30 });
    const i = analyseLipsync(synthVowel(270, 2290, 0.5), SR, { fps: 30 });
    const meanOf = (t: typeof a, v: "aaa" | "iii") =>
      t.reduce((s, f) => s + f.vowels[v], 0) / t.length;
    expect(meanOf(a, "aaa")).toBeGreaterThan(meanOf(i, "aaa"));
    expect(meanOf(i, "iii")).toBeGreaterThan(meanOf(a, "iii"));
  });

  it("smooths openness so the jaw cannot teleport between frames", () => {
    // Silence, then an abrupt loud vowel: attack must be ramped.
    const loud = synthVowel(730, 1090, 0.5);
    const samples = new Float32Array(SR);
    samples.set(loud.subarray(0, Math.min(loud.length, SR / 2)), SR / 2);
    const track = analyseLipsync(samples, SR, { fps: 30 });
    for (let i = 1; i < track.length; i++) {
      expect(Math.abs(track[i]!.openness - track[i - 1]!.openness)).toBeLessThan(0.6);
    }
  });

  it("returns an empty track for empty audio", () => {
    expect(analyseLipsync(new Float32Array(0), SR, { fps: 30 })).toEqual([]);
  });
});

describe("applyMouth", () => {
  it("writes only the six mouth-shape slots", () => {
    const pose = zeroPose();
    pose[POSE_INDEX.head_x] = 0.7;
    pose[POSE_INDEX.eyebrow_happy_left] = 0.4;
    applyMouth(pose, {
      time: 0,
      openness: 1,
      vowels: { aaa: 1, iii: 0, uuu: 0, eee: 0, ooo: 0 },
    });
    expect(pose[POSE_INDEX.mouth_aaa]).toBeCloseTo(1, 5);
    expect(pose[POSE_INDEX.mouth_iii]).toBe(0);
    expect(pose[POSE_INDEX.mouth_delta]).toBeGreaterThan(0);
    // untouched
    expect(pose[POSE_INDEX.head_x]).toBeCloseTo(0.7, 5);
    expect(pose[POSE_INDEX.eyebrow_happy_left]).toBeCloseTo(0.4, 5);
  });

  it("scales the vowel weights by openness", () => {
    const pose = zeroPose();
    applyMouth(pose, {
      time: 0,
      openness: 0.25,
      vowels: { aaa: 1, iii: 0, uuu: 0, eee: 0, ooo: 0 },
    });
    expect(pose[POSE_INDEX.mouth_aaa]).toBeCloseTo(0.25, 5);
  });

  it("leaves the mouth closed when openness is zero", () => {
    const pose = zeroPose();
    applyMouth(pose, {
      time: 0,
      openness: 0,
      vowels: { aaa: 0.5, iii: 0.5, uuu: 0, eee: 0, ooo: 0 },
    });
    for (const k of ["mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo"] as const) {
      expect(pose[POSE_INDEX[k]]).toBe(0);
    }
  });
});
