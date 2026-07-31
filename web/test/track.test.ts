import { describe, expect, it } from "vitest";
import { NUM_POSE_PARAMS, POSE_INDEX, POSE_RANGES } from "../src/pose/params.js";
import { buildPoseTrack } from "../src/track/posetrack.js";

const SR = 24000;

function resonate(x: Float32Array, freq: number, bw: number, sr: number): Float32Array {
  const r = Math.exp((-Math.PI * bw) / sr);
  const a1 = 2 * r * Math.cos((2 * Math.PI * freq) / sr);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) - r * r * (y[n - 2] ?? 0);
  }
  return y;
}

function speechLike(seconds: number, sr = SR): Float32Array {
  const n = Math.round(seconds * sr);
  const src = new Float32Array(n);
  for (let i = 0; i < n; i += Math.round(sr / 120)) src[i] = 1;
  const y = resonate(resonate(src, 730, 80, sr), 1090, 110, sr);
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  if (peak > 0) for (let i = 0; i < y.length; i++) y[i] = (y[i] ?? 0) / peak;
  return y;
}

describe("buildPoseTrack", () => {
  const audio = speechLike(2);

  it("produces one 45-float pose per frame for the audio duration", () => {
    const track = buildPoseTrack({ samples: audio, sampleRate: SR, fps: 30 });
    expect(track.fps).toBe(30);
    expect(track.frameCount).toBeGreaterThanOrEqual(59);
    expect(track.frameCount).toBeLessThanOrEqual(61);
    expect(track.data).toHaveLength(track.frameCount * NUM_POSE_PARAMS);
    expect(track.poseAt(0)).toHaveLength(NUM_POSE_PARAMS);
  });

  it("reports the duration it covers", () => {
    const track = buildPoseTrack({ samples: audio, sampleRate: SR, fps: 30 });
    expect(track.duration).toBeCloseTo(2, 1);
  });

  it("keeps every value inside its declared range", () => {
    const track = buildPoseTrack({
      samples: audio,
      sampleRate: SR,
      fps: 30,
      emotions: [{ start: 0, end: 2, emotion: "surprised" }],
    });
    for (let f = 0; f < track.frameCount; f++) {
      const pose = track.poseAt(f);
      for (let i = 0; i < NUM_POSE_PARAMS; i++) {
        const [lo, hi] = POSE_RANGES[i]!;
        expect(pose[i], `frame ${f} slot ${i}`).toBeGreaterThanOrEqual(lo);
        expect(pose[i], `frame ${f} slot ${i}`).toBeLessThanOrEqual(hi);
      }
    }
  });

  it("drives the mouth from the audio", () => {
    const track = buildPoseTrack({ samples: audio, sampleRate: SR, fps: 30 });
    const mid = track.poseAt(Math.floor(track.frameCount / 2));
    expect(mid[POSE_INDEX.mouth_aaa]!).toBeGreaterThan(0.2);
  });

  it("leaves the mouth shut for silent audio", () => {
    const track = buildPoseTrack({
      samples: new Float32Array(SR * 2),
      sampleRate: SR,
      fps: 30,
    });
    for (let f = 0; f < track.frameCount; f++) {
      expect(track.poseAt(f)[POSE_INDEX.mouth_aaa]).toBe(0);
    }
  });

  it("applies emotion spans on their own time range", () => {
    const track = buildPoseTrack({
      samples: audio,
      sampleRate: SR,
      fps: 30,
      emotions: [{ start: 0, end: 0.5, emotion: "happy" }],
      blendSeconds: 0.1,
    });
    expect(track.poseAt(6)[POSE_INDEX.eyebrow_happy_left]!).toBeGreaterThan(0.3);
    expect(track.poseAt(50)[POSE_INDEX.eyebrow_happy_left]!).toBe(0);
  });

  it("always animates breathing, even with no emotion and no audio", () => {
    const track = buildPoseTrack({
      samples: new Float32Array(SR),
      sampleRate: SR,
      fps: 30,
    });
    const values = Array.from({ length: track.frameCount }, (_, f) =>
      track.poseAt(f)[POSE_INDEX.breathing]!,
    );
    expect(Math.max(...values) - Math.min(...values)).toBeGreaterThan(0.3);
  });

  it("is deterministic — the offline and realtime renderers must agree", () => {
    const opts = {
      samples: audio,
      sampleRate: SR,
      fps: 30,
      seed: 7,
      emotions: [{ start: 0, end: 1, emotion: "happy" as const }],
    };
    expect(Array.from(buildPoseTrack(opts).data)).toEqual(
      Array.from(buildPoseTrack(opts).data),
    );
  });

  it("changes with the seed", () => {
    const base = { samples: audio, sampleRate: SR, fps: 30 };
    const a = buildPoseTrack({ ...base, seed: 1 }).data;
    const b = buildPoseTrack({ ...base, seed: 2 }).data;
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  it("supports a frame lookup by time", () => {
    const track = buildPoseTrack({ samples: audio, sampleRate: SR, fps: 30 });
    expect(track.frameIndexAt(0)).toBe(0);
    expect(track.frameIndexAt(1)).toBe(30);
    // Past the end clamps rather than throwing — realtime playback can overrun.
    expect(track.frameIndexAt(999)).toBe(track.frameCount - 1);
    expect(track.frameIndexAt(-5)).toBe(0);
  });

  it("honours a custom lipsync source", () => {
    const track = buildPoseTrack({
      samples: audio,
      sampleRate: SR,
      fps: 30,
      lipsync: {
        name: "test-stub",
        mouthTrack: () => [
          { time: 0, openness: 1, vowels: { aaa: 0, iii: 1, uuu: 0, eee: 0, ooo: 0 } },
        ],
      },
    });
    expect(track.poseAt(0)[POSE_INDEX.mouth_iii]).toBeCloseTo(1, 5);
    expect(track.poseAt(0)[POSE_INDEX.mouth_aaa]).toBe(0);
  });
});
