import { describe, expect, it } from "vitest";
import {
  linearToSrgb,
  rgbaToThaTensor,
  srgbToLinear,
  thaTensorToRgba,
} from "../src/render/image.js";

describe("sRGB transfer function", () => {
  it("round-trips", () => {
    for (const v of [0, 0.001, 0.04, 0.05, 0.2, 0.5, 0.9, 1]) {
      expect(linearToSrgb(srgbToLinear(v))).toBeCloseTo(v, 6);
    }
  });

  it("is the piecewise IEC 61966-2-1 curve, not a plain 2.2 gamma", () => {
    // Below the knee the curve is linear with slope 1/12.92.
    expect(srgbToLinear(0.04)).toBeCloseTo(0.04 / 12.92, 9);
    // Above it, the offset power form.
    expect(srgbToLinear(0.5)).toBeCloseTo(((0.5 + 0.055) / 1.055) ** 2.4, 9);
  });

  it("pins the endpoints", () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 9);
    expect(linearToSrgb(0)).toBe(0);
    expect(linearToSrgb(1)).toBeCloseTo(1, 9);
  });
});

describe("rgbaToThaTensor", () => {
  const size = 2;

  it("produces a planar (1,4,H,W) tensor in [-1,1]", () => {
    const rgba = new Uint8ClampedArray(size * size * 4).fill(255);
    const t = rgbaToThaTensor(rgba, size);
    expect(t).toHaveLength(4 * size * size);
    expect(Math.min(...t)).toBeGreaterThanOrEqual(-1);
    expect(Math.max(...t)).toBeLessThanOrEqual(1);
  });

  it("maps opaque white to +1 on every channel", () => {
    const rgba = new Uint8ClampedArray(size * size * 4).fill(255);
    const t = rgbaToThaTensor(rgba, size);
    for (let i = 0; i < t.length; i++) expect(t[i]).toBeCloseTo(1, 5);
  });

  it("maps a fully transparent pixel to -1 everywhere (premultiplied)", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const t = rgbaToThaTensor(rgba, size);
    // Premultiplying by alpha=0 zeroes RGB, and 0 maps to -1.
    expect(t[0]).toBeCloseTo(-1, 5);
    expect(t[3 * size * size]).toBeCloseTo(-1, 5);
  });

  it("premultiplies RGB by alpha in linear space", () => {
    // Mid-grey at half alpha.
    const rgba = new Uint8ClampedArray(size * size * 4);
    rgba[0] = 128;
    rgba[1] = 128;
    rgba[2] = 128;
    rgba[3] = 128;
    const t = rgbaToThaTensor(rgba, size);
    const alpha = 128 / 255;
    const expected = srgbToLinear(128 / 255) * alpha * 2 - 1;
    expect(t[0]).toBeCloseTo(expected, 5);
  });

  it("lays channels out planar, not interleaved", () => {
    const rgba = new Uint8ClampedArray(size * size * 4);
    // Pixel 0 opaque red.
    rgba[0] = 255;
    rgba[3] = 255;
    const t = rgbaToThaTensor(rgba, size);
    const plane = size * size;
    expect(t[0]).toBeCloseTo(1, 5); // R plane, pixel 0
    expect(t[plane]).toBeCloseTo(-1, 5); // G plane, pixel 0
    expect(t[3 * plane]).toBeCloseTo(1, 5); // A plane, pixel 0
  });
});

describe("thaTensorToRgba", () => {
  it("inverts rgbaToThaTensor for opaque pixels", () => {
    const size = 4;
    const rgba = new Uint8ClampedArray(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      rgba[i * 4] = (i * 17) % 256;
      rgba[i * 4 + 1] = (i * 39) % 256;
      rgba[i * 4 + 2] = (i * 71) % 256;
      rgba[i * 4 + 3] = 255;
    }
    const back = thaTensorToRgba(rgbaToThaTensor(rgba, size), size);
    for (let i = 0; i < rgba.length; i++) {
      expect(Math.abs(back[i]! - rgba[i]!), `byte ${i}`).toBeLessThanOrEqual(1);
    }
  });

  it("round-trips semi-transparent pixels within a quantisation step", () => {
    const size = 2;
    const rgba = new Uint8ClampedArray([
      200, 100, 50, 255, 200, 100, 50, 200, 10, 220, 130, 128, 0, 0, 0, 255,
    ]);
    const back = thaTensorToRgba(rgbaToThaTensor(rgba, size), size);
    for (let i = 0; i < rgba.length; i++) {
      expect(Math.abs(back[i]! - rgba[i]!), `byte ${i}`).toBeLessThanOrEqual(2);
    }
  });

  it("emits fully transparent black where alpha is zero", () => {
    const size = 1;
    const t = new Float32Array(4).fill(-1);
    const rgba = thaTensorToRgba(t, size);
    expect([...rgba]).toEqual([0, 0, 0, 0]);
  });

  it("clamps out-of-range network output instead of wrapping", () => {
    const size = 1;
    const t = new Float32Array([5, -5, 5, 5]); // way outside [-1,1]
    const rgba = thaTensorToRgba(t, size);
    expect(rgba[0]).toBe(255);
    expect(rgba[1]).toBe(0);
    expect(rgba[3]).toBe(255);
  });
});
