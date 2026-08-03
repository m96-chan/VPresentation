import { describe, expect, it } from "vitest";
import { alphaBounds } from "../src/render/image.js";

/** Build an RGBA buffer with an opaque rect inside a transparent frame. */
function withOpaque(size: number, box: { x: number; y: number; w: number; h: number }) {
  const px = new Uint8ClampedArray(size * size * 4);
  for (let y = box.y; y < box.y + box.h; y++) {
    for (let x = box.x; x < box.x + box.w; x++) {
      const i = (y * size + x) * 4;
      px[i] = 200;
      px[i + 3] = 255;
    }
  }
  return px;
}

describe("alphaBounds", () => {
  it("finds the opaque region", () => {
    const b = alphaBounds(withOpaque(64, { x: 10, y: 6, w: 20, h: 30 }), 64);
    expect(b).toEqual({ x: 10, y: 6, width: 20, height: 30 });
  });

  it("measures the real margins of a character with empty space below", () => {
    // char/character.png has 139 px of nothing under the feet out of 512; the
    // frame's bottom edge is not the character's.
    const b = alphaBounds(withOpaque(512, { x: 142, y: 52, w: 228, h: 321 }), 512);
    expect(b.y + b.height).toBe(373);
    expect(512 - (b.y + b.height)).toBe(139);
  });

  it("returns the full frame when everything is opaque", () => {
    const b = alphaBounds(withOpaque(32, { x: 0, y: 0, w: 32, h: 32 }), 32);
    expect(b).toEqual({ x: 0, y: 0, width: 32, height: 32 });
  });

  it("falls back to the full frame when nothing is opaque", () => {
    // A fully transparent image has no content box; anchoring to an empty rect
    // would divide by zero downstream.
    const b = alphaBounds(new Uint8ClampedArray(16 * 16 * 4), 16);
    expect(b).toEqual({ x: 0, y: 0, width: 16, height: 16 });
  });

  it("ignores nearly-transparent antialiasing fringe", () => {
    const px = new Uint8ClampedArray(32 * 32 * 4);
    // A faint halo at the very edge, real content in the middle.
    for (let i = 0; i < 32 * 32; i++) px[i * 4 + 3] = 3;
    for (let y = 10; y < 20; y++) {
      for (let x = 10; x < 20; x++) px[(y * 32 + x) * 4 + 3] = 255;
    }
    expect(alphaBounds(px, 32)).toEqual({ x: 10, y: 10, width: 10, height: 10 });
  });
});
