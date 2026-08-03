/**
 * Image <-> THA4 tensor conversion.
 *
 * THA4 works on planar `(1, 4, H, W)` float tensors in `[-1, 1]`, holding
 * **linear-light, alpha-premultiplied** RGBA. Getting any part of that wrong
 * produces output that looks almost right — washed out, or with dark fringes
 * around the silhouette — so the contract is pinned by tests here.
 *
 * Reference implementation: gui/coreml_poser.py (`load_thaa_image` /
 * `to_rgba_uint8`), which is what the native path already uses.
 */

export function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(v: number): number {
  const x = v < 0 ? 0 : v > 1 ? 1 : v;
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
}

/** Lookup table for the 256 possible byte values — this runs per pixel. */
const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) SRGB_TO_LINEAR[i] = srgbToLinear(i / 255);

/**
 * Interleaved RGBA bytes -> THA4 tensor `(1, 4, size, size)`.
 *
 * `pixels` is whatever `ImageData.data` / `getImageData` hands back.
 */
export function rgbaToThaTensor(
  pixels: Uint8ClampedArray | Uint8Array,
  size: number,
): Float32Array {
  const plane = size * size;
  const out = new Float32Array(4 * plane);

  for (let p = 0; p < plane; p++) {
    const i = p * 4;
    const alpha = (pixels[i + 3] ?? 0) / 255;
    // Premultiply in linear light, then map [0,1] -> [-1,1].
    out[p] = (SRGB_TO_LINEAR[pixels[i] ?? 0] ?? 0) * alpha * 2 - 1;
    out[plane + p] = (SRGB_TO_LINEAR[pixels[i + 1] ?? 0] ?? 0) * alpha * 2 - 1;
    out[2 * plane + p] = (SRGB_TO_LINEAR[pixels[i + 2] ?? 0] ?? 0) * alpha * 2 - 1;
    out[3 * plane + p] = alpha * 2 - 1;
  }
  return out;
}

/**
 * THA4 tensor `(1, 4, size, size)` -> interleaved RGBA bytes, straight alpha.
 *
 * The buffer is explicitly `ArrayBuffer`-backed (never `SharedArrayBuffer`) so
 * the result can be handed straight to `new ImageData(...)`.
 */
export function thaTensorToRgba(
  tensor: Float32Array,
  size: number,
): Uint8ClampedArray<ArrayBuffer> {
  const plane = size * size;
  const out = new Uint8ClampedArray(4 * plane);

  for (let p = 0; p < plane; p++) {
    const a = Math.min(1, Math.max(0, ((tensor[3 * plane + p] ?? -1) + 1) * 0.5));
    const i = p * 4;

    if (a < 1e-5) {
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 0;
      out[i + 3] = 0;
      continue;
    }

    for (let c = 0; c < 3; c++) {
      const premultiplied = ((tensor[c * plane + p] ?? -1) + 1) * 0.5;
      // Un-premultiply, clamp, then back to sRGB.
      const straight = Math.min(1, Math.max(0, premultiplied / a));
      out[i + c] = Math.round(linearToSrgb(straight) * 255);
    }
    out[i + 3] = Math.round(a * 255);
  }
  return out;
}

/**
 * Paste the 128x128 morphed face into the 512x512 character tensor.
 *
 * THA4's student splits the work: the face morpher renders the face alone,
 * and it is composited at rows 80..208, cols 192..320 — centred on (256, 144)
 * — before the body morpher warps the whole frame. Same offsets as
 * `gui/coreml_poser.py:64` and `crates/tha4/src/poser.rs:158`.
 */
export const FACE_ROW = 80;
export const FACE_COL = 192;
export const FACE_SIZE = 128;
export const IMAGE_SIZE = 512;

export function compositeFace(
  image: Float32Array,
  face: Float32Array,
  out = new Float32Array(image.length),
): Float32Array {
  out.set(image);
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const facePlane = FACE_SIZE * FACE_SIZE;

  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < FACE_SIZE; r++) {
      const dst = c * plane + (FACE_ROW + r) * IMAGE_SIZE + FACE_COL;
      const src = c * facePlane + r * FACE_SIZE;
      out.set(face.subarray(src, src + FACE_SIZE), dst);
    }
  }
  return out;
}

export interface Bounds {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Alpha above this counts as content; below it is antialiasing fringe. */
const OPAQUE_ENOUGH = 8;

/**
 * The bounding box of the actual artwork inside a transparent frame.
 *
 * Character images are not uniformly framed: `char.png` has 139 px of nothing
 * under the feet out of 512, while `djsaxia.png` reaches the bottom edge.
 * Standing the *frame* on the floor therefore leaves one of them hovering, so
 * the compositor anchors the content box instead.
 */
export function alphaBounds(pixels: Uint8ClampedArray | Uint8Array, size: number): Bounds {
  let minX = size;
  let minY = size;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if ((pixels[(y * size + x) * 4 + 3] ?? 0) <= OPAQUE_ENOUGH) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  // Nothing opaque: fall back to the whole frame rather than an empty box that
  // would divide by zero when anchoring.
  if (maxX < 0) return { x: 0, y: 0, width: size, height: size };
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}
