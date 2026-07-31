/**
 * Demo harness (rule 2): drive the real THA4 student from a real PoseTrack and
 * write the frames out as PNGs so the result can be looked at.
 *
 * This runs the *shipping* code path — `LivePoseEngine` and `StudentPoser` are
 * the same modules the browser uses — on native ORT, since Node has no
 * `navigator.gpu`. The browser gets the WebGPU EP instead; the graph and the
 * maths are identical.
 *
 * Usage (from web/):
 *   npx tsx scripts/demo-render.ts [charDir] [outDir] [frameCount]
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { PNG } from "pngjs";
// onnxruntime-**node**, not -web, and deliberately so: ORT Web's WASM EP has no
// GridSample kernel at all (it is absent from js/web/docs/operators.md and from
// the WebNN list), so `["wasm"]` fails to even create the session for the body
// morpher. In the browser the WebGPU EP does implement GridSample — see
// webgpu-operators.md, ai.onnx(16-19) — so the shipping path is WebGPU-only.
// Node has no navigator.gpu, so this harness runs the same graph on native ORT.
import * as ort from "onnxruntime-node";

import { LivePoseEngine, renderPoseFrames } from "../src/track/live.js";
import { StudentPoser } from "../src/render/student.js";
import { IMAGE_SIZE } from "../src/render/image.js";
import { POSE_INDEX } from "../src/pose/params.js";
import type { EmotionSpan } from "../src/emotion/emotion.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..");

const SAMPLE_RATE = 24000;

/** Two-pole resonator — the standard source-filter vocal-tract stand-in. */
function resonate(x: Float32Array, freq: number, bw: number, sr: number): Float32Array {
  const r = Math.exp((-Math.PI * bw) / sr);
  const a1 = 2 * r * Math.cos((2 * Math.PI * freq) / sr);
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    y[n] = (x[n] ?? 0) + a1 * (y[n - 1] ?? 0) - r * r * (y[n - 2] ?? 0);
  }
  return y;
}

/**
 * A spoken-vowel sequence: /a/ /i/ /u/ /e/ /o/, each held then released.
 *
 * VoxShot needs a browser (WebGPU + its Chatterbox weights), so this stands in
 * for its output here. The analyser only ever sees a `Float32Array` plus a
 * sample rate, which is exactly what `SynthesizedAudio.samples` is — so this
 * exercises the identical code path.
 */
function vowelSequence(): Float32Array {
  const vowels: Array<[string, number, number]> = [
    ["aaa", 730, 1090],
    ["iii", 270, 2290],
    ["uuu", 300, 870],
    ["eee", 530, 1840],
    ["ooo", 570, 840],
  ];
  const hold = 0.45;
  const gap = 0.15;
  const total = Math.round((hold + gap) * vowels.length * SAMPLE_RATE);
  const out = new Float32Array(total);

  vowels.forEach(([, f1, f2], index) => {
    const n = Math.round(hold * SAMPLE_RATE);
    const src = new Float32Array(n);
    for (let i = 0; i < n; i += Math.round(SAMPLE_RATE / 120)) src[i] = 1;
    let seg = resonate(resonate(src, f1, 80, SAMPLE_RATE), f2, 110, SAMPLE_RATE);

    let peak = 0;
    for (const v of seg) peak = Math.max(peak, Math.abs(v));
    if (peak > 0) seg = seg.map((v) => v / peak) as Float32Array;

    // Fade the edges so the analyser sees plausible onsets rather than clicks.
    const fade = Math.round(0.03 * SAMPLE_RATE);
    for (let i = 0; i < fade; i++) {
      seg[i] = (seg[i] ?? 0) * (i / fade);
      seg[n - 1 - i] = (seg[n - 1 - i] ?? 0) * (i / fade);
    }
    out.set(seg, Math.round(index * (hold + gap) * SAMPLE_RATE));
  });

  return out;
}

async function loadCharacter(pngPath: string): Promise<Uint8ClampedArray> {
  const png = PNG.sync.read(await readFile(pngPath));
  if (png.width !== IMAGE_SIZE || png.height !== IMAGE_SIZE) {
    throw new Error(`expected ${IMAGE_SIZE}x${IMAGE_SIZE}, got ${png.width}x${png.height}`);
  }
  return new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length);
}

function writePng(path: string, rgba: Uint8ClampedArray): void {
  const png = new PNG({ width: IMAGE_SIZE, height: IMAGE_SIZE });
  png.data = Buffer.from(rgba.buffer, rgba.byteOffset, rgba.length);
  writeFileSync(path, PNG.sync.write(png));
}

async function main(): Promise<void> {
  const charDir = resolve(process.argv[2] ?? join(REPO, "data/character_models/char"));
  const outDir = resolve(process.argv[3] ?? join(REPO, "out/demo"));
  const frameCount = Number(process.argv[4] ?? 10);
  mkdirSync(outDir, { recursive: true });

  console.log(`[demo] character : ${charDir}`);
  console.log(`[demo] output    : ${outDir}`);

  // --- 1. audio -> pose track ------------------------------------------------
  const samples = vowelSequence();
  const emotions: EmotionSpan[] = [
    { start: 0.0, end: 1.2, emotion: "happy" },
    { start: 1.2, end: 2.1, emotion: "surprised" },
    { start: 2.1, end: 3.0, emotion: "sad" },
  ];
  const engine = new LivePoseEngine({ fps: 30, seed: 20260731 });
  engine.beginSpeech(0, SAMPLE_RATE);
  engine.pushAudio(samples);
  for (const span of emotions) engine.addSpan(span);
  engine.endSpeech();
  const track = renderPoseFrames(
    engine,
    Math.max(1, Math.round((samples.length / SAMPLE_RATE) * 30)),
    30,
  );
  console.log(
    `[demo] audio ${(samples.length / SAMPLE_RATE).toFixed(2)}s -> ` +
      `${track.frameCount} frames @ ${track.fps}fps`,
  );

  // --- 2. load the student ---------------------------------------------------
  const t0 = performance.now();
  const poser = await StudentPoser.load({
    faceMorpher: join(charDir, "onnx/student_face_morpher.onnx"),
    bodyMorpher: join(charDir, "onnx/student_body_morpher.onnx"),
    executionProviders: ["cpu"],
    ort: ort as never,
  });
  poser.setCharacterPixels(await loadCharacter(join(charDir, "character.png")));
  console.log(`[demo] models loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // --- 3. render frames spread across the track ------------------------------
  const step = Math.max(1, Math.floor(track.frameCount / frameCount));
  const timings: number[] = [];

  for (let n = 0; n < frameCount; n++) {
    const f = Math.min(track.frameCount - 1, n * step);
    const pose = track.poseAt(f);

    const start = performance.now();
    const rgba = await poser.poseToRgba(pose);
    const ms = performance.now() - start;
    timings.push(ms);

    const name = `frame_${String(f).padStart(4, "0")}.png`;
    writePng(join(outDir, name), rgba);

    const vowels = (["mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo"] as const)
      .map((k) => `${k.slice(6)}=${pose[POSE_INDEX[k]]!.toFixed(2)}`)
      .join(" ");
    console.log(
      `[demo] ${name}  t=${(f / track.fps).toFixed(2)}s  ${ms.toFixed(0)}ms  ` +
        `${vowels}  blink=${pose[POSE_INDEX.eye_wink_left]!.toFixed(2)}  ` +
        `breath=${pose[POSE_INDEX.breathing]!.toFixed(2)}`,
    );
  }

  const mean = timings.reduce((a, b) => a + b, 0) / timings.length;
  console.log(
    `[demo] mean ${mean.toFixed(0)}ms/frame on native ORT ${poser.executionProviders.join("+")} ` +
      `(${(1000 / mean).toFixed(1)} fps) — the browser runs this on WebGPU`,
  );
  await poser.dispose();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
