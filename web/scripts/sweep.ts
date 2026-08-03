/**
 * Sweep a single pose parameter across its range and lay the results out as a
 * contact sheet.
 *
 * Used to find how far a parameter can actually be pushed before the student
 * model breaks down, and to settle which sign means which direction — neither
 * is documented anywhere in THA4.
 *
 * Usage (from web/):  npx tsx scripts/sweep.ts [charDir] [outPath] [params...]
 */
import { writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { PNG } from "pngjs";
import * as ort from "onnxruntime-node";

import { StudentPoser } from "../src/render/student.js";
import { IMAGE_SIZE } from "../src/render/image.js";
import { POSE_INDEX, zeroPose, type PoseParamName } from "../src/pose/params.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..");

const STEPS = [-1, -0.5, 0, 0.5, 1];

async function main(): Promise<void> {
  const charDir = resolve(process.argv[2] ?? join(REPO, "data/character_models/char"));
  const outPath = resolve(process.argv[3] ?? join(REPO, "out/sweep.png"));
  const params = (
    process.argv.length > 4 ? process.argv.slice(4) : ["head_x", "head_y", "neck_z", "body_y"]
  ) as PoseParamName[];

  const poser = await StudentPoser.load({
    faceMorpher: join(charDir, "onnx/student_face_morpher.onnx"),
    bodyMorpher: join(charDir, "onnx/student_body_morpher.onnx"),
    executionProviders: ["cpu"],
    ort: ort as never,
  });
  const still = PNG.sync.read(await readFile(join(charDir, "character.png")));
  poser.setCharacterPixels(
    new Uint8ClampedArray(still.data.buffer, still.data.byteOffset, still.data.length),
  );

  const sheet = new PNG({ width: IMAGE_SIZE * STEPS.length, height: IMAGE_SIZE * params.length });

  for (const [row, param] of params.entries()) {
    for (const [col, value] of STEPS.entries()) {
      const pose = zeroPose();
      pose[POSE_INDEX[param]] = value;
      const rgba = await poser.poseToRgba(pose);

      // Composite onto white so transparency does not read as black.
      const x0 = col * IMAGE_SIZE;
      const y0 = row * IMAGE_SIZE;
      for (let y = 0; y < IMAGE_SIZE; y++) {
        for (let x = 0; x < IMAGE_SIZE; x++) {
          const s = (y * IMAGE_SIZE + x) * 4;
          const d = ((y0 + y) * sheet.width + x0 + x) * 4;
          const a = (rgba[s + 3] ?? 0) / 255;
          for (let c = 0; c < 3; c++) {
            sheet.data[d + c] = Math.round((rgba[s + c] ?? 0) * a + 255 * (1 - a));
          }
          sheet.data[d + 3] = 255;
        }
      }
      console.log(`[sweep] ${param} = ${value}`);
    }
  }

  writeFileSync(outPath, PNG.sync.write(sheet));
  console.log(`[sweep] wrote ${outPath} (${sheet.width}x${sheet.height})`);
  console.log(`[sweep] rows: ${params.join(", ")}   cols: ${STEPS.join(", ")}`);
  await poser.dispose();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
