/**
 * Render a long clip and sample it, to check how far the character turns.
 * Usage (from web/):  npx tsx scripts/turn-demo.ts
 */
import { writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { PNG } from "pngjs";
import * as ort from "onnxruntime-node";
import { LivePoseEngine, renderPoseFrames } from "../src/track/live.js";
import { StudentPoser } from "../src/render/student.js";
import { IMAGE_SIZE } from "../src/render/image.js";
import { POSE_INDEX } from "../src/pose/params.js";

const SR = 24000, DIR = "../data/character_models/char";
function reson(x: Float32Array, f: number, bw: number) {
  const r = Math.exp(-Math.PI*bw/SR), a1 = 2*r*Math.cos(2*Math.PI*f/SR);
  const y = new Float32Array(x.length);
  for (let n=0;n<x.length;n++) y[n]=(x[n]??0)+a1*(y[n-1]??0)-r*r*(y[n-2]??0);
  return y;
}
// Speech, two long pauses (so the thinking gaze fires), speech again.
const PAUSES: Array<[number, number]> = [[3.2, 5.6], [8.4, 10.2]];
const secs = 12, n = secs*SR, src = new Float32Array(n);
for (let i=0;i<n;i+=Math.round(SR/120)) {
  const t = i/SR;
  if (PAUSES.some(([lo,hi]) => t >= lo && t < hi)) continue;
  if ((t*4)%1 < 0.62) src[i]=1;
}
let a = reson(reson(src,730,80),1090,110);
let pk=0; for (const v of a) pk=Math.max(pk,Math.abs(v));
a = a.map(v=>v/pk) as Float32Array;

async function main() {
  const engine = new LivePoseEngine({ fps: 30, seed: 20260731 });
  engine.beginSpeech(0, SR);
  engine.pushAudio(a);
  engine.endSpeech();
  const track = renderPoseFrames(engine, Math.round(secs * 30), 30);
  const poser = await StudentPoser.load({
    faceMorpher: `${DIR}/onnx/student_face_morpher.onnx`,
    bodyMorpher: `${DIR}/onnx/student_body_morpher.onnx`,
    executionProviders: ["cpu"], ort: ort as never });
  const still = PNG.sync.read(await readFile(`${DIR}/character.png`));
  poser.setCharacterPixels(new Uint8ClampedArray(still.data.buffer, still.data.byteOffset, still.data.length));
  
  const picks = [30, 75, 135, 150, 210, 270, 285, 330];
  const sheet = new PNG({ width: IMAGE_SIZE*picks.length, height: IMAGE_SIZE });
  for (const [k,f] of picks.entries()) {
    const pose = track.poseAt(f);
    const rgba = await poser.poseToRgba(pose);
    for (let y=0;y<IMAGE_SIZE;y++) for (let x=0;x<IMAGE_SIZE;x++) {
      const s=(y*IMAGE_SIZE+x)*4, d=(y*sheet.width + k*IMAGE_SIZE + x)*4;
      const al=(rgba[s+3]??0)/255;
      for (let c=0;c<3;c++) sheet.data[d+c]=Math.round((rgba[s+c]??0)*al+255*(1-al));
      sheet.data[d+3]=255;
    }
    console.log(`t=${(f/30).toFixed(1)}s head_x=${pose[POSE_INDEX.head_x]!.toFixed(2)} head_y=${pose[POSE_INDEX.head_y]!.toFixed(2)} iris_x=${pose[POSE_INDEX.iris_rotation_x]!.toFixed(2)} iris_y=${pose[POSE_INDEX.iris_rotation_y]!.toFixed(2)}`);
  }
  writeFileSync("../out/turn.png", PNG.sync.write(sheet));
  console.log("wrote out/turn.png");
}
main();
