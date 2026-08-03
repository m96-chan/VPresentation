import { describe, expect, it } from "vitest";
import { StudentPoser, type OrtLike, type OrtSession, type OrtTensor } from "../src/render/student.js";
import { IMAGE_SIZE } from "../src/render/image.js";
import { zeroPose } from "../src/pose/params.js";

/**
 * `StudentPoser` holds a shared scratch buffer and drives two ONNX sessions
 * that are implicitly single-flight. Two overlapping `pose()` calls produced
 * "Session mismatch" from ORT, which is how this surfaced: restarting the
 * renderer began a second run loop while the first still had a frame in the
 * air.
 */
class FakeSession implements OrtSession {
  inFlight = 0;
  maxInFlight = 0;

  constructor(
    readonly inputNames: string[],
    readonly outputNames: string[],
    private readonly outputLength: number,
  ) {}

  async run(): Promise<Record<string, OrtTensor>> {
    this.inFlight++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    await new Promise((r) => setTimeout(r, 5));
    this.inFlight--;
    return {
      [this.outputNames[0]!]: {
        data: new Float32Array(this.outputLength),
        dims: [1],
      },
    };
  }
}

function fakeOrt(face: FakeSession, body: FakeSession): OrtLike {
  let created = 0;
  return {
    InferenceSession: {
      create: async () => (created++ === 0 ? face : body),
    },
    Tensor: class {
      constructor(
        _type: string,
        readonly data: Float32Array,
        readonly dims: readonly number[],
      ) {}
    } as unknown as OrtLike["Tensor"],
  };
}

describe("StudentPoser concurrency", () => {
  it("runs one frame at a time even when called in parallel", async () => {
    const face = new FakeSession(["in0"], ["out0"], 4 * 128 * 128);
    const body = new FakeSession(["in0", "in1"], ["out0"], 4 * IMAGE_SIZE * IMAGE_SIZE);
    const poser = await StudentPoser.load({
      faceMorpher: "face",
      bodyMorpher: "body",
      ort: fakeOrt(face, body),
    });
    poser.setCharacterPixels(new Uint8ClampedArray(IMAGE_SIZE * IMAGE_SIZE * 4));

    await Promise.all([poser.pose(zeroPose()), poser.pose(zeroPose()), poser.pose(zeroPose())]);

    expect(body.maxInFlight, "overlapping body morpher runs").toBe(1);
    expect(face.maxInFlight, "overlapping face morpher runs").toBe(1);
  });

  it("keeps serving after a failed frame", async () => {
    const face = new FakeSession(["in0"], ["out0"], 4 * 128 * 128);
    const body = new FakeSession(["in0", "in1"], ["out0"], 4 * IMAGE_SIZE * IMAGE_SIZE);
    const poser = await StudentPoser.load({
      faceMorpher: "face",
      bodyMorpher: "body",
      ort: fakeOrt(face, body),
    });
    poser.setCharacterPixels(new Uint8ClampedArray(IMAGE_SIZE * IMAGE_SIZE * 4));

    // A rejected call must not wedge the queue behind it.
    await expect(poser.pose(new Float32Array(3))).rejects.toThrow();
    await expect(poser.pose(zeroPose())).resolves.toBeInstanceOf(Float32Array);
  });
});
