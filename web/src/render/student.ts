/**
 * THA4 student poser on ONNX Runtime Web.
 *
 * Two nets per character (mode_14):
 *   face_morpher(pose[0..39])        -> 128x128 face
 *   body_morpher(image, pose[0..45]) -> 512x512 posed frame
 * with the face composited into the character image in between.
 *
 * The models must be exported at **opset 17**. ORT Web's WebGPU EP registers
 * GridSample for `ai.onnx(16-19)` only, and the body morpher's warp is a
 * GridSample — at opset 20 that single node drops to the CPU kernel and
 * dominates the frame. See tools/export_student.py.
 */
import { FACE_SIZE, IMAGE_SIZE, compositeFace, rgbaToThaTensor, thaTensorToRgba } from "./image.js";
import { FACE_POSE_LENGTH, NUM_POSE_PARAMS, type Pose } from "../pose/params.js";

/** The slice of onnxruntime-web this module needs; injectable for Node runs. */
export interface OrtLike {
  InferenceSession: {
    create(path: string | ArrayBufferLike, options?: unknown): Promise<OrtSession>;
  };
  Tensor: new (type: string, data: Float32Array, dims: readonly number[]) => OrtTensor;
  env?: unknown;
}

export interface OrtTensor {
  readonly data: Float32Array | ArrayLike<number>;
  readonly dims: readonly number[];
}

export interface OrtSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}

export interface StudentPoserOptions {
  /** URL or path to `student_face_morpher.onnx`. */
  readonly faceMorpher: string | ArrayBufferLike;
  /** URL or path to `student_body_morpher.onnx`. */
  readonly bodyMorpher: string | ArrayBufferLike;
  /**
   * Defaults to WebGPU only — deliberately with **no WASM fallback**.
   *
   * ORT Web's WASM EP has no GridSample kernel at all (it is absent from both
   * js/web/docs/operators.md and the WebNN list), so a browser without WebGPU
   * cannot run the body morpher: session creation fails outright rather than
   * running slowly. Listing "wasm" here would only turn a clear error into a
   * confusing one.
   */
  readonly executionProviders?: readonly string[];
  /** The onnxruntime module. Defaults to a dynamic `onnxruntime-web` import. */
  readonly ort?: OrtLike;
}

async function defaultOrt(): Promise<OrtLike> {
  return (await import("onnxruntime-web")) as unknown as OrtLike;
}

export class StudentPoser {
  private character: Float32Array | null = null;
  private readonly composite = new Float32Array(4 * IMAGE_SIZE * IMAGE_SIZE);
  /**
   * Frames are serialised through this.
   *
   * The composite buffer is shared between calls and the ONNX sessions are
   * implicitly single-flight, so two overlapping `pose()` calls corrupt each
   * other — ORT reports it as "Session mismatch". It happens for real when the
   * renderer is restarted: `stop()` only clears a flag, so the old loop still
   * has a frame in the air when the new one starts.
   */
  private queue: Promise<unknown> = Promise.resolve();

  private constructor(
    private readonly ort: OrtLike,
    private readonly face: OrtSession,
    private readonly body: OrtSession,
    readonly executionProviders: readonly string[],
  ) {}

  static async load(options: StudentPoserOptions): Promise<StudentPoser> {
    const ort = options.ort ?? (await defaultOrt());
    const executionProviders = options.executionProviders ?? ["webgpu"];
    const sessionOptions = { executionProviders };

    const [face, body] = await Promise.all([
      ort.InferenceSession.create(options.faceMorpher, sessionOptions),
      ort.InferenceSession.create(options.bodyMorpher, sessionOptions),
    ]);
    return new StudentPoser(ort, face, body, executionProviders);
  }

  /** Set the character still. `pixels` is 512x512 interleaved RGBA. */
  setCharacterPixels(pixels: Uint8ClampedArray | Uint8Array): void {
    this.character = rgbaToThaTensor(pixels, IMAGE_SIZE);
  }

  /** Set the character from an already-encoded THA4 tensor. */
  setCharacterTensor(tensor: Float32Array): void {
    if (tensor.length !== 4 * IMAGE_SIZE * IMAGE_SIZE) {
      throw new Error(`character tensor must be 4x${IMAGE_SIZE}x${IMAGE_SIZE}`);
    }
    this.character = tensor;
  }

  /** Render one frame, returning the raw THA4 tensor `(1,4,512,512)`. */
  async pose(pose: Pose): Promise<Float32Array> {
    // Chain regardless of how the previous frame ended, so one failure does
    // not wedge every frame after it.
    const run = this.queue.then(
      () => this.poseNow(pose),
      () => this.poseNow(pose),
    );
    this.queue = run.catch(() => undefined);
    return run;
  }

  private async poseNow(pose: Pose): Promise<Float32Array> {
    if (!this.character) throw new Error("call setCharacterPixels() before pose()");
    if (pose.length !== NUM_POSE_PARAMS) {
      throw new Error(`pose must have ${NUM_POSE_PARAMS} values, got ${pose.length}`);
    }

    const faceIn = this.face.inputNames[0]!;
    const facePose = new this.ort.Tensor(
      "float32",
      pose.slice(0, FACE_POSE_LENGTH),
      [1, FACE_POSE_LENGTH],
    );
    const faceOut = await this.face.run({ [faceIn]: facePose });
    const faceTensor = faceOut[this.face.outputNames[0]!]!;

    compositeFace(
      this.character,
      faceTensor.data as Float32Array,
      this.composite,
    );

    const [imgIn, poseIn] = this.body.inputNames as unknown as [string, string];
    const bodyOut = await this.body.run({
      [imgIn]: new this.ort.Tensor("float32", this.composite, [1, 4, IMAGE_SIZE, IMAGE_SIZE]),
      [poseIn]: new this.ort.Tensor("float32", pose.slice(), [1, NUM_POSE_PARAMS]),
    });
    return bodyOut[this.body.outputNames[0]!]!.data as Float32Array;
  }

  /** Render one frame as interleaved RGBA bytes, ready for `putImageData`. */
  async poseToRgba(pose: Pose): Promise<Uint8ClampedArray<ArrayBuffer>> {
    return thaTensorToRgba(await this.pose(pose), IMAGE_SIZE);
  }

  async dispose(): Promise<void> {
    await this.face.release?.();
    await this.body.release?.();
  }
}

export { FACE_SIZE, IMAGE_SIZE };
