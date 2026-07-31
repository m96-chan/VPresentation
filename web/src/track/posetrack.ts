/**
 * PoseTrack — the shared artefact both renderers consume.
 *
 * Audio + emotion spans in, a dense `frameCount x 45` buffer out. The
 * realtime renderer walks it against the audio clock; the offline renderer
 * walks it start to finish. Because the whole thing is a deterministic
 * function of its inputs (including the idle seed), the two agree frame for
 * frame — which is the only reason "record what you just watched" is
 * trustworthy.
 */
import {
  NUM_POSE_PARAMS,
  clampPose,
  restPose,
  type Pose,
} from "../pose/params.js";
import {
  activeEmotions,
  applyEmotion,
  blendEmotions,
  type EmotionSpan,
} from "../emotion/emotion.js";
import { applyIdle, type IdleOptions } from "../idle/idle.js";
import {
  BodyMotionIntegrator,
  SpeechDynamicsTracker,
  writeBodyMotion,
} from "../motion/body.js";
import {
  LipsyncAnalyser,
  applyMouth,
  type LipsyncOptions,
  type LipsyncSource,
  type MouthFrame,
} from "../lipsync/lipsync.js";

export interface PoseTrackOptions {
  /** Mono PCM for the utterance — `SynthesizedAudio.samples` from VoxShot. */
  readonly samples: Float32Array;
  readonly sampleRate: number;
  readonly fps?: number;
  /** Sentence-level emotion, typically one span per sentence. */
  readonly emotions?: readonly EmotionSpan[];
  /** Crossfade between adjacent emotion spans, in seconds. */
  readonly blendSeconds?: number;
  /** Idle seed. Same seed + same audio => same frames. */
  readonly seed?: number;
  readonly idle?: IdleOptions;
  /** Scale the sway / gesture / posture layers of head and body motion. */
  readonly body?: {
    readonly swayScale?: number;
    readonly gestureScale?: number;
    readonly postureScale?: number;
    /** Lower = heavier and smoother head/body motion. */
    readonly stiffness?: number;
  };
  readonly lipsyncOptions?: LipsyncOptions;
  /** Swap in a different mouth estimator (e.g. a forced aligner). */
  readonly lipsync?: LipsyncSource;
  /** Passed through to the lipsync source, for text-aware estimators. */
  readonly text?: string;
}

export class PoseTrack {
  constructor(
    readonly fps: number,
    readonly frameCount: number,
    /** Flat `frameCount * 45` buffer. */
    readonly data: Float32Array,
  ) {}

  get duration(): number {
    return this.frameCount / this.fps;
  }

  /** A view onto frame `i` — no copy, so treat it as read-only. */
  poseAt(i: number): Float32Array {
    const start = i * NUM_POSE_PARAMS;
    return this.data.subarray(start, start + NUM_POSE_PARAMS);
  }

  /** Nearest frame index for `time` (seconds), clamped to the track. */
  frameIndexAt(time: number): number {
    const i = Math.round(time * this.fps);
    if (i < 0) return 0;
    return i >= this.frameCount ? this.frameCount - 1 : i;
  }
}

/** Nearest mouth frame for a given time, or a closed mouth if there is none. */
function mouthAt(track: readonly MouthFrame[], time: number, fps: number): MouthFrame {
  if (track.length === 0) {
    return { time, openness: 0, vowels: { aaa: 0, iii: 0, uuu: 0, eee: 0, ooo: 0 } };
  }
  const i = Math.min(track.length - 1, Math.max(0, Math.round(time * fps)));
  return track[i]!;
}

/**
 * Build a track from a complete utterance.
 *
 * Delegates to {@link PoseTrackBuilder} rather than duplicating the assembly,
 * so the batch and streaming paths are the same code and cannot drift.
 */
export function buildPoseTrack(options: PoseTrackOptions): PoseTrack {
  const { samples, sampleRate } = options;
  const builder = new PoseTrackBuilder(sampleRate, {
    fps: options.fps,
    seed: options.seed,
    blendSeconds: options.blendSeconds,
    idle: options.idle,
    body: options.body,
    lipsyncOptions: options.lipsyncOptions,
  });
  builder.addSpans(...(options.emotions ?? []));

  if (options.lipsync) {
    const duration = sampleRate > 0 ? samples.length / sampleRate : 0;
    builder.pushMouthFrames(
      options.lipsync.mouthTrack(samples, sampleRate, options.text),
      duration,
    );
  } else {
    builder.push(samples);
  }
  return builder.finish();
}

export interface PoseTrackBuilderOptions {
  readonly fps?: number;
  readonly blendSeconds?: number;
  readonly seed?: number;
  readonly idle?: IdleOptions;
  readonly body?: PoseTrackOptions["body"];
  readonly lipsyncOptions?: LipsyncOptions;
}

/**
 * Incremental pose-track construction, for `VoxShot.stream()`.
 *
 * Batch synthesis means nothing happens until the whole article has been
 * rendered; streaming lets playback start after the first sentence. The catch
 * is that almost everything here is stateful — lipsync smoothing, the speech
 * envelope, and the springs that give the body inertia — so chunks cannot be
 * processed independently and concatenated. If the springs restarted at every
 * sentence the head would snap back to centre each time.
 *
 * So this carries all of it, and a test pins streamed output to be identical
 * to batch output. That equality is what makes it safe to record a realtime
 * session and get the video you just watched.
 */
export class PoseTrackBuilder {
  readonly fps: number;
  private readonly seed: number;
  private readonly blendSeconds: number;
  private readonly options: PoseTrackBuilderOptions;

  private readonly lipsync: LipsyncAnalyser;
  private readonly dynamics: SpeechDynamicsTracker;
  private readonly body: BodyMotionIntegrator;

  private readonly spans: EmotionSpan[] = [];
  private readonly pending: MouthFrame[] = [];
  private readonly scratch: Pose = restPose();
  private frames: Float32Array = new Float32Array(0);
  private count = 0;
  /** Seconds of audio accepted so far — the horizon emotion spans are known to. */
  private accepted = 0;

  constructor(sampleRate: number, options: PoseTrackBuilderOptions = {}) {
    this.fps = options.fps ?? 30;
    this.seed = options.seed ?? 0;
    this.blendSeconds = options.blendSeconds ?? 0.35;
    this.options = options;
    this.lipsync = new LipsyncAnalyser(sampleRate, {
      fps: this.fps,
      ...options.lipsyncOptions,
    });
    this.dynamics = new SpeechDynamicsTracker(this.fps);
    this.body = new BodyMotionIntegrator(this.fps, options.body);
  }

  get frameCount(): number {
    return this.count;
  }

  poseAt(i: number): Float32Array {
    const start = i * NUM_POSE_PARAMS;
    return this.frames.subarray(start, start + NUM_POSE_PARAMS);
  }

  /** Register emotion spans up front, when they are all known in advance. */
  addSpans(...spans: EmotionSpan[]): void {
    this.spans.push(...spans);
  }

  /**
   * Feed mouth frames from a custom {@link LipsyncSource} instead of audio.
   *
   * The source decides its own frame rate and count, so the track length comes
   * from `durationSeconds` and frames are resampled onto it — matching what a
   * forced aligner producing sparse frames would need.
   */
  pushMouthFrames(frames: readonly MouthFrame[], durationSeconds: number): number {
    const total = Math.max(1, Math.round(durationSeconds * this.fps));
    for (let f = 0; f < total; f++) {
      this.pending.push(mouthAt(frames, f / this.fps, this.fps));
    }
    this.accepted = durationSeconds;
    return this.drain(false);
  }

  /** Append one synthesized chunk, plus the emotion span covering it. */
  push(samples: Float32Array, span?: EmotionSpan): number {
    if (span) this.spans.push(span);
    this.pending.push(...this.lipsync.push(samples));
    this.accepted = this.lipsync.duration;
    return this.drain(false);
  }

  /** No more audio: emit everything still buffered. */
  finish(): PoseTrack {
    this.pending.push(...this.lipsync.flush());
    this.drain(true);
    return new PoseTrack(this.fps, this.count, this.frames.subarray(0, this.count * NUM_POSE_PARAMS));
  }

  private grow(): void {
    const needed = (this.count + 1) * NUM_POSE_PARAMS;
    if (needed <= this.frames.length) return;
    const next = new Float32Array(Math.max(needed, this.frames.length * 2, NUM_POSE_PARAMS * 256));
    next.set(this.frames);
    this.frames = next;
  }

  private drain(final: boolean): number {
    // An emotion span crossfades *in* before it starts, so a frame close to the
    // end of the known audio might still gain a span from the next chunk. Hold
    // those back rather than emit an expression that later turns out wrong.
    const horizon = this.accepted - this.blendSeconds / 2;

    let emitted = 0;
    while (this.pending.length > 0) {
      const mouth = this.pending[0]!;
      if (!final && mouth.time >= horizon) break;
      this.pending.shift();

      const time = this.count / this.fps;
      const dynamic = this.dynamics.step(mouth.openness);
      const active = activeEmotions(this.spans, time, { blendSeconds: this.options.blendSeconds });

      this.scratch.fill(0);
      applyEmotion(this.scratch, blendEmotions(active));
      applyMouth(this.scratch, mouth);
      writeBodyMotion(
        this.scratch,
        this.body.step({
          time,
          seed: this.seed,
          speech: dynamic.level,
          accent: dynamic.accent,
          silence: dynamic.silence,
          emotions: active,
          ...this.options.body,
        }),
      );
      applyIdle(this.scratch, time, this.seed, this.options.idle);
      clampPose(this.scratch);

      this.grow();
      this.frames.set(this.scratch, this.count * NUM_POSE_PARAMS);
      this.count++;
      emitted++;
    }
    return emitted;
  }
}
