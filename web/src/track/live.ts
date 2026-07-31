/**
 * The live pose engine — a character that is always animating.
 *
 * The first design here built a finite PoseTrack from a finished utterance, so
 * nothing existed until audio did: no breathing, no blinking, no presence. That
 * is backwards. A VTuber is idling on stream whether or not it is mid-sentence,
 * and speech is *layered into* that, not the thing that starts it.
 *
 * So this runs on a clock rather than on an audio buffer. Idle (breathing,
 * blinking, sway, looking around) is a pure function of time and never stops.
 * An utterance is scheduled at an absolute time, and while the clock is inside
 * it the mouth, speech gesture and emotion layers are mixed in.
 *
 * It also removes the streaming failure mode. Playback no longer depends on
 * synthesis keeping up: if the next chunk is late the character simply carries
 * on breathing instead of desynchronising.
 */
import { PoseTrack } from "./posetrack.js";
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
import { applyIdle, breathingAmount, type IdleOptions } from "../idle/idle.js";
import {
  BodyMotionIntegrator,
  SpeechDynamicsTracker,
  writeBodyMotion,
  type SpeechDynamic,
} from "../motion/body.js";
import {
  LipsyncAnalyser,
  applyMouth,
  type LipsyncOptions,
  type MouthFrame,
} from "../lipsync/lipsync.js";

export interface LivePoseEngineOptions {
  /** Rate the speech layers are analysed at. The clock itself is continuous. */
  readonly fps?: number;
  readonly seed?: number;
  readonly blendSeconds?: number;
  readonly idle?: IdleOptions;
  readonly body?: {
    readonly swayScale?: number;
    readonly gestureScale?: number;
    readonly postureScale?: number;
    readonly turnScale?: number;
    readonly stiffness?: number;
    /** Centre of the idle facing range; set from the presenter's corner. */
    readonly headingBias?: number;
  };
  readonly lipsyncOptions?: LipsyncOptions;
}

const SILENT: MouthFrame = {
  time: 0,
  openness: 0,
  vowels: { aaa: 0, iii: 0, uuu: 0, eee: 0, ooo: 0 },
};

/** An utterance placed on the engine's timeline. */
interface Utterance {
  readonly startTime: number;
  readonly analyser: LipsyncAnalyser;
  readonly dynamics: SpeechDynamicsTracker;
  /**
   * Analysed frames. Consumed in clock order, so frames the clock has passed
   * are dropped; `frameOffset` is the absolute index of `mouth[0]`.
   */
  readonly mouth: MouthFrame[];
  readonly speech: SpeechDynamic[];
  frameOffset: number;
  ended: boolean;
}

export class LivePoseEngine {
  readonly fps: number;
  private readonly seed: number;
  private readonly options: LivePoseEngineOptions;
  private readonly body: BodyMotionIntegrator;

  private readonly spans: EmotionSpan[] = [];
  private utterance: Utterance | undefined;
  private readonly pose: Pose = restPose();

  private lastTime = 0;
  private started = false;
  /**
   * Clock time speech was last audible; the thinking gaze keys off this.
   *
   * Starts at 0, not -Infinity: otherwise silence is already infinite on the
   * first frame and the character opens every session mid-stare.
   */
  private lastVoiced = 0;

  constructor(options: LivePoseEngineOptions = {}) {
    this.fps = options.fps ?? 30;
    this.seed = options.seed ?? 0;
    this.options = options;
    this.body = new BodyMotionIntegrator(this.fps, options.body);
  }

  /** Place an emotion span on the timeline. */
  addSpan(span: EmotionSpan): void {
    this.spans.push(span);
  }

  /** Open an utterance beginning at `startTime` on the engine clock. */
  beginSpeech(startTime: number, sampleRate: number): void {
    this.utterance = {
      startTime,
      analyser: new LipsyncAnalyser(sampleRate, {
        fps: this.fps,
        ...this.options.lipsyncOptions,
      }),
      dynamics: new SpeechDynamicsTracker(this.fps),
      mouth: [],
      speech: [],
      frameOffset: 0,
      ended: false,
    };
  }

  /**
   * Append synthesized audio to the open utterance, with the emotion span it
   * carries. Analysis happens here, ahead of playback, so `frameAt` stays a
   * cheap lookup.
   */
  pushAudio(samples: Float32Array, span?: EmotionSpan): void {
    const utterance = this.utterance;
    if (!utterance) throw new Error("call beginSpeech() before pushAudio()");
    if (span) this.spans.push(span);
    this.ingest(utterance, utterance.analyser.push(samples));
  }

  /** Close the utterance, flushing the frames whose windows ran past the end. */
  endSpeech(): void {
    const utterance = this.utterance;
    if (!utterance) return;
    this.ingest(utterance, utterance.analyser.flush());
    utterance.ended = true;
  }

  /** Seconds of audio scheduled so far, or 0 when nothing is queued. */
  get speechEnd(): number {
    return this.utterance ? this.utterance.startTime + this.utterance.analyser.duration : 0;
  }

  private ingest(utterance: Utterance, frames: MouthFrame[]): void {
    for (const frame of frames) {
      utterance.mouth.push(frame);
      utterance.speech.push(utterance.dynamics.step(frame.openness));
    }
    this.pruneUtterance(utterance);
  }

  /**
   * Drop analysed frames the clock has already played.
   *
   * Matters for one long utterance — the debug text box, or a page read in a
   * single go. A hundred minutes at 30 fps is ~190k frame objects otherwise.
   */
  private pruneUtterance(utterance: Utterance): void {
    const played = Math.floor((this.lastTime - utterance.startTime) * this.fps);
    const drop = played - utterance.frameOffset - this.fps; // keep a second of slack
    if (drop < this.fps * 10) return;
    utterance.mouth.splice(0, drop);
    utterance.speech.splice(0, drop);
    utterance.frameOffset += drop;
  }

  /**
   * The pose at `time` (seconds on the engine clock).
   *
   * Must be called with non-decreasing time: the body springs integrate the
   * real elapsed `dt`, which is what lets the realtime renderer drop frames
   * without the motion changing character. Backwards time is clamped rather
   * than trusted.
   */
  frameAt(time: number): Pose {
    const dt = this.started ? Math.max(0, time - this.lastTime) : 0;
    this.lastTime = Math.max(this.lastTime, time);
    this.started = true;
    this.pruneSpans(time);

    const { mouth, speech } = this.sampleSpeech(time);
    if (speech.level > 0.04) this.lastVoiced = time;

    const active = activeEmotions(this.spans, time, {
      blendSeconds: this.options.blendSeconds,
    });

    this.pose.fill(0);
    applyEmotion(this.pose, blendEmotions(active));
    applyMouth(this.pose, mouth);
    writeBodyMotion(
      this.pose,
      this.body.step(
        {
          time,
          seed: this.seed,
          speech: speech.level,
          accent: speech.accent,
          // Silence is measured on the engine clock, not inside an utterance,
          // so an idling character reads as thinking rather than as merely
          // having no audio loaded.
          silence: Math.max(0, time - this.lastVoiced),
          breath: breathingAmount(time),
          emotions: active,
          ...this.options.body,
        },
        dt,
      ),
    );
    applyIdle(this.pose, time, this.seed, this.options.idle);
    return clampPose(this.pose);
  }

  /**
   * Drop spans the clock has passed.
   *
   * They can never affect another frame, and `activeEmotions` walks the whole
   * list every frame — so keeping them makes per-frame cost grow with the
   * length of the presentation as well as leaking.
   */
  private pruneSpans(time: number): void {
    if (this.spans.length < 32) return;
    const horizon = time - (this.options.blendSeconds ?? 0.35);
    let keep = 0;
    for (const span of this.spans) {
      if (span.end >= horizon) this.spans[keep++] = span;
    }
    this.spans.length = keep;
  }

  private sampleSpeech(time: number): { mouth: MouthFrame; speech: SpeechDynamic } {
    const utterance = this.utterance;
    const silent = { mouth: SILENT, speech: { level: 0, accent: 0, silence: 0 } };
    if (!utterance) return silent;

    const offset = time - utterance.startTime;
    if (offset < 0) return silent;

    const index = Math.round(offset * this.fps) - utterance.frameOffset;
    if (index < 0 || index >= utterance.mouth.length) return silent;

    return {
      mouth: utterance.mouth[index] ?? SILENT,
      speech: utterance.speech[index] ?? silent.speech,
    };
  }
}

/**
 * Step a live engine at a fixed rate into a finite buffer.
 *
 * The offline renderer wants every frame at an exact cadence, which is just
 * this engine driven by a synthetic clock — so both paths run the same code
 * and a pre-render matches what the live view was doing.
 */
export function renderPoseFrames(
  engine: LivePoseEngine,
  frameCount: number,
  fps: number,
): PoseTrack {
  const data = new Float32Array(frameCount * NUM_POSE_PARAMS);
  for (let f = 0; f < frameCount; f++) {
    data.set(engine.frameAt(f / fps), f * NUM_POSE_PARAMS);
  }
  return new PoseTrack(fps, frameCount, data);
}
