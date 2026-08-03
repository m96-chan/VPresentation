/**
 * Realtime renderer — draw the character in step with playing audio.
 *
 * The audio clock is the master. Audio cannot be stretched to wait for a slow
 * frame, so when inference falls behind, {@link planFrame} jumps to the frame
 * the clock is actually on and the intermediate ones are dropped. That keeps
 * lip sync correct at the cost of smoothness, which is the right trade for a
 * talking character.
 */
import type { StudentPoser } from "./student.js";
import { IMAGE_SIZE } from "./image.js";
import { planFrame, type FrameSource } from "./schedule.js";

export interface RealtimeRendererOptions {
  readonly poser: StudentPoser;
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  /** Elapsed seconds since playback began. Usually an AudioContext clock. */
  readonly clock: () => number;
  /**
   * Where the rendered character goes.
   *
   * Default is `putImageData` straight onto `canvas`. The compositor supplies
   * its own sink so the character can be scaled into a layout instead of
   * owning the whole frame.
   */
  readonly present?: (rgba: Uint8ClampedArray<ArrayBuffer>) => void;
  /** Called after each drawn frame, for stats/overlays. */
  readonly onFrame?: (frame: number, renderMs: number) => void;
}

export interface RealtimeStats {
  readonly rendered: number;
  readonly dropped: number;
  readonly meanRenderMs: number;
}

export class RealtimeRenderer {
  private running = false;
  private finished: Promise<void> = Promise.resolve();
  private rendered = 0;
  private dropped = 0;
  private totalMs = 0;
  private readonly ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

  constructor(private readonly options: RealtimeRendererOptions) {
    const ctx = options.canvas.getContext("2d");
    if (!ctx) throw new Error("could not get a 2d context for the render canvas");
    this.ctx = ctx as CanvasRenderingContext2D;
    // Only claim the canvas when we are the one painting it; with a compositor
    // the canvas is the composed frame and must keep its own size.
    if (!options.present) {
      options.canvas.width = IMAGE_SIZE;
      options.canvas.height = IMAGE_SIZE;
    }
  }

  get stats(): RealtimeStats {
    return {
      rendered: this.rendered,
      dropped: this.dropped,
      meanRenderMs: this.rendered > 0 ? this.totalMs / this.rendered : 0,
    };
  }

  /**
   * Stop the loop, resolving once the frame in flight has finished.
   *
   * Awaiting matters: starting a new renderer while the old one still has a
   * frame in the air ran two poses at once, which ORT rejects.
   */
  stop(): Promise<void> {
    this.running = false;
    return this.finished;
  }

  /**
   * Render `track` against the clock, resolving when it runs out.
   *
   * `isComplete` lets the track still be growing: while it returns false,
   * running off the end means synthesis has not caught up yet and the renderer
   * waits instead of finishing.
   */
  async play(track: FrameSource, isComplete: () => boolean = () => true): Promise<RealtimeStats> {
    this.running = true;
    let last: number | null = null;

    while (this.running) {
      const plan = planFrame(track, this.options.clock(), last, isComplete());

      if (plan.action === "done") break;
      if (plan.action === "skip") {
        await new Promise((resolve) => setTimeout(resolve, 1));
        continue;
      }

      if (last !== null && plan.frame > last + 1) this.dropped += plan.frame - last - 1;
      await this.draw(track.poseAt(plan.frame), plan.frame);
      last = plan.frame;
    }

    this.running = false;
    return this.stats;
  }

  /**
   * Drive a live engine for as long as it is running.
   *
   * This is the normal mode: the character animates continuously and speech is
   * layered in when there is any. There is no track to reach the end of, so it
   * runs until {@link stop} is called. Frames are paced to the target rate; if
   * inference is slower than that it simply runs at whatever rate it manages,
   * and the engine integrates the real elapsed time so the motion keeps its
   * character rather than going into slow motion.
   */
  async run(engine: { frameAt(time: number): Float32Array }, fps = 30): Promise<RealtimeStats> {
    this.running = true;
    let done!: () => void;
    this.finished = new Promise<void>((resolve) => {
      done = resolve;
    });
    const interval = 1000 / fps;
    let nextDue = performance.now();

    while (this.running) {
      const now = performance.now();
      if (now < nextDue) {
        await new Promise((resolve) => setTimeout(resolve, Math.min(16, nextDue - now)));
        continue;
      }
      // Never try to catch up a backlog: pace from now, not from the missed
      // deadline, or a slow frame turns into a burst of frames.
      nextDue = now + interval;

      await this.draw(engine.frameAt(this.options.clock()), this.rendered);
    }

    this.running = false;
    done();
    return this.stats;
  }

  private async draw(pose: Float32Array, frame: number): Promise<void> {
    const started = performance.now();
    const rgba = await this.options.poser.poseToRgba(pose);
    const elapsed = performance.now() - started;

    if (this.options.present) {
      this.options.present(rgba);
    } else {
      this.ctx.putImageData(new ImageData(rgba, IMAGE_SIZE, IMAGE_SIZE), 0, 0);
    }

    this.rendered++;
    this.totalMs += elapsed;
    this.options.onFrame?.(frame, elapsed);
  }
}
