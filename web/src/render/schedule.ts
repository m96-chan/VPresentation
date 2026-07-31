/**
 * Frame scheduling, shared by both renderers.
 *
 * Realtime playback is slaved to the audio clock: audio cannot be stretched to
 * wait for a slow frame, so when rendering falls behind the right answer is to
 * jump to the frame the clock is actually on and drop the ones in between.
 * The offline renderer never calls this — it walks every frame — which is
 * exactly the difference between the two paths.
 */

export type FramePlan =
  | { readonly action: "render"; readonly frame: number }
  | { readonly action: "skip" }
  | { readonly action: "done" };

/** The minimum a renderer needs; satisfied by both PoseTrack and its builder. */
export interface FrameSource {
  readonly fps: number;
  readonly frameCount: number;
  poseAt(i: number): Float32Array;
}

/**
 * Decide what to do at `clockSeconds`, given the frame currently on screen
 * (`lastFrame`, or `null` if nothing has been drawn yet).
 */
export function planFrame(
  track: FrameSource,
  clockSeconds: number,
  lastFrame: number | null,
  /**
   * Whether the track is finished growing. While streaming it is not: running
   * past the last frame then means "synthesis has not caught up", which is a
   * wait, not the end of the performance.
   */
  complete = true,
): FramePlan {
  const wanted = Math.floor(Math.max(0, clockSeconds) * track.fps);

  if (wanted >= track.frameCount) {
    if (!complete) return { action: "skip" };
    return lastFrame === track.frameCount - 1 || track.frameCount === 0
      ? { action: "done" }
      : { action: "render", frame: track.frameCount - 1 };
  }
  if (lastFrame !== null && wanted <= lastFrame) return { action: "skip" };
  return { action: "render", frame: wanted };
}
