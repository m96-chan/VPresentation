import { describe, expect, it } from "vitest";
import { PoseTrack } from "../src/track/posetrack.js";
import { NUM_POSE_PARAMS } from "../src/pose/params.js";
import { planFrame } from "../src/render/schedule.js";

function track(frameCount: number, fps = 30): PoseTrack {
  return new PoseTrack(fps, frameCount, new Float32Array(frameCount * NUM_POSE_PARAMS));
}

describe("planFrame", () => {
  const t = track(90); // 3s @ 30fps

  it("renders frame 0 at the start", () => {
    expect(planFrame(t, 0, null)).toEqual({ action: "render", frame: 0 });
  });

  it("skips when the clock has not advanced past the current frame", () => {
    // 1/30s = 0.0333; at 0.01s we are still on frame 0.
    expect(planFrame(t, 0.01, 0)).toEqual({ action: "skip" });
  });

  it("advances when the clock crosses into the next frame", () => {
    expect(planFrame(t, 1 / 30, 0)).toEqual({ action: "render", frame: 1 });
  });

  it("jumps ahead rather than replaying, when rendering fell behind", () => {
    // The renderer was on frame 1 but 1.0s of audio has elapsed: go to 30,
    // do not walk frames 2..29. Dropping frames is correct here — the audio
    // is the master clock and cannot be slowed down.
    expect(planFrame(t, 1.0, 1)).toEqual({ action: "render", frame: 30 });
  });

  it("finishes once the clock runs past the track", () => {
    expect(planFrame(t, 3.5, 89)).toEqual({ action: "done" });
  });

  it("still renders the final frame before finishing", () => {
    expect(planFrame(t, 89 / 30, 88)).toEqual({ action: "render", frame: 89 });
  });

  it("clamps a negative clock to the first frame", () => {
    expect(planFrame(t, -1, null)).toEqual({ action: "render", frame: 0 });
  });

  it("treats an empty-ish track as immediately done", () => {
    expect(planFrame(track(1), 5, 0)).toEqual({ action: "done" });
  });
});
