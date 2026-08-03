import { describe, expect, it } from "vitest";
import { FocusCamera, focusRect } from "../src/slides/focus.js";

const PAGE = { width: 595, height: 842 }; // A4 portrait
const FRAME = 16 / 9;

describe("focusRect", () => {
  it("frames the block at the output aspect ratio", () => {
    const r = focusRect({ x: 60, y: 300, width: 470, height: 40 }, PAGE, FRAME, {});
    expect(r.width / r.height).toBeCloseTo(FRAME, 3);
  });

  it("actually zooms in — the whole point is legibility", () => {
    // An A4 page shown whole in a 16:9 frame is unreadable; focusing on one
    // paragraph is what makes the text large enough to follow.
    const r = focusRect({ x: 60, y: 300, width: 470, height: 40 }, PAGE, FRAME, {});
    expect(r.height).toBeLessThan(PAGE.height / 2);
  });

  it("contains the block it is focusing", () => {
    const block = { x: 60, y: 300, width: 470, height: 40 };
    const r = focusRect(block, PAGE, FRAME, {});
    expect(r.x).toBeLessThanOrEqual(block.x);
    expect(r.y).toBeLessThanOrEqual(block.y);
    expect(r.x + r.width).toBeGreaterThanOrEqual(block.x + block.width);
    expect(r.y + r.height).toBeGreaterThanOrEqual(block.y + block.height);
  });

  it("stays inside the page", () => {
    for (const block of [
      { x: 0, y: 0, width: 200, height: 20 },
      { x: 395, y: 822, width: 200, height: 20 },
      { x: 0, y: 400, width: 595, height: 20 },
    ]) {
      const r = focusRect(block, PAGE, FRAME, {});
      expect(r.x, JSON.stringify(block)).toBeGreaterThanOrEqual(-0.01);
      expect(r.y).toBeGreaterThanOrEqual(-0.01);
      expect(r.x + r.width).toBeLessThanOrEqual(PAGE.width + 0.01);
      expect(r.y + r.height).toBeLessThanOrEqual(PAGE.height + 0.01);
    }
  });

  it("never zooms past the page width", () => {
    const r = focusRect({ x: 0, y: 400, width: 595, height: 300 }, PAGE, FRAME, {});
    expect(r.width).toBeLessThanOrEqual(PAGE.width + 0.01);
  });

  it("respects a minimum zoom so a one-line block is not blown up absurdly", () => {
    const tiny = focusRect({ x: 60, y: 300, width: 40, height: 8 }, PAGE, FRAME, {});
    expect(tiny.width).toBeGreaterThan(PAGE.width * 0.3);
  });

  it("can be told to show the whole page", () => {
    // Not forced to the frame aspect: A4 portrait cannot be covered by a 16:9
    // region, so this is the page's own shape and gets letterboxed.
    const r = focusRect(null, PAGE, FRAME, {});
    expect(r.width).toBeCloseTo(PAGE.width, 1);
    expect(r.height).toBeCloseTo(PAGE.height, 1);
    expect(r.x).toBeCloseTo(0, 1);
  });
});

describe("FocusCamera", () => {
  const block = { x: 60, y: 300, width: 470, height: 40 };

  it("starts on the whole page", () => {
    const camera = new FocusCamera(PAGE, FRAME);
    expect(camera.rect.width).toBeCloseTo(PAGE.width, 1);
  });

  it("does not jump when the target changes", () => {
    const camera = new FocusCamera(PAGE, FRAME);
    const before = camera.rect;
    camera.focus(block);
    const after = camera.step(1 / 30);
    // A cut would be jarring mid-sentence; it eases across instead.
    expect(Math.abs(after.y - before.y)).toBeLessThan(PAGE.height * 0.15);
  });

  it("arrives at the target if given time", () => {
    const camera = new FocusCamera(PAGE, FRAME);
    camera.focus(block);
    for (let i = 0; i < 200; i++) camera.step(1 / 30);

    const target = focusRect(block, PAGE, FRAME, {});
    expect(camera.rect.x).toBeCloseTo(target.x, 0);
    expect(camera.rect.y).toBeCloseTo(target.y, 0);
    expect(camera.rect.width).toBeCloseTo(target.width, 0);
  });

  it("stays inside the page while moving", () => {
    // The rect is a region *of the page*; a compositor letterboxes it. Letting
    // it wander outside would sample nothing and show gaps.
    const camera = new FocusCamera(PAGE, FRAME);
    camera.focus(block);
    for (let i = 0; i < 40; i++) {
      const r = camera.step(1 / 30);
      expect(r.x).toBeGreaterThanOrEqual(-1);
      expect(r.y).toBeGreaterThanOrEqual(-1);
      expect(r.x + r.width).toBeLessThanOrEqual(PAGE.width + 1);
      expect(r.y + r.height).toBeLessThanOrEqual(PAGE.height + 1);
    }
  });

  it("is stable at a large timestep", () => {
    const camera = new FocusCamera(PAGE, FRAME);
    camera.focus(block);
    for (let i = 0; i < 20; i++) camera.step(0.7);
    expect(Number.isFinite(camera.rect.x)).toBe(true);
    expect(camera.rect.width).toBeGreaterThan(0);
  });

  it("pulls back out when focus is released", () => {
    const camera = new FocusCamera(PAGE, FRAME);
    camera.focus(block);
    for (let i = 0; i < 200; i++) camera.step(1 / 30);
    camera.focus(null);
    for (let i = 0; i < 200; i++) camera.step(1 / 30);
    expect(camera.rect.width).toBeCloseTo(PAGE.width, 0);
  });
});
