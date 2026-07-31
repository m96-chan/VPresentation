/**
 * The reading camera: zoom the page to the passage being spoken.
 *
 * An A4 paper letterboxed into a 16:9 frame is unreadable — the text ends up a
 * few pixels tall. Following the narration solves both problems at once: the
 * viewer can see which passage is being read, *and* it is large enough to read.
 *
 * Movement goes through the same damped spring as the character's head, for
 * the same reason: cutting between paragraphs mid-sentence is jarring, and a
 * camera with mass reads as deliberate.
 */
import { Spring } from "../motion/body.js";

export interface Box {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface FocusOptions {
  /** Extra space around the block, as a fraction of its size. */
  readonly padding?: number;
  /** Never show less than this fraction of the page width. */
  readonly minWidthFraction?: number;
}

const DEFAULTS = { padding: 0.55, minWidthFraction: 0.42 } as const;

/** Clamp a rect of the frame's aspect ratio around `block`, inside the page. */
export function focusRect(
  block: Box | null,
  page: { width: number; height: number },
  frameAspect: number,
  options: FocusOptions,
): Box {
  if (!block) return fitPage(page);

  const padding = options.padding ?? DEFAULTS.padding;
  const minWidth = (options.minWidthFraction ?? DEFAULTS.minWidthFraction) * page.width;

  // Grow the block by the padding, then to the frame's aspect ratio, then to
  // the minimum zoom — whichever ends up largest wins.
  const padded = {
    width: block.width * (1 + padding),
    height: block.height * (1 + padding * 2),
  };

  let width = Math.max(padded.width, padded.height * frameAspect, minWidth);
  let height = width / frameAspect;

  // Never zoom out past the page itself.
  if (width > page.width) {
    width = page.width;
    height = width / frameAspect;
  }
  if (height > page.height) {
    height = page.height;
    width = height * frameAspect;
  }

  const centreX = block.x + block.width / 2;
  const centreY = block.y + block.height / 2;
  return {
    x: clamp(centreX - width / 2, 0, Math.max(0, page.width - width)),
    y: clamp(centreY - height / 2, 0, Math.max(0, page.height - height)),
    width,
    height,
  };
}

/**
 * The whole page.
 *
 * Deliberately *not* forced to the frame's aspect ratio. A4 portrait simply
 * cannot be covered by a 16:9 region, so "show everything" has to be the page's
 * own shape and the compositor letterboxes it. Forcing the aspect here would
 * silently crop the page instead.
 */
function fitPage(page: { width: number; height: number }): Box {
  return { x: 0, y: 0, width: page.width, height: page.height };
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Frequency in Hz. Slow enough to read as a camera move, not a cut. */
const CAMERA_FREQUENCY = 0.9;
const CAMERA_DAMPING = 1;

export class FocusCamera {
  private readonly springs: { x: Spring; y: Spring; width: Spring; height: Spring };
  private target: Box;
  private current: Box;

  constructor(
    private readonly page: { width: number; height: number },
    private readonly frameAspect: number,
    private readonly options: FocusOptions = {},
  ) {
    this.target = fitPage(page);
    this.current = this.target;
    this.springs = {
      x: new Spring(CAMERA_FREQUENCY, CAMERA_DAMPING, this.target.x),
      y: new Spring(CAMERA_FREQUENCY, CAMERA_DAMPING, this.target.y),
      width: new Spring(CAMERA_FREQUENCY, CAMERA_DAMPING, this.target.width),
      height: new Spring(CAMERA_FREQUENCY, CAMERA_DAMPING, this.target.height),
    };
  }

  get rect(): Box {
    return this.current;
  }

  /** Aim at a block, or at the whole page with `null`. */
  focus(block: Box | null): void {
    this.target = focusRect(block, this.page, this.frameAspect, this.options);
  }

  step(dt: number): Box {
    // All four sprung independently. The aspect ratio passes through
    // intermediate values mid-move, which is fine: the compositor letterboxes
    // whatever region it is handed, and a zoom that also reshapes is exactly
    // what a camera move between a full page and a paragraph looks like.
    this.current = {
      x: this.springs.x.step(this.target.x, dt),
      y: this.springs.y.step(this.target.y, dt),
      width: this.springs.width.step(this.target.width, dt),
      height: this.springs.height.step(this.target.height, dt),
    };
    return this.current;
  }
}
