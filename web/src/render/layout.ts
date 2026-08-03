/**
 * Layout presets for the compositor (issue #7).
 *
 * Pure geometry: given a canvas and the aspect ratios of the two sources, work
 * out where each one goes. Keeping it free of canvas calls means the awkward
 * part — letterboxing, insets, not overlapping — is testable without a DOM.
 *
 * Neither source is ever stretched. A slide is typically 16:9 or 4:3 and the
 * character is square, so something always has to be letterboxed; distorting
 * the deck is never the right answer.
 */

export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export const LAYOUT_NAMES = [
  "half",
  "picture-in-picture",
  "slide-only",
  "character-only",
] as const;

export type LayoutName = (typeof LAYOUT_NAMES)[number];

export interface LayoutInput {
  readonly width: number;
  readonly height: number;
  /** Slide width / height. */
  readonly slideAspect: number;
  /** Character width / height; 1 for THA4's square output. */
  readonly characterAspect: number;
  /** Margin around and between the panels, as a fraction of the short side. */
  readonly padding?: number;
  /**
   * Which side the presenter stands on.
   *
   * A VTuber is nearly always in a bottom corner, so the character is anchored
   * to the bottom of its box in every layout and only the side varies. The
   * deck goes to the opposite side, and the idle facing follows it.
   */
  readonly characterSide?: "left" | "right";
}

export interface Composition {
  /** The slide's content rect, fitted to `slideAspect`. */
  readonly slide?: Rect;
  /**
   * The whole region reserved for the slide, before fitting.
   *
   * The compositor draws into this and letterboxes the visible region itself,
   * so the panel stays put while the reading camera zooms inside it. Fitting
   * the panel to the page instead left a portrait page in a narrow column,
   * where a zoomed-in landscape region had nowhere to grow.
   */
  readonly slidePanel?: Rect;
  readonly character?: Rect;
  /** CSS colour to clear with, or `null` to leave the frame transparent. */
  readonly background: string | null;
  /** Draw order, back to front. */
  readonly order: ReadonlyArray<"slide" | "character">;
}

const BACKGROUND = "#101014";

/** Largest rect of `aspect` that fits inside `box`, centred. */
function fit(box: Rect, aspect: number): Rect {
  if (box.width <= 0 || box.height <= 0 || !Number.isFinite(aspect) || aspect <= 0) {
    return { x: box.x, y: box.y, width: 0, height: 0 };
  }
  const byWidth = box.width / aspect <= box.height;
  const width = byWidth ? box.width : box.height * aspect;
  const height = byWidth ? box.width / aspect : box.height;
  return {
    x: box.x + (box.width - width) / 2,
    y: box.y + (box.height - height) / 2,
    width,
    height,
  };
}

/**
 * Same fit, but standing on the bottom edge of the box.
 *
 * A standing illustration centred vertically hovers with a gap under its feet.
 * Anchoring it to the bottom is what makes it read as a presenter in the
 * corner rather than a portrait pasted onto the frame.
 */
function stand(box: Rect, aspect: number): Rect {
  const r = fit(box, aspect);
  return { ...r, y: box.y + box.height - r.height };
}

export function composeLayout(name: LayoutName, input: LayoutInput): Composition {
  const { width, height, slideAspect, characterAspect } = input;
  const side = input.characterSide ?? "right";
  const pad = (input.padding ?? 0.02) * Math.min(width, height);
  const inner: Rect = {
    x: pad,
    y: pad,
    width: Math.max(0, width - pad * 2),
    height: Math.max(0, height - pad * 2),
  };

  switch (name) {
    case "half": {
      // The presenter-beside-slides arrangement: character in one bottom
      // corner at roughly a third, deck filling the rest.
      const characterShare = 0.34;
      const columnGap = pad;
      const characterWidth = Math.max(0, inner.width * characterShare - columnGap / 2);
      const slideWidth = Math.max(0, inner.width - characterWidth - columnGap);

      const characterX = side === "left" ? inner.x : inner.x + slideWidth + columnGap;
      const slideX = side === "left" ? inner.x + characterWidth + columnGap : inner.x;

      const slidePanel: Rect = {
        x: slideX,
        y: inner.y,
        width: slideWidth,
        height: inner.height,
      };
      return {
        character: stand(
          { x: characterX, y: inner.y, width: characterWidth, height: inner.height },
          characterAspect,
        ),
        slide: fit(slidePanel, slideAspect),
        slidePanel,
        background: BACKGROUND,
        order: ["slide", "character"],
      };
    }

    case "picture-in-picture": {
      // Slide edge to edge, character inset over it. No padding on the slide:
      // the deck is the content, the presenter is the overlay.
      const slidePanel: Rect = { x: 0, y: 0, width, height };
      const slide = fit(slidePanel, slideAspect);
      const size = Math.min(width, height) * 0.28;
      const character = stand(
        {
          x: side === "left" ? pad : Math.max(0, width - pad - size),
          y: Math.max(0, height - pad - size),
          width: Math.min(size, width),
          height: Math.min(size, height),
        },
        characterAspect,
      );
      return { slide, slidePanel, character, background: BACKGROUND, order: ["slide", "character"] };
    }

    case "slide-only": {
      const slidePanel: Rect = { x: 0, y: 0, width, height };
      return {
        slide: fit(slidePanel, slideAspect),
        slidePanel,
        background: BACKGROUND,
        order: ["slide"],
      };
    }

    case "character-only":
      // Transparent on purpose: this is the chroma-key / OBS feed (issue #9),
      // and painting a background would defeat keying it out.
      return {
        character: stand({ x: 0, y: 0, width, height }, characterAspect),
        background: null,
        order: ["character"],
      };
  }
}
