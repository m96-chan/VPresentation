import { describe, expect, it } from "vitest";
import { LAYOUT_NAMES, composeLayout, type LayoutName } from "../src/render/layout.js";

const CANVAS = { width: 1920, height: 1080 };
const SLIDE_16_9 = 16 / 9;
const SLIDE_4_3 = 4 / 3;
const CHARACTER = 1;

function compose(name: LayoutName, slideAspect = SLIDE_16_9) {
  return composeLayout(name, { ...CANVAS, slideAspect, characterAspect: CHARACTER });
}

describe("layout presets", () => {
  it("offers the presets the compositor issue asks for", () => {
    expect(LAYOUT_NAMES).toContain("half");
    expect(LAYOUT_NAMES).toContain("picture-in-picture");
    expect(LAYOUT_NAMES).toContain("slide-only");
    expect(LAYOUT_NAMES).toContain("character-only");
  });

  it("keeps every rect inside the canvas", () => {
    for (const name of LAYOUT_NAMES) {
      for (const aspect of [SLIDE_16_9, SLIDE_4_3]) {
        const c = compose(name, aspect);
        for (const [label, rect] of [
          ["slide", c.slide],
          ["character", c.character],
        ] as const) {
          if (!rect) continue;
          expect(rect.x, `${name}/${label}.x`).toBeGreaterThanOrEqual(-0.01);
          expect(rect.y, `${name}/${label}.y`).toBeGreaterThanOrEqual(-0.01);
          expect(rect.x + rect.width, `${name}/${label} right`).toBeLessThanOrEqual(
            CANVAS.width + 0.01,
          );
          expect(rect.y + rect.height, `${name}/${label} bottom`).toBeLessThanOrEqual(
            CANVAS.height + 0.01,
          );
        }
      }
    }
  });

  it("preserves the slide's aspect ratio — letterboxed, never stretched", () => {
    for (const name of LAYOUT_NAMES) {
      for (const aspect of [SLIDE_16_9, SLIDE_4_3]) {
        const { slide } = compose(name, aspect);
        if (!slide) continue;
        expect(slide.width / slide.height, `${name} @ ${aspect}`).toBeCloseTo(aspect, 3);
      }
    }
  });

  it("keeps the character square", () => {
    for (const name of LAYOUT_NAMES) {
      const { character } = compose(name);
      if (!character) continue;
      expect(character.width / character.height, name).toBeCloseTo(1, 3);
    }
  });
});

describe("half", () => {
  it("puts the character beside the deck, not over it", () => {
    // Which side is configurable and defaults to the right, so this checks
    // that they are side by side rather than a fixed order.
    const { slidePanel, character } = compose("half");
    expect(character).toBeDefined();
    expect(slidePanel).toBeDefined();
    const apart =
      character!.x >= slidePanel!.x + slidePanel!.width - 0.01 ||
      character!.x + character!.width <= slidePanel!.x + 0.01;
    expect(apart).toBe(true);
  });

  it("gives the slide the larger share", () => {
    const { slide, character } = compose("half");
    expect(slide!.width).toBeGreaterThan(character!.width);
  });
});

describe("picture-in-picture", () => {
  it("fills the frame with the slide and insets the character", () => {
    const { slide, character } = compose("picture-in-picture");
    expect(slide!.width).toBeCloseTo(CANVAS.width, 0);
    expect(character!.width).toBeLessThan(CANVAS.width / 3);
  });

  it("tucks the character into the bottom-right", () => {
    const { character } = compose("picture-in-picture");
    expect(character!.x + character!.width).toBeGreaterThan(CANVAS.width * 0.6);
    expect(character!.y + character!.height).toBeGreaterThan(CANVAS.height * 0.6);
  });

  it("draws the character last, so it sits over the slide", () => {
    expect(compose("picture-in-picture").order).toEqual(["slide", "character"]);
  });
});

describe("single-source layouts", () => {
  it("slide-only omits the character", () => {
    const c = compose("slide-only");
    expect(c.character).toBeUndefined();
    expect(c.slide).toBeDefined();
  });

  it("character-only omits the slide", () => {
    const c = compose("character-only");
    expect(c.slide).toBeUndefined();
    expect(c.character).toBeDefined();
  });

  it("character-only has no background, so it can be keyed straight out", () => {
    // Issue #9 wants a transparent feed for OBS; painting a background here
    // would defeat that.
    expect(compose("character-only").background).toBeNull();
  });

  it("the slide layouts do paint a background", () => {
    expect(compose("half").background).not.toBeNull();
    expect(compose("slide-only").background).not.toBeNull();
  });
});

describe("odd canvases", () => {
  it("handles a portrait canvas without inverting anything", () => {
    const c = composeLayout("half", {
      width: 720,
      height: 1280,
      slideAspect: SLIDE_16_9,
      characterAspect: CHARACTER,
    });
    expect(c.slide!.width).toBeGreaterThan(0);
    expect(c.character!.width).toBeGreaterThan(0);
    expect(c.slide!.width / c.slide!.height).toBeCloseTo(SLIDE_16_9, 3);
  });

  it("survives a degenerate canvas", () => {
    const c = composeLayout("half", {
      width: 0,
      height: 0,
      slideAspect: SLIDE_16_9,
      characterAspect: CHARACTER,
    });
    for (const rect of [c.slide, c.character]) {
      if (!rect) continue;
      expect(Number.isFinite(rect.width)).toBe(true);
      expect(rect.width).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("character placement", () => {
  it("stands the character on the bottom of its box, not floating in the middle", () => {
    // A VTuber sits in a bottom corner. Centring a standing illustration
    // vertically leaves it hovering with a gap under its feet.
    for (const name of ["half", "picture-in-picture", "character-only"] as const) {
      const { character } = compose(name);
      if (!character) continue;
      expect(character.y + character.height, name).toBeGreaterThan(CANVAS.height * 0.85);
    }
  });

  it("defaults to the bottom right", () => {
    const { character } = compose("half");
    expect(character!.x + character!.width).toBeGreaterThan(CANVAS.width * 0.6);
  });

  it("can be put on the left instead", () => {
    const left = composeLayout("half", {
      ...CANVAS,
      slideAspect: SLIDE_16_9,
      characterAspect: CHARACTER,
      characterSide: "left",
    });
    expect(left.character!.x).toBeLessThan(CANVAS.width * 0.4);
    // ...and the deck moves to the other side rather than overlapping.
    expect(left.slide!.x).toBeGreaterThan(left.character!.x);
  });

  it("keeps the columns from overlapping on either side", () => {
    for (const side of ["left", "right"] as const) {
      const c = composeLayout("half", {
        ...CANVAS,
        slideAspect: SLIDE_16_9,
        characterAspect: CHARACTER,
        characterSide: side,
      });
      const character = c.character!;
      const panel = c.slidePanel!;
      const gap =
        side === "left"
          ? panel.x - (character.x + character.width)
          : character.x - (panel.x + panel.width);
      expect(gap, side).toBeGreaterThanOrEqual(-0.01);
    }
  });

  it("puts the picture-in-picture inset on the chosen side", () => {
    const left = composeLayout("picture-in-picture", {
      ...CANVAS,
      slideAspect: SLIDE_16_9,
      characterAspect: CHARACTER,
      characterSide: "left",
    });
    expect(left.character!.x).toBeLessThan(CANVAS.width * 0.2);
  });
});
