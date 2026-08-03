/**
 * Compositor (issue #7): character + slide into one frame.
 *
 * The character is rendered at THA4's native 512x512 into an offscreen canvas
 * and drawn into the layout's rect. Doing it that way keeps inference at a
 * fixed size regardless of output resolution — scaling the *output* is free,
 * scaling the model is not.
 */
import { IMAGE_SIZE, alphaBounds, type Bounds } from "./image.js";
import { composeLayout, type Composition, type LayoutName } from "./layout.js";

export interface CompositorOptions {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  readonly layout?: LayoutName;
  readonly characterSide?: "left" | "right";
  /** Overrides the layout's own background; use `null` for chroma-key output. */
  readonly background?: string | null;
}

/** Largest rect of `aspect` centred inside `box`. */
function fitInto(
  box: { x: number; y: number; width: number; height: number },
  aspect: number,
): { x: number; y: number; width: number; height: number } {
  if (!Number.isFinite(aspect) || aspect <= 0 || box.width <= 0 || box.height <= 0) return box;
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

export class Compositor {
  private readonly ctx: CanvasRenderingContext2D;
  /** THA4's output, held at native size and scaled only on the way out. */
  private readonly characterCanvas = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  private readonly characterCtx: OffscreenCanvasRenderingContext2D;

  layout: LayoutName;
  /** Which bottom corner the presenter stands in. */
  characterSide: "left" | "right";
  /**
   * The artwork's bounding box inside the 512x512 frame.
   *
   * Character images are framed inconsistently — one has 139 px of nothing
   * under the feet — so the *content* is what gets stood on the floor, not the
   * frame. Undefined means the whole frame.
   */
  characterContent: Bounds | undefined;
  slide: ImageBitmap | undefined;
  /**
   * Region of the slide bitmap to show, in bitmap pixels.
   *
   * Undefined shows the whole page. The reading camera sets this so an A4
   * paper can be zoomed to the paragraph being spoken — letterboxed into the
   * layout's rect, never stretched.
   */
  slideViewport: { x: number; y: number; width: number; height: number } | undefined;

  constructor(private readonly options: CompositorOptions) {
    const ctx = options.canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context for the compositor");
    this.ctx = ctx as CanvasRenderingContext2D;

    const characterCtx = this.characterCanvas.getContext("2d");
    if (!characterCtx) throw new Error("no 2d context for the character buffer");
    this.characterCtx = characterCtx;

    this.layout = options.layout ?? "half";
    this.characterSide = options.characterSide ?? "right";
    this.characterContent = undefined;
  }

  get width(): number {
    return this.options.canvas.width;
  }

  get height(): number {
    return this.options.canvas.height;
  }

  /** Update the character image from a THA4 frame. */
  setCharacter(rgba: Uint8ClampedArray<ArrayBuffer>): void {
    this.characterCtx.putImageData(new ImageData(rgba, IMAGE_SIZE, IMAGE_SIZE), 0, 0);
  }

  /** Measure the artwork's extent once, from the character still. */
  measureCharacter(rgba: Uint8ClampedArray | Uint8Array): Bounds {
    this.characterContent = alphaBounds(rgba, IMAGE_SIZE);
    return this.characterContent;
  }

  /**
   * Current layout geometry, for hit-testing or overlays.
   *
   * Keyed on the *page's* aspect ratio, never the camera's viewport. Deriving
   * it from the viewport made the panel — and with it the character's position
   * — reflow on every frame as the camera zoomed: the whole layout breathed at
   * 30 fps. The panel is furniture and has to stay put; the camera moves
   * inside it.
   */
  geometry(): Composition {
    return composeLayout(this.layout, {
      width: this.width,
      height: this.height,
      slideAspect: this.slide ? this.slide.width / this.slide.height : 16 / 9,
      characterAspect: 1,
      characterSide: this.characterSide,
    });
  }

  private source(): { x: number; y: number; width: number; height: number } | undefined {
    if (!this.slide) return undefined;
    const full = { x: 0, y: 0, width: this.slide.width, height: this.slide.height };
    const v = this.slideViewport;
    if (!v || v.width <= 0 || v.height <= 0) return full;

    // Clamp into the bitmap: sampling outside it draws nothing and leaves gaps.
    const x = Math.max(0, Math.min(v.x, full.width));
    const y = Math.max(0, Math.min(v.y, full.height));
    return {
      x,
      y,
      width: Math.max(1, Math.min(v.width, full.width - x)),
      height: Math.max(1, Math.min(v.height, full.height - y)),
    };
  }

  /**
   * Draw the character so its *artwork* fills the rect and stands on its
   * bottom edge, rather than its transparent frame doing so.
   */
  private drawCharacter(rect: { x: number; y: number; width: number; height: number }): void {
    const content = this.characterContent;
    if (!content || content.width <= 0 || content.height <= 0) {
      this.ctx.drawImage(this.characterCanvas, rect.x, rect.y, rect.width, rect.height);
      return;
    }

    // Where the artwork should land: its own aspect, fitted, standing on the
    // bottom of the rect.
    const aspect = content.width / content.height;
    const byWidth = rect.width / aspect <= rect.height;
    const width = byWidth ? rect.width : rect.height * aspect;
    const height = byWidth ? rect.width / aspect : rect.height;
    const destX = rect.x + (rect.width - width) / 2;
    const destY = rect.y + rect.height - height;

    // Then place the whole frame so the content ends up there.
    const scale = width / content.width;
    this.ctx.drawImage(
      this.characterCanvas,
      destX - content.x * scale,
      destY - content.y * scale,
      IMAGE_SIZE * scale,
      IMAGE_SIZE * scale,
    );
  }

  /** Draw one composed frame. */
  draw(): void {
    const geometry = this.geometry();
    const background =
      this.options.background !== undefined ? this.options.background : geometry.background;

    this.ctx.clearRect(0, 0, this.width, this.height);
    if (background !== null) {
      this.ctx.fillStyle = background;
      this.ctx.fillRect(0, 0, this.width, this.height);
    }

    for (const layer of geometry.order) {
      if (layer === "slide" && this.slide) {
        const panel = geometry.slidePanel ?? geometry.slide;
        if (!panel) continue;
        const src = this.source()!;
        // Letterbox the visible region inside the fixed panel, so zooming
        // changes what is shown without moving the panel or stretching it.
        const dest = fitInto(panel, src.width / src.height);
        this.ctx.drawImage(
          this.slide,
          src.x,
          src.y,
          src.width,
          src.height,
          dest.x,
          dest.y,
          dest.width,
          dest.height,
        );
      } else if (layer === "character" && geometry.character) {
        this.drawCharacter(geometry.character);
      }
    }
  }
}
