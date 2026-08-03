/**
 * PDF decks (issue #5).
 *
 * Rendered with pdf.js rather than PyMuPDF / pdf2image as that issue first
 * assumed — the whole app moved into the browser, so the deck has to render
 * there too.
 *
 * Pages are rasterised on demand and cached, because re-rendering a page every
 * frame would swamp THA4's share of the GPU for no benefit: a slide is static
 * until the deck advances.
 */
import * as pdfjs from "pdfjs-dist";
import { groupBlocks, type Block, type PositionedItem } from "./blocks.js";
import type { PDFDocumentProxy } from "pdfjs-dist";

// pdf.js does its parsing in a worker. Bundled via Vite's ?url so the path
// survives the build rather than pointing at a CDN.
import workerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

export interface SlidePage {
  readonly index: number;
  readonly width: number;
  readonly height: number;
  /** Width / height, for the layout. */
  readonly aspect: number;
  /** Text blocks with positions, in reading order. */
  readonly blocks: readonly Block[];
}

/**
 * Flatten pdf.js text items into lines.
 *
 * `hasEOL` marks the item that ends a visual line. Honouring it is what keeps
 * bullets on separate lines — without it a whole slide collapses into one
 * run-on string and `pageTextToScript` has nothing to split on.
 */
function extractText(content: { items: ReadonlyArray<unknown> }): string {
  let out = "";
  for (const item of content.items) {
    if (typeof item !== "object" || item === null || !("str" in item)) continue;
    const { str, hasEOL } = item as { str: string; hasEOL?: boolean };
    out += str;
    out += hasEOL ? "\n" : "";
  }
  return out;
}

/**
 * pdf.js text items -> positioned items in top-left coordinates.
 *
 * PDF space has its origin at the bottom left and `transform` carries the
 * glyph matrix, so `[4]`/`[5]` are the baseline position and `[3]` the vertical
 * scale — which is the font size for unrotated text.
 */
function toPositioned(
  content: { items: ReadonlyArray<unknown> },
  pageHeight: number,
): PositionedItem[] {
  const out: PositionedItem[] = [];
  for (const raw of content.items) {
    if (typeof raw !== "object" || raw === null || !("str" in raw)) continue;
    const item = raw as {
      str: string;
      transform: number[];
      width: number;
      height: number;
    };
    const transform = item.transform ?? [];
    const fontSize = Math.abs(transform[3] ?? item.height ?? 10) || 10;
    const x = transform[4] ?? 0;
    const baseline = transform[5] ?? 0;
    out.push({
      str: item.str,
      x,
      // Baseline is the *bottom* of the glyphs, so the top is a font size above.
      y: pageHeight - baseline - fontSize,
      width: item.width || item.str.length * fontSize * 0.5,
      height: item.height || fontSize,
      fontSize,
    });
  }
  return out;
}

/**
 * How many rasterised pages to keep.
 *
 * These are big: an A4 page rendered wide enough for the reading camera to
 * zoom into is around 2048x2900, roughly 24 MB as an ImageBitmap. Caching a
 * whole paper without eviction was hundreds of megabytes and killed the tab.
 * Three is enough for the current page plus a step either way.
 */
const MAX_CACHED_PAGES = 3;

export class SlideDeck {
  private readonly cache = new Map<string, ImageBitmap>();

  private constructor(
    private readonly doc: PDFDocumentProxy,
    readonly pages: readonly SlidePage[],
    readonly text: readonly string[],
  ) {}

  static async load(source: ArrayBuffer | Uint8Array): Promise<SlideDeck> {
    const data = source instanceof Uint8Array ? source : new Uint8Array(source);
    const doc = await pdfjs.getDocument({ data }).promise;

    const pages: SlidePage[] = [];
    const text: string[] = [];

    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const viewport = page.getViewport({ scale: 1 });
      const content = await page.getTextContent();
      pages.push({
        index: i - 1,
        width: viewport.width,
        height: viewport.height,
        aspect: viewport.height > 0 ? viewport.width / viewport.height : 16 / 9,
        blocks: groupBlocks(toPositioned(content, viewport.height), viewport.width, viewport.height),
      });

      text.push(extractText(content));
    }

    return new SlideDeck(doc, pages, text);
  }

  get pageCount(): number {
    return this.pages.length;
  }

  /** Rasterise a page to `targetWidth` device pixels, memoised. */
  async render(index: number, targetWidth: number): Promise<ImageBitmap> {
    const page = this.pages[index];
    if (!page) throw new Error(`no page ${index} (deck has ${this.pageCount})`);

    const width = Math.max(1, Math.round(targetWidth));
    const key = `${index}@${width}`;
    const cached = this.cache.get(key);
    if (cached) {
      // Refresh recency.
      this.cache.delete(key);
      this.cache.set(key, cached);
      return cached;
    }

    const scale = width / page.width;
    const proxy = await this.doc.getPage(index + 1);
    const viewport = proxy.getViewport({ scale });

    const canvas = new OffscreenCanvas(
      Math.round(viewport.width),
      Math.round(viewport.height),
    );
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no 2d context for slide rendering");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);

    await proxy.render({ canvas, canvasContext: context, viewport } as never).promise;

    const bitmap = canvas.transferToImageBitmap();
    this.cache.set(key, bitmap);

    // Map iteration order doubles as the LRU order. Evicted bitmaps are closed
    // explicitly; dropping the reference alone leaves the GPU memory to the
    // garbage collector's discretion.
    while (this.cache.size > MAX_CACHED_PAGES) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.get(oldest)?.close();
      this.cache.delete(oldest);
    }
    return bitmap;
  }

  async destroy(): Promise<void> {
    for (const bitmap of this.cache.values()) bitmap.close();
    this.cache.clear();
    await this.doc.cleanup();
  }
}
