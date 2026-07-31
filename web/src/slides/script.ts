/**
 * Turning slide text into something worth reading aloud.
 *
 * PDF text extraction gives ragged fragments: bullet glyphs, hard line breaks
 * mid-phrase, page-number footers. Fed straight to TTS that produces run-on
 * mumbling, so each line is normalised into a sentence — which also gives the
 * synthesiser natural pauses between bullets.
 */

/** Bullet glyphs decks actually use, stripped only at the start of a line. */
const BULLET = /^[•‣▪◦●·*\-–—]\s*/;
const SENTENCE_END = /[.!?…。！？]$/;
/** A bare "12" or "3 / 20" footer. */
const PAGE_NUMBER = /^\d+(\s*[/／]\s*\d+)?$/;

/** Normalise one page's extracted text into speakable prose. */
export function pageTextToScript(text: string): string {
  const sentences: string[] = [];

  for (const raw of text.split(/[\r\n]+/)) {
    // Strip the bullet *before* collapsing, so "- foo" and "-foo" both work,
    // while a hyphen inside a word is untouched.
    const line = raw.trim().replace(BULLET, "").replace(/\s+/g, " ").trim();
    if (line.length === 0) continue;
    if (PAGE_NUMBER.test(line)) continue;

    sentences.push(SENTENCE_END.test(line) ? line : `${line}.`);
  }

  return sentences.join(" ");
}

export interface SlideScript {
  /** Zero-based page index. */
  readonly page: number;
  readonly text: string;
  /** False for pages with nothing readable — a diagram or a photo. */
  readonly speakable: boolean;
}

/**
 * One script entry per page.
 *
 * Empty pages are kept rather than dropped: the narration has to stay lined up
 * with the deck, and a picture-only slide still occupies a position in it.
 */
export function splitDeckScript(pages: readonly string[]): SlideScript[] {
  return pages.map((raw, page) => {
    const text = pageTextToScript(raw);
    return { page, text, speakable: text.length > 0 };
  });
}
