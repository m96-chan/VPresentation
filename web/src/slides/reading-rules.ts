/**
 * What not to read aloud.
 *
 * A paper is full of text that is on the page for a reader, not a listener:
 * running headers, page numbers, extracted equations that come out as symbol
 * soup, and forty bibliography entries. Reading them is worse than silence —
 * it is fluent nonsense, which is harder to sit through than a gap.
 *
 * Every decision carries a reason so the operator can see *why* something was
 * dropped and adjust, rather than wondering where a paragraph went.
 */
import type { Block } from "./blocks.js";

export interface ReadingRules {
  /** Stop reading once a bibliography heading appears. */
  readonly skipReferences: boolean;
  /** Fraction of page height at top and bottom treated as running furniture. */
  readonly marginFraction: number;
  /** Below this many words, a block is only read if it looks like a heading. */
  readonly minWords: number;
  /** Above this fraction of non-word characters, treat it as an equation. */
  readonly maxSymbolRatio: number;
  /** Font size above body text that marks a heading, as a multiplier. */
  readonly headingScale: number;
}

export const DEFAULT_RULES: ReadingRules = Object.freeze({
  skipReferences: true,
  marginFraction: 0.045,
  minWords: 2,
  maxSymbolRatio: 0.4,
  headingScale: 1.15,
});

export type SkipReason =
  | "references"
  | "margin"
  | "page-number"
  | "symbols"
  | "too-short"
  | "running-text";

export interface ReadingDecision {
  readonly block: Block;
  readonly read: boolean;
  readonly reason?: SkipReason;
}

export interface ReadingContext {
  readonly width: number;
  readonly height: number;
  readonly rules?: ReadingRules;
  /** Median body font size; inferred from the blocks when omitted. */
  readonly bodyFontSize?: number;
  /** Normalised strings identified as running headers/footers. */
  readonly runningText?: ReadonlySet<string>;
}

/** Fraction of pages a string must appear on to count as furniture. */
const RUNNING_TEXT_SHARE = 0.6;
/** Running headers are short; a repeated paragraph is still a paragraph. */
const RUNNING_TEXT_MAX_CHARS = 100;
/** ...and they sit near the page edges, not in the body. */
const RUNNING_TEXT_EDGE = 0.15;

function normalise(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Find text repeated across pages — running headers and footers.
 *
 * The margin rule alone is not enough: on a real A4 paper "Preprint. Under
 * review." sits 44 pt from the top, outside any margin fraction that would not
 * also eat the title. Being on nearly every page is the reliable signal.
 */
export function detectRunningText(
  pages: ReadonlyArray<readonly Block[]>,
  pageHeight: number,
): Set<string> {
  const out = new Set<string>();
  if (pages.length < 2) return out;

  const edge = pageHeight * RUNNING_TEXT_EDGE;
  const seen = new Map<string, number>();

  for (const blocks of pages) {
    // Only short blocks near the top or bottom edge are candidates. Without
    // those two constraints any repetition counted, and a deck that reuses a
    // paragraph lost its body text wholesale.
    const candidates = blocks.filter(
      (b) =>
        b.text.length <= RUNNING_TEXT_MAX_CHARS &&
        (b.y + b.height <= edge || b.y >= pageHeight - edge),
    );
    // Count each string once per page, so a phrase repeated within one page
    // does not masquerade as a header.
    for (const text of new Set(candidates.map((b) => normalise(b.text)))) {
      if (text.length === 0) continue;
      seen.set(text, (seen.get(text) ?? 0) + 1);
    }
  }

  const threshold = Math.max(2, Math.ceil(pages.length * RUNNING_TEXT_SHARE));
  for (const [text, count] of seen) {
    if (count >= threshold) out.add(text);
  }
  return out;
}

/** A heading that starts the back matter, not a passing mention. */
const REFERENCE_HEADING =
  /^\s*(\d+\.?\s*)?(references?|bibliography|works cited|参考文献|引用文献)\s*$/i;
const PAGE_NUMBER = /^\d+(\s*[/／]\s*\d+)?$/;

function symbolRatio(text: string): number {
  const stripped = text.replace(/\s/g, "");
  if (stripped.length === 0) return 1;
  // Letters, digits and ordinary sentence punctuation count as "word-like".
  const wordLike = stripped.replace(/[^\p{L}\p{N}.,;:!?'"()%\-]/gu, "").length;
  return 1 - wordLike / stripped.length;
}

/**
 * A single-token block that reads as a section heading rather than a scrap.
 *
 * "Conclusion" and "Fig." are both one token, so word count cannot separate
 * them; the trailing period and the short stem are what mark an abbreviation.
 * Font size would be the better signal, but it is only reliable once there is
 * enough of the page to infer a body size from.
 */
function isBareHeading(text: string): boolean {
  if (/\s/.test(text)) return false;
  if (text.endsWith(".")) return false;
  return text.replace(/[^\p{L}]/gu, "").length >= 5;
}

function medianFontSize(blocks: readonly Block[]): number {
  if (blocks.length === 0) return 10;
  const sizes = blocks.map((b) => b.fontSize).sort((a, b) => a - b);
  return sizes[Math.floor(sizes.length / 2)] ?? 10;
}

export function applyReadingRules(
  blocks: readonly Block[],
  context: ReadingContext,
): ReadingDecision[] {
  const rules = context.rules ?? DEFAULT_RULES;
  const body = context.bodyFontSize ?? medianFontSize(blocks);
  const margin = context.height * rules.marginFraction;

  let inReferences = false;
  const decisions: ReadingDecision[] = [];

  for (const block of blocks) {
    const text = block.text.trim();
    const isHeading = block.fontSize >= body * rules.headingScale;

    if (rules.skipReferences && REFERENCE_HEADING.test(text)) {
      inReferences = true;
    }
    if (inReferences) {
      decisions.push({ block, read: false, reason: "references" });
      continue;
    }

    if (context.runningText?.has(normalise(text))) {
      decisions.push({ block, read: false, reason: "running-text" });
      continue;
    }

    if (PAGE_NUMBER.test(text)) {
      decisions.push({ block, read: false, reason: "page-number" });
      continue;
    }

    // Running headers and footers live in the margins. A title sits high on the
    // page too, so the test is the block's *bottom* against the top margin —
    // real content extends past it.
    const inTopMargin = block.y + block.height <= margin;
    const inBottomMargin = block.y >= context.height - margin;
    if (rules.marginFraction > 0 && (inTopMargin || inBottomMargin)) {
      decisions.push({ block, read: false, reason: "margin" });
      continue;
    }

    if (symbolRatio(text) > rules.maxSymbolRatio) {
      decisions.push({ block, read: false, reason: "symbols" });
      continue;
    }

    const words = text.split(/\s+/).filter(Boolean).length;
    if (words < rules.minWords && !isHeading && !isBareHeading(text)) {
      decisions.push({ block, read: false, reason: "too-short" });
      continue;
    }

    decisions.push({ block, read: true });
  }

  return decisions;
}
