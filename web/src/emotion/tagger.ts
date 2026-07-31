/**
 * Text -> emotion.
 *
 * A lexicon, not a model: it runs in the browser with no extra download, is
 * deterministic, and is easy to inspect when a sentence reads wrong. RSS prose
 * is mostly neutral anyway, so the job is really "notice the few sentences
 * that carry affect" rather than fine-grained classification.
 *
 * Swap in an LLM tagger later by producing {@link EmotionSpan}s some other way
 * — nothing downstream depends on how the spans were derived.
 */
import type { EmotionName, EmotionSpan } from "./emotion.js";

const LEXICON: ReadonlyArray<readonly [EmotionName, readonly string[]]> = [
  [
    "happy",
    [
      "great", "good", "excellent", "wonderful", "brilliant", "success",
      "succeeded", "delighted", "happy", "celebrate", "win", "wins", "won",
      "improved", "record", "milestone", "launch", "launched", "love", "best",
    ],
  ],
  [
    "sad",
    [
      "sad", "sadly", "unfortunately", "cancelled", "canceled", "failure",
      "failed", "loss", "lost", "decline", "shut", "shutdown", "crashed",
      "died", "regret", "disappointing", "downturn", "layoffs", "worst",
      "terrible", "awful", "dreadful", "tragic", "grim", "bleak", "struggling",
    ],
  ],
  [
    "angry",
    [
      "slammed", "outrageous", "criticised", "criticized", "critics", "furious",
      "angry", "attack", "attacked", "blame", "blamed", "scandal", "backlash",
      "condemned", "dispute", "banned", "violation",
    ],
  ],
  [
    "surprised",
    [
      "surprising", "surprisingly", "astonishing", "astonishingly", "shocking",
      "unexpected", "unexpectedly", "suddenly", "revealed", "discovered",
      "breakthrough", "unprecedented", "nobody", "never",
    ],
  ],
  [
    "serious",
    [
      "warning", "warned", "risk", "risks", "critical", "urgent", "security",
      "vulnerability", "breach", "must", "required",
    ],
  ],
];

export interface EmotionTag {
  readonly emotion: EmotionName;
  /** 0..1 scale for the preset. */
  readonly intensity: number;
}

const WORD_SPLIT = /[^a-z']+/;

/** Classify one sentence. */
export function tagEmotion(text: string): EmotionTag {
  // Whole-word matching: "sad" must not fire on "Sadler", nor "win" on
  // "window". Splitting is enough and avoids a regex per keyword.
  const words = new Set(text.toLowerCase().split(WORD_SPLIT).filter(Boolean));

  let best: EmotionName = "neutral";
  let bestHits = 0;
  for (const [emotion, keywords] of LEXICON) {
    let hits = 0;
    for (const word of keywords) if (words.has(word)) hits++;
    if (hits > bestHits) {
      bestHits = hits;
      best = emotion;
    }
  }

  const exclaimed = text.includes("!");

  if (bestHits === 0) {
    // A bare question still deserves a raised brow.
    if (text.includes("?")) return { emotion: "surprised", intensity: 0.4 };
    return { emotion: "neutral", intensity: 1 };
  }

  // Saturating: the second and third cue add less than the first.
  let intensity = 1 - 0.55 ** bestHits;
  if (exclaimed) intensity += 0.15;
  return { emotion: best, intensity: Math.min(1, intensity) };
}
