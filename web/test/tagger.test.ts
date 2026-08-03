import { describe, expect, it } from "vitest";
import { tagEmotion } from "../src/emotion/tagger.js";

describe("tagEmotion", () => {
  it("falls back to neutral for plain statements", () => {
    expect(tagEmotion("The release is scheduled for Tuesday.").emotion).toBe("neutral");
  });

  it("recognises positive news", () => {
    expect(tagEmotion("Great news — the launch was a huge success!").emotion).toBe("happy");
  });

  it("recognises bad news", () => {
    expect(tagEmotion("Sadly, the project was cancelled after the failure.").emotion).toBe("sad");
  });

  it("recognises anger", () => {
    expect(tagEmotion("Critics slammed the outrageous decision.").emotion).toBe("angry");
  });

  it("recognises surprise", () => {
    expect(tagEmotion("Astonishingly, nobody had noticed the bug for years.").emotion).toBe(
      "surprised",
    );
  });

  it("treats a question as mild surprise, not neutral", () => {
    expect(tagEmotion("So what happens next?").emotion).toBe("surprised");
  });

  it("is case-insensitive and ignores punctuation glued to words", () => {
    expect(tagEmotion("WONDERFUL!").emotion).toBe("happy");
    expect(tagEmotion("wonderful.").emotion).toBe("happy");
  });

  it("does not match a keyword inside an unrelated longer word", () => {
    // "sad" must not fire on "Sadler", nor "win" on "window".
    expect(tagEmotion("Sadler opened the window.").emotion).toBe("neutral");
  });

  it("scales intensity with how many cues it found", () => {
    const one = tagEmotion("A good result.");
    const many = tagEmotion("Excellent, wonderful, brilliant news!");
    expect(many.intensity).toBeGreaterThan(one.intensity);
    expect(many.intensity).toBeLessThanOrEqual(1);
  });

  it("gives an exclamation mark a nudge without changing the emotion", () => {
    expect(tagEmotion("A good result!").intensity).toBeGreaterThan(
      tagEmotion("A good result.").intensity,
    );
  });

  it("picks the dominant emotion when cues conflict", () => {
    expect(tagEmotion("A terrible, awful, dreadful but good day.").emotion).toBe("sad");
  });
});
