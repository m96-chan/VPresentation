import { describe, expect, it } from "vitest";
import { pageTextToScript, splitDeckScript } from "../src/slides/script.js";

describe("pageTextToScript", () => {
  it("joins bullet fragments into speakable sentences", () => {
    const text = "Roadmap\n• Ship the renderer\n• Fix the lip sync\n• Write the docs";
    const script = pageTextToScript(text);
    expect(script).not.toContain("•");
    expect(script).toContain("Roadmap");
    expect(script).toContain("Ship the renderer");
  });

  it("gives every bullet a sentence ending, so TTS pauses between them", () => {
    const script = pageTextToScript("• one\n• two");
    expect(script).toBe("one. two.");
  });

  it("leaves already-punctuated lines alone", () => {
    expect(pageTextToScript("It works! Does it?")).toBe("It works! Does it?");
  });

  it("collapses the whitespace PDFs are full of", () => {
    expect(pageTextToScript("A   b\n\n\n   c")).toBe("A b. c.");
  });

  it("drops a bare page number", () => {
    // Slide footers are noise; reading "twelve" mid-sentence is worse than
    // silence.
    expect(pageTextToScript("Findings\n12")).toBe("Findings.");
  });

  it("drops a lone slide-count footer", () => {
    expect(pageTextToScript("Summary\n3 / 20")).toBe("Summary.");
  });

  it("strips the bullet glyphs decks actually use", () => {
    for (const glyph of ["•", "‣", "▪", "◦", "-", "–", "—", "*"]) {
      expect(pageTextToScript(`${glyph} hello`), glyph).toBe("hello.");
    }
  });

  it("keeps a hyphen inside a word", () => {
    expect(pageTextToScript("state-of-the-art results")).toBe("state-of-the-art results.");
  });

  it("returns empty for a page with nothing readable", () => {
    expect(pageTextToScript("   \n \n 7 ")).toBe("");
    expect(pageTextToScript("")).toBe("");
  });

  it("does not run two pages' worth of text together", () => {
    const a = pageTextToScript("End of one");
    expect(a.endsWith(".")).toBe(true);
  });
});

describe("splitDeckScript", () => {
  it("gives one entry per page, in order", () => {
    const pages = ["Title", "• first\n• second", "Thanks"];
    const script = splitDeckScript(pages);
    expect(script).toHaveLength(3);
    expect(script[0]!.page).toBe(0);
    expect(script[2]!.page).toBe(2);
  });

  it("keeps empty pages, so page numbers still line up", () => {
    // A picture-only slide has nothing to say, but skipping it would
    // desynchronise the deck from the narration.
    const script = splitDeckScript(["Intro", "   ", "Outro"]);
    expect(script).toHaveLength(3);
    expect(script[1]!.text).toBe("");
  });

  it("marks which pages have anything to read", () => {
    const script = splitDeckScript(["Intro", "  ", "Outro"]);
    expect(script.map((s) => s.speakable)).toEqual([true, false, true]);
  });

  it("handles an empty deck", () => {
    expect(splitDeckScript([])).toEqual([]);
  });
});
