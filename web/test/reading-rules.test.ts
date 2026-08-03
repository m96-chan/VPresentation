import { describe, expect, it } from "vitest";
import { applyReadingRules, DEFAULT_RULES, detectRunningText } from "../src/slides/reading-rules.js";
import type { Block } from "../src/slides/blocks.js";

const PAGE = { width: 595, height: 842 };

let seq = 0;
function block(text: string, over: Partial<Block> = {}): Block {
  return {
    text,
    x: 60,
    y: 300,
    width: 470,
    height: 40,
    column: -1,
    order: seq++,
    fontSize: 10,
    ...over,
  };
}

function run(blocks: Block[], rules = DEFAULT_RULES) {
  return applyReadingRules(blocks, { ...PAGE, rules });
}

describe("what gets read", () => {
  it("keeps ordinary prose", () => {
    const [decision] = run([block("We evaluate the method on three benchmarks.")]);
    expect(decision!.read).toBe(true);
  });

  it("keeps a heading", () => {
    const [decision] = run([block("3. Experimental Setup", { fontSize: 14 })]);
    expect(decision!.read).toBe(true);
  });
});

describe("references", () => {
  it("stops reading at the bibliography heading", () => {
    const decisions = run([
      block("The results are promising."),
      block("References", { fontSize: 12 }),
      block("[1] Smith, J. Something. In Proc. of Things, 2021."),
      block("[2] Doe, A. Another. Journal of Stuff, 2022."),
    ]);
    expect(decisions.map((d) => d.read)).toEqual([true, false, false, false]);
    expect(decisions[2]!.reason).toBe("references");
  });

  it("also recognises Bibliography and 参考文献", () => {
    for (const heading of ["Bibliography", "REFERENCES", "参考文献"]) {
      const decisions = run([block("Closing remarks."), block(heading), block("[1] Smith, J. 2021.")]);
      expect(decisions[2]!.read, heading).toBe(false);
    }
  });

  it("does not trip on the word 'references' inside a sentence", () => {
    const decisions = run([
      block("This section references prior work in some detail and continues."),
      block("More body text follows here."),
    ]);
    expect(decisions.every((d) => d.read)).toBe(true);
  });
});

describe("page furniture", () => {
  it("skips a header in the top margin", () => {
    const [decision] = run([block("Preprint. Under review.", { y: 15, height: 10 })]);
    expect(decision!.read).toBe(false);
    expect(decision!.reason).toBe("margin");
  });

  it("skips a footer in the bottom margin", () => {
    const [decision] = run([block("7", { y: PAGE.height - 25, height: 10 })]);
    expect(decision!.read).toBe(false);
  });

  it("keeps body text that merely starts high on the page", () => {
    const [decision] = run([block("A Study of Things", { y: 70, height: 30, fontSize: 18 })]);
    expect(decision!.read).toBe(true);
  });

  it("skips a bare page number anywhere", () => {
    expect(run([block("12")])[0]!.read).toBe(false);
    expect(run([block("3 / 20")])[0]!.read).toBe(false);
  });
});

describe("unreadable fragments", () => {
  it("skips a block that is mostly symbols", () => {
    // Extracted equations come out as soup; reading them is worse than silence.
    const [decision] = run([block("∑ x  = ∫ ƒ(θ) dθ ≤ ∞ ⊗ ∇ ⟨ϕ⟩ ± √ 𝔼")]);
    expect(decision!.read).toBe(false);
    expect(decision!.reason).toBe("symbols");
  });

  it("keeps prose that merely contains a symbol", () => {
    const [decision] = run([block("The loss decreases by 12% when α is 0.5 in our setup.")]);
    expect(decision!.read).toBe(true);
  });

  it("skips a one-word scrap", () => {
    expect(run([block("Fig.")])[0]!.read).toBe(false);
  });

  it("keeps a bare section heading, which is only one word", () => {
    // "Conclusion" is a heading; "Fig." is a caption scrap. Both are single
    // tokens, so word count alone cannot separate them.
    expect(run([block("Conclusion")])[0]!.read).toBe(true);
    expect(run([block("Abstract")])[0]!.read).toBe(true);
  });
});

describe("configurability", () => {
  it("can be told to read the references after all", () => {
    const decisions = run([block("Some closing remarks."), block("References"), block("[1] Smith, J. Something, 2021.")], {
      ...DEFAULT_RULES,
      skipReferences: false,
    });
    expect(decisions.every((d) => d.read)).toBe(true);
  });

  it("can be told to keep the margins", () => {
    const decisions = run([block("Preprint. Under review.", { y: 15, height: 10 })], {
      ...DEFAULT_RULES,
      marginFraction: 0,
    });
    expect(decisions[0]!.read).toBe(true);
  });

  it("reports a reason for everything it drops", () => {
    const decisions = run([block("12"), block("Fig."), block("∑ ∫ ⊗ ∇ ⟨ϕ⟩ ± √")]);
    for (const d of decisions.filter((d) => !d.read)) {
      expect(d.reason, d.block.text).toBeTruthy();
    }
  });
});

describe("running headers and footers", () => {
  it("only considers short text near the page edges", () => {
    // A body paragraph repeated across pages is still body text. Treating any
    // repetition as furniture is catastrophic: on a 34-page deck it skipped
    // 705 of 818 blocks and left almost nothing to read.
    const long =
      "We evaluate the proposed approach on three standard benchmarks and report mean " +
      "accuracy over five random seeds, with ablations isolating each component.";
    const pages = Array.from({ length: 6 }, () => [
      block("Preprint. Under review.", { y: 44, height: 8 }),
      block(long, { y: 300, height: 60 }),
    ]);
    const repeated = detectRunningText(pages, PAGE.height);
    expect([...repeated]).toContain("preprint. under review.");
    expect(repeated.size).toBe(1);
  });

  it("ignores a short string repeated in the middle of the page", () => {
    const pages = Array.from({ length: 6 }, () => [block("Table 1", { y: 400, height: 10 })]);
    expect(detectRunningText(pages, PAGE.height).size).toBe(0);
  });

  it("detects text repeated across pages", () => {
    // The margin rule alone missed "Preprint. Under review." on a real A4
    // paper: at 44 pt from the top it sits outside any sane margin fraction.
    // What actually marks it as furniture is that it is on every page.
    const pages = [
      [block("Preprint. Under review.", { y: 44 }), block("Page one body text here.")],
      [block("Preprint. Under review.", { y: 44 }), block("Page two body text here.")],
      [block("Preprint. Under review.", { y: 44 }), block("Page three body text here.")],
    ];
    const repeated = detectRunningText(pages, PAGE.height);
    expect([...repeated]).toContain("preprint. under review.");
    expect([...repeated]).not.toContain("page one body text here.");
  });

  it("needs more than one page to call anything repeated", () => {
    expect(detectRunningText([[block("Only page", { y: 20 })]], PAGE.height).size).toBe(0);
  });

  it("ignores a phrase that merely appears twice in a long deck", () => {
    const pages = Array.from({ length: 10 }, (_, i) =>
      i < 2 ? [block("A common phrase.", { y: 20 })] : [block(`Unique body ${i}.`, { y: 20 })],
    );
    expect(detectRunningText(pages, PAGE.height).size).toBe(0);
  });

  it("skips the detected furniture when asked to", () => {
    const decisions = applyReadingRules([block("Preprint. Under review.", { y: 44 })], {
      ...PAGE,
      rules: DEFAULT_RULES,
      runningText: new Set(["preprint. under review."]),
    });
    expect(decisions[0]!.read).toBe(false);
    expect(decisions[0]!.reason).toBe("running-text");
  });
});
