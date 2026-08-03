import { describe, expect, it } from "vitest";
import { groupBlocks, type PositionedItem } from "../src/slides/blocks.js";

const PAGE = { width: 595, height: 842 }; // A4 portrait, points

/** Build a line of text at a given position. */
function line(str: string, x: number, y: number, size = 10, width = str.length * size * 0.5): PositionedItem {
  return { str, x, y, width, height: size, fontSize: size };
}

describe("line grouping", () => {
  it("joins items that share a baseline into one line", () => {
    const blocks = groupBlocks(
      [line("Hello ", 60, 100), line("world", 95, 100)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks).toHaveLength(1);
    expect(blocks[0]!.text).toBe("Hello world");
  });

  it("does not join items on different baselines", () => {
    const blocks = groupBlocks(
      [line("first", 60, 100), line("second", 60, 400)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks.length).toBeGreaterThan(1);
  });

  it("merges consecutive lines of a paragraph into one block", () => {
    const blocks = groupBlocks(
      [line("The quick brown fox", 60, 100), line("jumps over the lazy dog", 60, 113)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks).toHaveLength(1);
    expect(blocks[0]!.text).toBe("The quick brown fox jumps over the lazy dog");
  });

  it("splits paragraphs separated by a blank line", () => {
    const blocks = groupBlocks(
      [line("First paragraph here", 60, 100), line("Second paragraph here", 60, 160)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks).toHaveLength(2);
  });
});

describe("bounding boxes", () => {
  it("covers every item in the block", () => {
    const blocks = groupBlocks(
      [line("aaa", 60, 100, 10, 40), line("bbbbbb", 60, 113, 10, 90)],
      PAGE.width,
      PAGE.height,
    );
    const b = blocks[0]!;
    expect(b.x).toBeCloseTo(60, 1);
    expect(b.y).toBeCloseTo(100, 1);
    expect(b.width).toBeCloseTo(90, 1);
    expect(b.y + b.height).toBeGreaterThanOrEqual(123);
  });

  it("never produces a zero-area box", () => {
    for (const b of groupBlocks([line("x", 60, 100)], PAGE.width, PAGE.height)) {
      expect(b.width).toBeGreaterThan(0);
      expect(b.height).toBeGreaterThan(0);
    }
  });
});

describe("two-column papers", () => {
  // A conference paper: two columns with a gutter down the middle.
  const twoColumn: PositionedItem[] = [
    line("Left column first line", 55, 100),
    line("left column second line", 55, 113),
    line("Right column first line", 310, 100),
    line("right column second line", 310, 113),
    line("Left column later", 55, 300),
    line("Right column later", 310, 300),
  ];

  it("detects the columns", () => {
    const blocks = groupBlocks(twoColumn, PAGE.width, PAGE.height);
    expect(new Set(blocks.map((b) => b.column)).size).toBe(2);
  });

  it("reads the whole left column before the right one", () => {
    // Reading a two-column paper in raw y order interleaves the columns and
    // produces nonsense, which is the single biggest failure mode here.
    const order = groupBlocks(twoColumn, PAGE.width, PAGE.height).map((b) => b.text);
    const firstRight = order.findIndex((t) => t.startsWith("Right"));
    const lastLeft = order.map((t) => t.startsWith("Left")).lastIndexOf(true);
    expect(lastLeft).toBeLessThan(firstRight);
  });

  it("treats a full-width line as spanning, not as a column", () => {
    // A title or a figure caption that crosses the gutter.
    const blocks = groupBlocks(
      [line("A Title That Spans The Whole Page Width", 55, 50, 14, 480), ...twoColumn],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks[0]!.text).toContain("Title");
    expect(blocks[0]!.column).toBe(-1);
  });

  it("leaves a single-column page in plain reading order", () => {
    const blocks = groupBlocks(
      [line("one", 60, 100, 10, 470), line("two", 60, 200, 10, 470), line("three", 60, 300, 10, 470)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks.map((b) => b.text)).toEqual(["one", "two", "three"]);
    expect(blocks.every((b) => b.column === -1)).toBe(true);
  });
});

describe("edge cases", () => {
  it("returns nothing for no items", () => {
    expect(groupBlocks([], PAGE.width, PAGE.height)).toEqual([]);
  });

  it("ignores whitespace-only items", () => {
    expect(groupBlocks([line("   ", 60, 100)], PAGE.width, PAGE.height)).toEqual([]);
  });

  it("numbers the blocks in reading order", () => {
    const blocks = groupBlocks(
      [line("b", 60, 200, 10, 470), line("a", 60, 100, 10, 470)],
      PAGE.width,
      PAGE.height,
    );
    expect(blocks.map((b) => b.order)).toEqual([0, 1]);
    expect(blocks[0]!.text).toBe("a");
  });
});

describe("spanning lines between columns", () => {
  it("places a spanning line after column content that sits above it", () => {
    // A full-width paragraph below a pair of column headings must be read
    // after them, not before. Ordering purely by "spanning first" put a body
    // paragraph ahead of its own section heading.
    const blocks = groupBlocks(
      [
        line("Title Across The Page", 55, 40, 14, 480),
        line("Left heading", 55, 150),
        line("Right heading", 310, 150),
        line("A full width paragraph that runs across both columns here", 55, 260, 10, 480),
      ],
      PAGE.width,
      PAGE.height,
    );
    const order = blocks.map((b) => b.text);
    expect(order.indexOf("Left heading")).toBeLessThan(
      order.findIndex((t) => t.startsWith("A full width")),
    );
    expect(order.indexOf("Title Across The Page")).toBe(0);
  });
});
