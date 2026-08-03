/**
 * Text items -> readable blocks with positions.
 *
 * Reading a page aloud needs more than its text: to focus on the passage being
 * spoken, each block has to carry where it is. An A4 paper scaled to fit a
 * 16:9 frame is unreadable, so the viewport zooms to the current block instead
 * — which only works if blocks have boxes.
 *
 * The hard part is **reading order**. A two-column paper walked in raw y order
 * interleaves the columns and produces nonsense, which is the worst failure
 * available here: it sounds fluent and is complete gibberish.
 *
 * Coordinates are top-left origin, in PDF points.
 */

export interface PositionedItem {
  readonly str: string;
  /** Left edge. */
  readonly x: number;
  /** Top edge. */
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly fontSize: number;
}

export interface Block {
  readonly text: string;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  /** Column index, or -1 for a line that spans the page. */
  readonly column: number;
  /** Position in reading order. */
  readonly order: number;
  readonly fontSize: number;
}

interface Line {
  items: PositionedItem[];
  x: number;
  y: number;
  right: number;
  bottom: number;
  fontSize: number;
}

/** Items on the same baseline, within this fraction of the font size. */
const BASELINE_TOLERANCE = 0.6;
/** A vertical gap larger than this many line heights starts a new block. */
const PARAGRAPH_GAP = 1.8;

function toLines(items: readonly PositionedItem[]): Line[] {
  const usable = items.filter((i) => i.str.trim().length > 0);
  if (usable.length === 0) return [];

  const sorted = [...usable].sort((a, b) => a.y - b.y || a.x - b.x);
  const lines: Line[] = [];

  for (const item of sorted) {
    const tolerance = Math.max(1, item.fontSize * BASELINE_TOLERANCE);
    // Only merge into a line that is still open at this y — a two-column page
    // has two lines live at the same height.
    const host = lines.find(
      (l) =>
        Math.abs(l.y - item.y) <= tolerance &&
        item.x >= l.x - tolerance &&
        item.x <= l.right + item.fontSize * 3,
    );

    if (host) {
      host.items.push(item);
      host.x = Math.min(host.x, item.x);
      host.right = Math.max(host.right, item.x + item.width);
      host.bottom = Math.max(host.bottom, item.y + item.height);
      host.fontSize = Math.max(host.fontSize, item.fontSize);
    } else {
      lines.push({
        items: [item],
        x: item.x,
        y: item.y,
        right: item.x + item.width,
        bottom: item.y + item.height,
        fontSize: item.fontSize,
      });
    }
  }

  for (const l of lines) l.items.sort((a, b) => a.x - b.x);
  return lines;
}

/**
 * Find the gutter of a two-column layout.
 *
 * Looks for a vertical band near the middle that no line crosses. Returns the
 * band's centre, or `null` for a single-column page.
 */
function findGutter(lines: readonly Line[], pageWidth: number): number | null {
  if (lines.length < 4) return null;

  const centre = pageWidth / 2;
  const band = pageWidth * 0.08;

  // Lines that would straddle a centre gutter: titles, captions, wide figures.
  const straddling = lines.filter((l) => l.x < centre - band && l.right > centre + band);
  // A couple of spanning lines is normal (title, section headers); a page made
  // of them is single-column.
  if (straddling.length > lines.length * 0.35) return null;

  const left = lines.filter((l) => l.right <= centre + band).length;
  const right = lines.filter((l) => l.x >= centre - band).length;
  if (left < 2 || right < 2) return null;

  return centre;
}

function toText(line: Line): string {
  return line.items
    .map((i) => i.str)
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

/** Merge vertically adjacent lines of one column into paragraph blocks. */
function toBlocks(lines: readonly Line[], column: number, startOrder: number): Block[] {
  const blocks: Block[] = [];
  let current: Line[] = [];

  const flush = () => {
    if (current.length === 0) return;
    const x = Math.min(...current.map((l) => l.x));
    const y = Math.min(...current.map((l) => l.y));
    const right = Math.max(...current.map((l) => l.right));
    const bottom = Math.max(...current.map((l) => l.bottom));
    const text = current.map(toText).filter(Boolean).join(" ");
    if (text.length > 0) {
      blocks.push({
        text,
        x,
        y,
        width: Math.max(1, right - x),
        height: Math.max(1, bottom - y),
        column,
        order: startOrder + blocks.length,
        fontSize: Math.max(...current.map((l) => l.fontSize)),
      });
    }
    current = [];
  };

  for (const line of [...lines].sort((a, b) => a.y - b.y)) {
    const previous = current[current.length - 1];
    if (previous) {
      const gap = line.y - previous.bottom;
      const sameParagraph =
        gap <= previous.fontSize * PARAGRAPH_GAP &&
        // A jump in size means a heading, not a continuation.
        Math.abs(line.fontSize - previous.fontSize) < previous.fontSize * 0.3;
      if (!sameParagraph) flush();
    }
    current.push(line);
  }
  flush();

  return blocks;
}

/**
 * Group positioned text items into blocks, in reading order.
 *
 * Column assignment is strict and mutually exclusive: a line is *left* only if
 * it ends before the gutter, *right* only if it starts after it, and spanning
 * otherwise. An earlier version used an overlapping tolerance band on both
 * tests, so a centred line (a running header, an author list) matched both and
 * was emitted twice — and therefore read aloud twice.
 *
 * Spanning lines also act as horizontal rules that divide the page into bands.
 * Hoisting them all to the front instead put a title's body text before its
 * own heading; ordering by (band, column, y) is what a person actually does.
 */
export function groupBlocks(
  items: readonly PositionedItem[],
  pageWidth: number,
  pageHeight: number,
): Block[] {
  void pageHeight;
  const lines = toLines(items);
  if (lines.length === 0) return [];

  const gutter = findGutter(lines, pageWidth);
  if (gutter === null) {
    return toBlocks(lines, -1, 0).map((b, order) => ({ ...b, order }));
  }

  const columnOf = (l: Line): number => {
    if (l.right <= gutter) return 0;
    if (l.x >= gutter) return 1;
    return -1;
  };

  const spanning = lines.filter((l) => columnOf(l) === -1).sort((a, b) => a.y - b.y);
  const bandOf = (y: number): number => spanning.filter((s) => s.y < y).length;

  // Each band contributes: its spanning lines first, then the left column in
  // full, then the right.
  const groups: Array<{ band: number; rank: number; lines: Line[] }> = [];
  const push = (band: number, rank: number, line: Line) => {
    const existing = groups.find((g) => g.band === band && g.rank === rank);
    if (existing) existing.lines.push(line);
    else groups.push({ band, rank, lines: [line] });
  };

  // Column content first, so each band knows where its columns start.
  const columnTop = new Map<number, number>();
  for (const line of lines) {
    const column = columnOf(line);
    if (column === -1) continue;
    const band = bandOf(line.y);
    push(band, column, line);
    columnTop.set(band, Math.min(columnTop.get(band) ?? Infinity, line.y));
  }

  // A spanning line goes before its band's columns if it sits above them, and
  // after if below. Always putting spanning lines first read a full-width
  // paragraph ahead of the headings it follows.
  for (const line of lines) {
    if (columnOf(line) !== -1) continue;
    const band = bandOf(line.y);
    const top = columnTop.get(band);
    push(band, top === undefined || line.y < top ? -1 : 2, line);
  }

  groups.sort((a, b) => a.band - b.band || a.rank - b.rank);

  const out: Block[] = [];
  for (const group of groups) {
    // Rank 2 is "spanning, below this band's columns" — an ordering device,
    // not a third column, so it reports as spanning.
    out.push(...toBlocks(group.lines, group.rank === 2 ? -1 : group.rank, 0));
  }
  return out.map((b, order) => ({ ...b, order }));
}
