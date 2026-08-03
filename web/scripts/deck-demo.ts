/**
 * Demo (rule 2): read a PDF deck and show what the character would say.
 *
 * Text extraction and script building run in Node against pdf.js's legacy
 * build. Rasterisation is deliberately not exercised here — it needs
 * OffscreenCanvas, which Node does not have; that half is browser-only.
 *
 * Usage (from web/):  npx tsx scripts/deck-demo.ts [deck.pdf]
 */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { pageTextToScript } from "../src/slides/script.js";
import { groupBlocks, type PositionedItem } from "../src/slides/blocks.js";
import { applyReadingRules, detectRunningText, DEFAULT_RULES } from "../src/slides/reading-rules.js";
import { focusRect } from "../src/slides/focus.js";
import { composeLayout, LAYOUT_NAMES } from "../src/render/layout.js";

function toPositioned(items: ReadonlyArray<unknown>, pageHeight: number): PositionedItem[] {
  const out: PositionedItem[] = [];
  for (const raw of items) {
    if (typeof raw !== "object" || raw === null || !("str" in raw)) continue;
    const item = raw as { str: string; transform: number[]; width: number; height: number };
    const t = item.transform ?? [];
    const fontSize = Math.abs(t[3] ?? item.height ?? 10) || 10;
    out.push({
      str: item.str,
      x: t[4] ?? 0,
      y: pageHeight - (t[5] ?? 0) - fontSize,
      width: item.width || item.str.length * fontSize * 0.5,
      height: item.height || fontSize,
      fontSize,
    });
  }
  return out;
}

async function main(): Promise<void> {
  const path = resolve(process.argv[2] ?? "../out/sample_paper.pdf");
  const doc = await getDocument({ data: new Uint8Array(await readFile(path)) }).promise;
  console.log(`[deck] ${path} — ${doc.numPages} pages`);

  const all: Array<{ v: { width: number; height: number }; blocks: ReturnType<typeof groupBlocks> }> = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const v = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    all.push({ v, blocks: groupBlocks(toPositioned(content.items, v.height), v.width, v.height) });
  }

  const runningText = detectRunningText(all.map((p) => p.blocks), all[0]?.v.height ?? 842);
  if (runningText.size > 0) {
    console.log(`[deck] running text: ${[...runningText].map((t) => JSON.stringify(t)).join(", ")}`);
  }

  for (const [index, { v, blocks }] of all.entries()) {
    const i = index + 1;
    const decisions = applyReadingRules(blocks, {
      width: v.width,
      height: v.height,
      rules: DEFAULT_RULES,
      runningText,
    });

    const columns = new Set(blocks.map((b) => b.column));
    console.log(
      `\n[page ${i}] ${Math.round(v.width)}x${Math.round(v.height)} pt, ` +
        `${blocks.length} blocks, columns: ${[...columns].join(",")}`,
    );

    for (const d of decisions) {
      const b = d.block;
      const where = `col${b.column} @${Math.round(b.x)},${Math.round(b.y)}`;
      if (!d.read) {
        console.log(`  SKIP  ${where.padEnd(18)} [${d.reason}] "${b.text.slice(0, 46)}"`);
        continue;
      }
      const f = focusRect(b, { width: v.width, height: v.height }, 16 / 9, {});
      const zoom = (v.width / f.width).toFixed(2);
      console.log(
        `  READ  ${where.padEnd(18)} zoom ${zoom}x  "${pageTextToScript(b.text).slice(0, 46)}"`,
      );
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
