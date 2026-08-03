/**
 * Guard the local patch against the Chatterbox KV cache leak (issue #20).
 *
 * `ChatterboxModel.generate` asks the base `generate` for a dict, which makes
 * it skip `past_key_values.dispose()`, and then throws the cache away — so
 * every synthesized chunk leaks its whole KV cache on the GPU. Measured at
 * +62 MiB per utterance, which is what killed a 34-page read-aloud with
 * `vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY`.
 *
 * The fix lives in `patches/`, applied by `patch-package` on postinstall, and
 * that is exactly why it needs a test: a patch is silent when it stops being
 * applied. The two ways it can vanish are installing without the postinstall
 * hook, and bumping `@huggingface/transformers` past the version in the patch
 * file name. Both are caught here rather than in a browser an hour into a
 * presentation.
 *
 * Delete all of this — patch, hook and test — once upstream disposes the cache
 * itself (huggingface/transformers.js, still unfixed on `main`).
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const WEB = dirname(dirname(fileURLToPath(import.meta.url)));
const MODULE = join(WEB, "node_modules", "@huggingface", "transformers");

/** The entry Vite resolves for the browser — the one the patch has to hit. */
const BROWSER_BUNDLE = join(MODULE, "dist", "transformers.web.js");

function patchFiles(): string[] {
  const dir = join(WEB, "patches");
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".patch")) : [];
}

describe("the Chatterbox KV cache patch", () => {
  it("ships a patch for @huggingface/transformers", () => {
    expect(patchFiles()).toContain("@huggingface+transformers+4.2.0.patch");
  });

  it("names the version that is actually installed", () => {
    const installed = JSON.parse(readFileSync(join(MODULE, "package.json"), "utf8")).version;
    // patch-package matches on this file name; a bumped dependency leaves the
    // patch on disk while nothing applies it.
    expect(patchFiles()).toContain(`@huggingface+transformers+${installed}.patch`);
  });

  // `toContain` on the bundle itself would dump two megabytes of minified
  // JavaScript into the failure output, so these assert on the answer instead.
  it("is applied to the browser bundle", () => {
    const bundle = readFileSync(BROWSER_BUNDLE, "utf8");
    expect(bundle.includes("await past_key_values?.dispose();")).toBe(true);
  });

  it("disposes the cache that ChatterboxModel.generate destructures", () => {
    const bundle = readFileSync(BROWSER_BUNDLE, "utf8");
    // Destructuring it is what makes the disposal reachable: without the extra
    // binding the identifier in the patched line is simply not in scope.
    const bound =
      "const { sequences, audio_tokens, speaker_embeddings, speaker_features, past_key_values } =";
    expect(bundle.includes(bound)).toBe(true);
  });
});
