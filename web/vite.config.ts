import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// The config is loaded as ESM (package.json has "type": "module"), so there is
// no __dirname here.
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..");
const DATA = join(REPO, "data");

const TYPES: Record<string, string> = {
  ".onnx": "application/octet-stream",
  ".png": "image/png",
  ".json": "application/json",
  ".yaml": "text/yaml",
};

/**
 * Serve the repo's `data/` tree at `/data/*` during dev.
 *
 * The character models and their ONNX exports live outside `web/`, and they
 * are gitignored and large — copying them into `public/` would duplicate
 * hundreds of MB, so they are streamed from where they already are.
 */
function serveData(): Plugin {
  return {
    name: "vpresentation-serve-data",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split("?")[0];
        if (!url?.startsWith("/data/")) return next();

        // Contain the path: no escaping out of data/ via "..".
        const target = normalize(join(DATA, decodeURIComponent(url.slice("/data/".length))));
        if (!target.startsWith(DATA)) {
          res.statusCode = 403;
          return res.end("forbidden");
        }

        try {
          if (!(await stat(target)).isFile()) return next();
          res.setHeader("Content-Type", TYPES[extname(target)] ?? "application/octet-stream");
          res.end(await readFile(target));
        } catch {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [serveData()],
  server: {
    headers: {
      // Transformers.js / ORT Web want threads, which need cross-origin
      // isolation for SharedArrayBuffer.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
