/**
 * Demo app: RSS -> VoxShot -> PoseTrack -> THA4 student, in the browser.
 *
 * Both renderers share everything up to the PoseTrack; they differ only in
 * what consumes it. Realtime chases the audio clock and drops frames when
 * inference lags; pre-render walks every frame and only then plays the result
 * back into a recorder.
 */
import { ChatterboxEngine, VoxShot } from "voxshot";

import { streamSpeech, toAudioBuffer } from "./pipeline.js";
import { StudentPoser } from "./render/student.js";
import { RealtimeRenderer, type RealtimeStats } from "./render/realtime.js";
import { LivePoseEngine, renderPoseFrames } from "./track/live.js";
import { SlideDeck } from "./slides/pdf.js";
import {
  applyReadingRules,
  detectRunningText,
  DEFAULT_RULES,
  type ReadingDecision,
} from "./slides/reading-rules.js";
import { FocusCamera } from "./slides/focus.js";
import { pageTextToScript } from "./slides/script.js";
import { Compositor } from "./render/compositor.js";
import { POSE_INDEX } from "./pose/params.js";
import type { LayoutName } from "./render/layout.js";
import { IMAGE_SIZE } from "./render/image.js";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const ui = {
  character: $<HTMLSelectElement>("character"),
  gpu: $<HTMLParagraphElement>("gpu"),
  engine: $<HTMLSelectElement>("engine"),
  voice: $<HTMLSelectElement>("voice"),
  voiceName: $<HTMLInputElement>("voice-name"),
  saveVoice: $<HTMLButtonElement>("save-voice"),
  reference: $<HTMLInputElement>("reference"),
  pdf: $<HTMLInputElement>("pdf"),
  prev: $<HTMLButtonElement>("prev"),
  next: $<HTMLButtonElement>("next"),
  deckStatus: $<HTMLParagraphElement>("deck-status"),
  layout: $<HTMLSelectElement>("layout"),
  side: $<HTMLSelectElement>("side"),
  loading: $<HTMLDivElement>("loading"),
  loadingText: $<HTMLParagraphElement>("loading-text"),
  loadingSub: $<HTMLParagraphElement>("loading-sub"),
  script: $<HTMLTextAreaElement>("script"),
  speakText: $<HTMLButtonElement>("speak-text"),
  realtime: $<HTMLButtonElement>("realtime"),
  clean: $<HTMLButtonElement>("clean"),
  debug: $<HTMLButtonElement>("debug"),
  log: $<HTMLDivElement>("log"),
  stop: $<HTMLButtonElement>("stop"),
  stage: $<HTMLCanvasElement>("stage"),
};

/**
 * Show a blocking overlay while nothing is moving yet.
 *
 * Startup is genuinely slow — the THA4 student, then up to a gigabyte of
 * Chatterbox weights, then rasterising the first page — and until the first
 * frame lands the canvas is simply blank. Without this it reads as broken.
 */
function setLoading(text: string | null, detail = ""): void {
  ui.loading.hidden = text === null;
  if (text !== null) {
    ui.loadingText.textContent = text;
    ui.loadingSub.textContent = detail;
  }
}

function log(message: string): void {
  ui.log.textContent += `${message}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

let deck: SlideDeck | undefined;
let pageReading: ReadingDecision[][] = [];
let camera: FocusCamera | undefined;
let page = 0;
let compositor: Compositor | undefined;
let presenting = false;
let tts: VoxShot | undefined;
let ttsKind = "";
let poser: StudentPoser | undefined;
let loadedCharacter = "";
let characterStill: Uint8ClampedArray | undefined;
let busy = false;

function setBusy(value: boolean): void {
  busy = value;
  ui.realtime.disabled = value || !deck;
  ui.speakText.disabled = value;
  ui.pdf.disabled = value;
}

// --- lazy resources --------------------------------------------------------

async function getPoser(): Promise<StudentPoser> {
  const name = ui.character.value;
  if (poser && loadedCharacter === name) return poser;

  await poser?.dispose();
  log(`loading THA4 student "${name}" (WebGPU)…`);

  const base = `/data/character_models/${name}`;
  const next = await StudentPoser.load({
    faceMorpher: `${base}/onnx/student_face_morpher.onnx`,
    bodyMorpher: `${base}/onnx/student_body_morpher.onnx`,
  });

  // Decode the character still through a canvas to get raw RGBA.
  const bitmap = await createImageBitmap(await (await fetch(`${base}/character.png`)).blob());
  const scratch = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  const ctx = scratch.getContext("2d");
  if (!ctx) throw new Error("no 2d context");
  ctx.drawImage(bitmap, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const stillPixels = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  next.setCharacterPixels(stillPixels);
  characterStill = stillPixels;

  poser = next;
  loadedCharacter = name;
  log(`student ready on ${next.executionProviders.join("+")}`);
  return next;
}

/**
 * Get a VoxShot instance for the selected engine.
 *
 * The engine is chosen explicitly rather than inferred from whether a file
 * happens to be attached. Inferring it meant forgetting the reference audio
 * silently produced PlaceholderEngine output — speech-*shaped* noise that
 * looks like the pipeline working while being nothing of the kind.
 */
async function getTts(): Promise<VoxShot> {
  const wanted = ui.engine.value;
  if (tts && ttsKind === wanted) return tts;

  await tts?.dispose();
  tts = undefined;

  if (wanted === "placeholder") {
    log("engine: PlaceholderEngine — speech-SHAPED noise, not speech");
    tts = await VoxShot.create();
    const samples = new Float32Array(24000);
    for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i * 0.05) * 0.2;
    await tts.cloneVoice({ samples, sampleRate: 24000 });
  } else {
    log("engine: Chatterbox (WebGPU) — first run downloads the model…");
    setLoading("Downloading the voice model", "first run only, around 1 GB");
    tts = await VoxShot.create({
      engine: new ChatterboxEngine({
        onProgress: (event) => {
          if ("file" in event && event.progress !== undefined) {
            const percent = Math.round(event.progress);
            log(`  ${event.file}: ${percent}%`);
            setLoading("Downloading the voice model", `${event.file} — ${percent}%`);
          } else if ("status" in event) {
            log(`  ${event.status}`);
          }
        },
      }),
    });
    await refreshVoices();
  }

  ttsKind = wanted;
  setLoading(null);
  return tts;
}

/** Repopulate the saved-voice list from IndexedDB. */
async function refreshVoices(): Promise<void> {
  if (!tts) return;
  const names = await tts.listVoices();
  ui.voice.replaceChildren(
    ...(names.length === 0
      ? [new Option("(none saved)", "")]
      : names.map((n) => new Option(n, n))),
  );
}

/**
 * Make sure the engine has a voice loaded, failing loudly if it does not.
 *
 * Chatterbox is zero-shot: without a speaker embedding it cannot speak at all,
 * and that has to be an error the operator sees rather than a silent downgrade.
 */
async function ensureVoice(engine: VoxShot): Promise<void> {
  if (engine.currentVoice) return;

  const file = ui.reference.files?.[0];
  if (file) {
    log(`cloning voice from ${file.name}…`);
    await engine.cloneVoice(await file.arrayBuffer());
    return;
  }

  const saved = ui.voice.value;
  if (saved) {
    log(`loading saved voice "${saved}"…`);
    await engine.useVoice(saved);
    return;
  }

  throw new Error(
    "no voice — attach reference audio (a few seconds of speech) or pick a saved one",
  );
}

async function cloneAndSave(): Promise<void> {
  setBusy(true);
  try {
    const engine = await getTts();
    const file = ui.reference.files?.[0];
    if (!file) throw new Error("choose a reference audio file first");

    log(`cloning voice from ${file.name}…`);
    await engine.cloneVoice(await file.arrayBuffer());

    const name = ui.voiceName.value.trim() || file.name.replace(/\.[^.]+$/, "");
    await engine.saveVoice(name);
    await refreshVoices();
    ui.voice.value = name;
    log(`saved voice "${name}" — it will persist across reloads`);
  } catch (err) {
    log(`voice failed: ${(err as Error).message}`);
  } finally {
    setBusy(false);
  }
}

// --- actions ---------------------------------------------------------------

async function loadDeck(file: File): Promise<void> {
  setBusy(true);
  try {
    setLoading("Reading the deck", file.name);
    log(`loading ${file.name}…`);
    await deck?.destroy();
    deck = await SlideDeck.load(await file.arrayBuffer());
    page = 0;

    // Decide what to read *now*, so the operator can see the rules' effect
    // before pressing play rather than discovering it mid-presentation.
    // Running headers can only be spotted across the whole deck, so this is a
    // two-pass job: find the repeated text first, then judge each page.
    const runningText = detectRunningText(
      deck.pages.map((p) => p.blocks),
      deck.pages[0]?.height ?? 842,
    );
    if (runningText.size > 0) {
      log(`  running text: ${[...runningText].map((t) => `"${t.slice(0, 30)}"`).join(", ")}`);
    }
    pageReading = deck.pages.map((p) =>
      applyReadingRules(p.blocks, {
        width: p.width,
        height: p.height,
        rules: DEFAULT_RULES,
        runningText,
      }),
    );

    const words = pageReading
      .flat()
      .filter((d) => d.read)
      .reduce((s, d) => s + d.block.text.split(/\s+/).length, 0);
    // ~150 wpm is a normal reading pace. Worth stating up front: a 34-page
    // paper is close to two hours, which nothing downstream should try to
    // synthesize or pre-render in one go.
    const minutes = words / 150;
    log(`deck: ${deck.pageCount} pages, ~${words} words -> ~${minutes.toFixed(0)} min of speech`);
    if (minutes > 10) {
      log(
        `  ⚠ that is a long read — ~${Math.round(minutes * 60 * 30).toLocaleString()} frames ` +
          `at 30fps. Capture it with OBS rather than recording in the page.`,
      );
    }
    for (const [i, decisions] of pageReading.entries()) {
      const read = decisions.filter((d) => d.read);
      const skipped = decisions.filter((d) => !d.read);
      const why = new Map<string, number>();
      for (const d of skipped) why.set(d.reason ?? "?", (why.get(d.reason ?? "?") ?? 0) + 1);
      log(
        `  p${i + 1}: ${read.length} block(s) to read` +
          (skipped.length
            ? `, ${skipped.length} skipped (${[...why].map(([r, n]) => `${r}×${n}`).join(", ")})`
            : ""),
      );
    }

    await ensureRunning();
    setLoading("Rendering page 1", `${deck.pageCount} pages`);
    await showPage(0);
    setLoading(null);
  } catch (err) {
    setLoading(null);
    log(`deck failed: ${(err as Error).message}`);
    console.error(err);
  } finally {
    setBusy(false);
    updateDeckControls();
  }
}

function updateDeckControls(): void {
  const has = deck !== undefined;
  ui.prev.disabled = !has || page === 0;
  ui.next.disabled = !has || page >= (deck?.pageCount ?? 0) - 1;
  ui.realtime.disabled = !has || busy;
  const readable = pageReading[page]?.filter((d) => d.read).length ?? 0;
  ui.deckStatus.textContent = has
    ? `page ${page + 1} / ${deck!.pageCount} — ${readable} block(s) to read`
    : "No deck loaded.";
}

/** Rasterise a page at the size the layout will actually draw it. */
async function showPage(index: number): Promise<void> {
  if (!deck || !compositor) return;
  page = Math.max(0, Math.min(deck.pageCount - 1, index));
  const meta = deck.pages[page]!;

  // Rasterise well above display size: the camera zooms in, and upscaling a
  // page rendered at panel size would be exactly as unreadable as not zooming.
  const geometry = compositor.geometry();
  const panel = geometry.slidePanel ?? geometry.slide;
  const panelWidth = panel?.width ?? compositor.width;
  const width = Math.max(1400, Math.round(panelWidth * 2.5));
  const bitmap = await deck.render(page, width);
  compositor.slide = bitmap;

  // The camera works in page points; the compositor wants bitmap pixels.
  pageScale = bitmap.width / meta.width;
  // Framed to the *panel*, so a focused block fills it exactly instead of
  // being letterboxed a second time.
  camera = new FocusCamera(
    { width: meta.width, height: meta.height },
    Math.max(0.2, (panel?.width ?? 16) / (panel?.height ?? 9)),
  );
  applyCamera();
  updateDeckControls();
}

let pageScale = 1;

/** Push the camera's current rect into the compositor, in bitmap pixels. */
function applyCamera(): void {
  if (!compositor || !camera) return;
  const r = camera.rect;
  compositor.slideViewport = {
    x: r.x * pageScale,
    y: r.y * pageScale,
    width: r.width * pageScale,
    height: r.height * pageScale,
  };
}

/**
 * The character animates from the moment the model loads, not from the moment
 * there is audio. Speaking is something it does *while* idling.
 */
let engine: LivePoseEngine | undefined;
let renderer: RealtimeRenderer | undefined;
let audioContext: AudioContext | undefined;
let clockOrigin = 0;

async function ensureRunning(): Promise<LivePoseEngine> {
  if (engine && renderer && compositor) return engine;

  setLoading("Loading the character model", "THA4 student, WebGPU");
  const student = await getPoser();
  audioContext ??= new AudioContext();
  await audioContext.resume();

  compositor = new Compositor({
    canvas: ui.stage,
    layout: ui.layout.value as LayoutName,
    characterSide: ui.side.value as "left" | "right",
  });
  if (characterStill) {
    const content = compositor.measureCharacter(characterStill);
    const below = IMAGE_SIZE - (content.y + content.height);
    log(`character artwork: ${content.width}x${content.height}, ${below}px of empty frame below`);
  }

  engine = new LivePoseEngine({ fps: 30, seed: 20260731, body: { headingBias: headingBias() } });
  clockOrigin = audioContext.currentTime;

  renderer = new RealtimeRenderer({
    poser: student,
    canvas: ui.stage,
    clock: () => (audioContext ? audioContext.currentTime - clockOrigin : 0),
    // The compositor owns the frame: the character goes into its buffer and
    // the composed result is drawn once per frame.
    present: (rgba) => {
      // The camera advances on the render clock, so its easing is tied to what
      // is actually being drawn rather than to a separate timer.
      camera?.step(1 / 30);
      applyCamera();
      compositor!.setCharacter(rgba);
      compositor!.draw();
      drawDebug();
      // The overlay comes down on the first *drawn* frame, not when loading
      // finishes — that is the moment there is actually something to look at.
      if (ui.loading.hidden === false) setLoading(null);
    },
  });
  void renderer.run(engine, 30).catch((err) => {
    log(`renderer stopped: ${(err as Error).message}`);
    console.error(err);
  });

  log("idle animation running — breathing, blinking, looking around");
  ui.stop.disabled = false;
  return engine;
}

/**
 * Which way the presenter should idle.
 *
 * Standing bottom-right puts the deck on their left, so they should face left.
 * Yaw is `head_y` (`head_x` is pitch), and positive yaw is the viewer's left —
 * per THA4's own mocap converters, not guessed from renders.
 */
function headingBias(): number {
  return ui.side.value === "right" ? 0.45 : -0.45;
}

let debugOverlay = false;

/**
 * Draw the numbers that decide where the character is looking.
 *
 * Added because "it still faces right" and the measured source disagreed, and
 * there was no way to tell which. Sign conventions here are unintuitive enough
 * that an argument about them should be settled with values, not impressions.
 */
function drawDebug(): void {
  if (!debugOverlay || !compositor || !engine) return;
  const ctx = ui.stage.getContext("2d");
  if (!ctx) return;

  const pose = engine.frameAt(audioContext ? audioContext.currentTime - clockOrigin : 0);
  const v = (name: keyof typeof POSE_INDEX) => pose[POSE_INDEX[name]]!.toFixed(2).padStart(5);
  const yaw = pose[POSE_INDEX.head_y]!;
  const lines = [
    `corner    bottom-${ui.side.value}   bias ${headingBias().toFixed(2)}`,
    `yaw       ${v("head_y")}  (head_y, + = viewer LEFT)`,
    `pitch     ${v("head_x")}  (head_x, + = chin UP)`,
    `gaze yaw  ${v("iris_rotation_y")}  gaze pitch ${v("iris_rotation_x")}`,
    `facing    ${yaw > 0.02 ? "◀ LEFT" : yaw < -0.02 ? "RIGHT ▶" : "centre"}`,
  ];

  ctx.save();
  ctx.font = "13px ui-monospace, monospace";
  ctx.textBaseline = "top";
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  ctx.fillRect(8, 8, w, lines.length * 17 + 12);
  ctx.fillStyle = "#7CFC9B";
  lines.forEach((l, i) => ctx.fillText(l, 16, 15 + i * 17));
  ctx.restore();
}

async function stopRunning(): Promise<void> {
  presenting = false;
  // Awaited: the loop may still have a frame in flight, and starting a new
  // renderer on top of it runs two poses at once.
  await renderer?.stop();
  renderer = undefined;
  engine = undefined;
  compositor = undefined;
  ui.stop.disabled = true;
  log("stopped");
}

/**
 * Speak one utterance into the running engine, returning when the audio has
 * finished sounding.
 *
 * Audio is scheduled ahead of the clock and chunks are fed in as they are
 * synthesized. If synthesis falls behind, the character keeps idling rather
 * than desynchronising — which is why the engine runs on a clock instead of on
 * an audio buffer.
 */
async function speakInto(live: LivePoseEngine, text: string): Promise<void> {
  const tts = await getTts();
  await ensureVoice(tts);
  const context = audioContext!;

  let startTime = 0;
  let cursor = 0;
  let drift = 0;
  let opened = false;
  let underruns = 0;

  const t0 = performance.now();

  for await (const chunk of streamSpeech({ tts, text, fps: 30, seed: 20260731 })) {
    const buffer = toAudioBuffer(chunk.audio.samples, chunk.audio.sampleRate, context);

    if (!opened) {
      startTime = context.currentTime - clockOrigin + 0.4;
      cursor = clockOrigin + startTime;
      live.beginSpeech(startTime, chunk.audio.sampleRate);
      opened = true;
      log(`  speaking after ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    }

    live.pushAudio(chunk.audio.samples, chunk.span);

    // `cursor` is where the next chunk *should* start, in context time.
    const at = Math.max(cursor, context.currentTime);
    if (at > cursor + 1e-3) {
      underruns++;
      drift += at - cursor;
      log(`  ⚠ synthesis ${(at - cursor).toFixed(2)}s behind — audio re-based`);
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.start(at);
    // Advance from where playback *actually* starts. Advancing from the ideal
    // timeline meant that after one underrun every remaining chunk was also in
    // the past, so they all fired at once and played on top of each other.
    cursor = at + buffer.duration;
  }

  live.endSpeech();
  if (underruns > 0) log(`  ${underruns} underrun(s), ${drift.toFixed(2)}s drift`);

  // Wait for the audio to finish sounding before the caller moves on.
  const remaining = cursor - context.currentTime;
  if (remaining > 0) await new Promise((r) => setTimeout(r, remaining * 1000));
}

/** Debug path: read whatever is in the textarea. */
async function speakText(text: string): Promise<void> {
  setBusy(true);
  try {
    const live = await ensureRunning();
    await speakInto(live, text);
    log("done — still idling");
  } catch (err) {
    setLoading(null);
    log(`speaking failed: ${(err as Error).message}`);
    console.error(err);
  } finally {
    setBusy(false);
  }
}

/**
 * Present the deck: read each page block by block, camera following.
 *
 * Following the narration is what makes an A4 paper legible at all — the whole
 * page letterboxed into a 16:9 panel puts the body text a few pixels tall.
 */
async function presentDeck(): Promise<void> {
  if (!deck) return;
  setBusy(true);
  presenting = true;
  try {
    const live = await ensureRunning();

    for (let i = page; i < deck.pageCount && presenting; i++) {
      await showPage(i);
      const decisions = pageReading[i] ?? [];
      const toRead = decisions.filter((d) => d.read);
      log(`page ${i + 1}/${deck.pageCount} — ${toRead.length} block(s)`);

      if (toRead.length === 0) {
        // A figure-only page. Hold on it rather than skipping, so it still gets
        // time on screen and the narration stays lined up with the deck.
        camera?.focus(null);
        await new Promise((r) => setTimeout(r, 2500));
        continue;
      }

      for (const decision of toRead) {
        if (!presenting) break;
        camera?.focus(decision.block);
        const text = pageTextToScript(decision.block.text);
        if (!text) continue;
        log(`  ▸ "${text.slice(0, 60)}${text.length > 60 ? "…" : ""}"`);
        await speakInto(live, text);
      }
    }
    camera?.focus(null);
    log(presenting ? "deck finished — still idling" : "presentation stopped");
  } catch (err) {
    setLoading(null);
    log(`presentation failed: ${(err as Error).message}`);
    console.error(err);
  } finally {
    presenting = false;
    setBusy(false);
  }
}

// --- wiring ----------------------------------------------------------------

ui.pdf.addEventListener("change", () => {
  const file = ui.pdf.files?.[0];
  if (file) void loadDeck(file);
});
ui.prev.addEventListener("click", () => void showPage(page - 1));
ui.next.addEventListener("click", () => void showPage(page + 1));
ui.side.addEventListener("change", () => {
  if (compositor) {
    // The idle facing has to follow the corner, so the engine is rebuilt.
    void stopRunning()
      .then(() => ensureRunning())
      .then(() => showPage(page));
    log(`presenter corner: bottom ${ui.side.value} — facing the deck`);
  }
});
ui.layout.addEventListener("change", () => {
  if (compositor) {
    compositor.layout = ui.layout.value as LayoutName;
    // The slide's rect changed, so it needs rasterising at the new size.
    void showPage(page);
    log(`layout: ${ui.layout.value}`);
  }
});

ui.realtime.addEventListener("click", () => void presentDeck());
ui.speakText.addEventListener("click", () => void speakText(ui.script.value));
ui.stop.addEventListener("click", () => void stopRunning());
ui.debug.addEventListener("click", () => {
  debugOverlay = !debugOverlay;
  log(`debug overlay ${debugOverlay ? "on" : "off"}`);
});
ui.clean.addEventListener("click", () => {
  document.body.classList.toggle("clean");
  // Escape is the only way back once the panel is hidden.
  log(document.body.classList.contains("clean") ? "clean output — Esc to exit" : "");
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.body.classList.remove("clean");
});
ui.saveVoice.addEventListener("click", () => void cloneAndSave());
ui.engine.addEventListener("change", () =>
  log(`engine set to ${ui.engine.value} — loads on next speak`),
);
ui.character.addEventListener("change", () => {
  if (!busy) log(`character set to ${ui.character.value}`);
});

ui.gpu.textContent =
  "gpu" in navigator
    ? "WebGPU available."
    : "WebGPU NOT available — THA4 cannot run. ORT Web's WASM EP has no GridSample kernel.";
