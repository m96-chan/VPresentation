# VPresentation

English | [日本語](README.ja.md)

[![Status](https://img.shields.io/badge/status-early%20WIP-orange)](#-status)
[![License](https://img.shields.io/badge/license-MIT%20(code)-blue)](LICENSE)
[![Built with THA4](https://img.shields.io/badge/built%20with-Talking%20Head%20Anime%204-ff69b4)](https://github.com/pkhungurn/talking-head-anime-4-demo)

**A VTuber presentation, from a single 2D character illustration.**

VPresentation turns one character illustration and a slide deck (PPTX / PDF)
into a livestream-style video where the character appears to be presenting
the slides. It's built on
[Talking Head Anime 4 (THA4)](https://github.com/pkhungurn/talking-head-anime-4-demo)
to animate the illustration, synced to slide-advance effects.

> ⚠️ **Status: early concept / development.** This README describes the
> intended goal and design. Most listed features are not implemented yet.

---

## Table of contents

- [Concept](#-concept)
- [Features (goals)](#-features-goals)
- [Architecture (planned)](#️-architecture-planned)
- [Tech stack (under discussion)](#-tech-stack-under-discussion)
- [Requirements (tentative)](#-requirements-tentative)
- [Usage (planned)](#-usage-planned)
- [Development (THA4 engine)](#️-development-tha4-engine)
- [Roadmap](#️-roadmap)
- [Credits](#-credits--related-projects)
- [License](#-license)

---

## 🎯 Concept

- **One illustration is enough** — no Live2D-style multi-layer rigging.
  Motion is generated from a single front-facing standing illustration.
- **Slides as-is** — load the PowerPoint (PPTX) / PDF deck you already have,
  directly.
- **Made for streaming** — output is designed to drop straight into OBS and
  similar tools (window capture / virtual camera / chroma key).

---

## ✨ Features (goals)

### Character display
- [x] Expression / head motion generation from a single 2D illustration
      (THA4-based)
- [x] Idle motion — auto-looping blink / breathing
- [x] Lip-sync driven by TTS audio (waveform-driven, see
      [Reading a feed aloud](#-reading-a-feed-aloud))
- [x] Automatic switching between expression presets from the script's text
- [ ] Lip-sync driven by microphone input

### Slide integration
- [ ] **PPTX** loading and page rendering
- [ ] **PDF** loading and page rendering
- [ ] Slide advance / back (keyboard shortcuts, external triggers)
- [ ] Speaker notes lookup

### Slide-advance animation
Streaming impact matters, so multiple switchable transition styles are
planned:
- [ ] Fade / slide-in / zoom / flip / page-turn, etc.
- [ ] Character reactions synced to the transition (e.g. raising a hand on
      the next slide)
- [ ] Customizable transition timing and easing

### Streaming & output
- [ ] Chroma-key (transparent background) output
- [ ] Virtual camera output / window capture support
- [ ] Layout presets (full slide + picture-in-picture, half-and-half, etc.)

---

## 🏗️ Architecture (planned)

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 2D Illustr.  │──▶ │  THA4 inference  │──▶ │                  │
│  (1 PNG)     │    │ (expr/head/mouth)│    │                  │
└──────────────┘    └─────────────────┘    │    Compositor    │
                                            │ (char + slide +  │──▶ Stream output
┌──────────────┐    ┌─────────────────┐    │   transitions)   │   (OBS / virtual cam)
│ PPTX / PDF   │──▶ │  Slide renderer  │──▶ │                  │
└──────────────┘    └─────────────────┘    └──────────────────┘
                            ▲
                    ┌───────┴────────┐
                    │  Control UI /   │
                    │   shortcuts     │
                    └────────────────┘
```

Key components:

| Component | Role |
| --- | --- |
| THA4 inference engine | Generates expression/head/mouth motion from pose parameters on the standing illustration |
| Slide renderer | Converts PPTX / PDF pages into image frames |
| Compositor | Merges character video with slides and applies transition animation |
| Control UI | Slide advance, expression switching, and other operator controls |
| Output layer | Chroma-key / virtual camera / window output |

---

## 🔧 Tech stack (under discussion)

- **Character animation**: Talking Head Anime 4 (THA4) — PyTorch
- **Slide conversion**: PPTX (`python-pptx` or rendering via LibreOffice), PDF
  (`pdf2image` / `PyMuPDF`, etc.)
- **Compositing & output**: TBD (rendering engine, virtual camera
  integration)

> The stack isn't finalized yet. It will be tuned against THA4's
> requirements (e.g. GPU inference).

---

## 📦 Requirements (tentative)

- A GPU (NVIDIA/CUDA or Apple Silicon/Metal) is recommended for THA4
  inference
- Model weights must be fetched separately from THA4's distribution (see
  [`tools/fetch_weights.sh`](tools/fetch_weights.sh))
- More detail will be added as the implementation matures

---

## 🚀 Usage (planned)

> Not implemented yet — the following is the target experience.

1. Prepare a standing character illustration (front-facing, transparent PNG)
2. Prepare your presentation deck (`.pptx` / `.pdf`)
3. Launch the app and load the illustration and deck
4. Choose a slide-transition style and layout
5. Feed the output into OBS (or similar) and go live

---

## 🛠️ Development (THA4 engine)

Character animation is implemented in **Rust + candle + Metal** on the
[m96-chan/candle](https://github.com/m96-chan/candle) fork (`avatacam`
branch), targeting Apple Silicon (Metal), NVIDIA (CUDA), and CPU from the
same codebase.

```bash
# 1. Fetch the candle fork (submodule)
git submodule update --init --recursive

# 2. Fetch THA4's pretrained weights (~610MB, CC BY-NC 4.0; not vendored in this repo)
./tools/fetch_weights.sh

# 3. Confirm candle + Metal work (Phase 0 smoke test)
cargo run -p tha4 --bin metal-smoke
cargo test -p tha4
```

### Implementation phases (issue #4)
- [x] **Phase 0**: Foundation — wire up the candle fork / confirm Metal
      execution / fetch weights
- [x] **Phase 1**: Add `grid_sample` / `affine_grid` ops to the candle fork
- [x] **Phase 2**: Port the 5 networks (eyebrow_decomposer →
      eyebrow_morphing_combiner → face_morpher → body_morpher → upscaler)
- [x] **Phase 3**: Wire up the pipeline + `char.png` preprocessing +
      blink/breathing loop → animation

> THA4's general poser is made of 5 networks and relies on `grid_sample`
> warping, which candle doesn't implement — it's being ported incrementally.

**Real-time (student) path**: the general poser above can pose *any* image
but is too slow to stream (a few fps at best). Real-time playback needs a
per-character **distilled student model** — see
[`DISTILL.md`](DISTILL.md) for the full distillation procedure (NVIDIA GPU
required) that turns a teacher-posed illustration into a ~25fps student.

---

## 🗣️ Presenting a deck

`web/` is a browser app: load a PDF, and the character reads each page and
advances through the deck. Speech comes from
[VoxShot](https://github.com/m96-chan/voxshot) (zero-shot TTS, WebGPU +
ONNX Runtime Web); the mouth is recovered from its audio.

A plain-text box is kept alongside it for debugging — useful for exercising
lip sync without loading a deck.

```
PDF ─► pdf.js ─┬─► page bitmap (LRU 3) ─────────┐
               │                                 │
               └─► blocks + reading rules ─┐     │
                                            ▼    │
                          VoxShot.stream() ─► audio chunks
                                    │             │
                        emotion tagger    lip sync (waveform)
                                    ▼             │
              LivePoseEngine — always running      │
     breathing · blinking · looking around · thinking
                                    ▼             ▼
                            45 floats ──►  Compositor  ──► canvas ──► OBS
                                        (layout · reading camera)
```

### Length is unbounded, so nothing may accumulate

A 34-page paper is around 16,000 words — **roughly 106 minutes** of speech, or
190,000 frames. Everything on the live path is therefore bounded, and
`test/endurance.test.ts` pins it:

| | was | now |
| --- | --- | --- |
| retained audio | the whole utterance (2.1 M samples per minute) | a sliding window, < 2 s |
| analysed mouth frames | one per frame, forever | pruned once the clock passes them |
| emotion spans | one per chunk, forever | pruned past the blend horizon |
| rasterised pages | every page, 11 MB each (377 MB at 34 pages) | LRU of 3, explicitly closed |

**Recording is external.** There is no in-page recorder: an hours-long reading
is not something MediaRecorder or a frame buffer survives, and the numbers are
not close. The *Clean output* button hides everything but the canvas for a
window or browser-source capture, and the `character-only` layout gives a
transparent feed to chroma-key (issue #9).

**The character animates from the moment the model loads, not from the moment
there is audio.** The first design built a finite pose track out of a finished
utterance, so nothing existed until something was spoken — no breathing, no
blinking, no presence. A VTuber is idling on stream whether or not it is
mid-sentence, so the engine runs on a clock and speech is layered into it.

That also removes streaming's failure mode. Playback no longer depends on
synthesis keeping up: if the next chunk is late, the character carries on
breathing instead of desynchronising.

```bash
cd web
npm install
npm test          # 115 unit tests
npm run dev       # then open the printed URL in a WebGPU browser
```

### Decks

Pages are rasterised with **pdf.js** and cached — a slide is static until the
deck advances, so re-rendering it every frame would take GPU time from THA4 for
nothing. Issue #5 originally assumed PyMuPDF / pdf2image; the app moved into the
browser, so the deck renders there too.

Page text is extracted **with positions**, grouped into blocks in reading
order, and filtered before anything is spoken.

**Reading order** is the part that matters. A two-column paper walked in raw
y order interleaves the columns and produces fluent gibberish — the worst
available failure, because it sounds fine. Lines are assigned strictly to the
left column, the right column, or *spanning*; spanning lines then divide the
page into bands, and each band is read as spanning-above → left → right →
spanning-below.

**Reading rules** (`src/slides/reading-rules.ts`) drop what is on the page for
a reader rather than a listener, each with a reason so it is visible why:

| Reason | |
| --- | --- |
| `references` | everything from the bibliography heading on |
| `running-text` | repeated across pages — the reliable way to catch a header like "Preprint. Under review." that sits outside any sane margin |
| `page-number` | `12`, `3 / 20` |
| `symbols` | extracted equations, which come out as soup |
| `margin` | running furniture in the top/bottom margins |
| `too-short` | `Fig.` and friends — but a bare `Conclusion` is kept |

**The reading camera** (`src/slides/focus.ts`) zooms to the block being spoken.
This is what makes an A4 paper legible at all: the whole page letterboxed into
a 16:9 panel puts body text a few pixels tall. It moves on the same damped
spring as the character's head, because cutting between paragraphs mid-sentence
is jarring. Pages are rasterised well above panel size, since the camera zooms
in and upscaling would be exactly as unreadable as not zooming.

A page with nothing readable — a chart, a photo — is **kept in the deck and
held on screen** rather than skipped, so the narration stays lined up.

Layouts (issue #7) are pure geometry in `src/render/layout.ts`, so the fiddly
part is tested without a DOM. Neither source is ever stretched:

| Preset | |
| --- | --- |
| `half` | presenter beside the deck — the usual presentation-stream framing |
| `picture-in-picture` | deck edge to edge, presenter inset |
| `slide-only` | |
| `character-only` | transparent background, for chroma-keying into OBS (issue #9) |

The presenter **stands in a bottom corner** (right by default, switchable),
because that is where a VTuber sits. Centring a standing illustration in its
column left it hovering with a gap under its feet.

Startup is slow enough to need saying so: the THA4 student, then up to a
gigabyte of Chatterbox weights, then the first page rasterised. A loading
overlay covers all of it and comes down on the first *drawn* frame — the moment
there is actually something to look at, rather than when loading merely
finishes.

### Why lip sync is estimated from audio

VoxShot exposes **no phoneme timings** — `synthesize()` resolves to a bare
`Float32Array`, and its Chatterbox backend is autoregressive, so there is no
duration predictor to borrow an alignment from. The mouth is therefore
recovered from the waveform: per frame, LPC (autocorrelation →
Levinson-Durbin) fits an all-pole vocal-tract filter, its first two resonances
give F1/F2, and those are matched in mel space against canonical vowel targets
to weight THA4's `mouth_aaa / iii / uuu / eee / ooo`. Swap in a forced aligner
later by implementing `LipsyncSource`.

Sentence-level *emotion* timing is exact, though. Synthesis runs through
`VoxShot.stream()`, which yields one `SynthesizedAudio` per chunk without
saying which text produced it — but it does not have to. Internally it calls
the exported `splitSentences(text, {maxLength, minLength})` and yields in
order, so calling that same function with the same options reproduces the chunk
list and the two zip together positionally. (Which is why `maxChunkLength` /
`minChunkLength` must match whatever the `VoxShot` instance was built with.)

Streaming means playback starts after the *first sentence* rather than after
the whole article. That needed the pose track to be built incrementally, and
almost everything in it is stateful — lipsync smoothing, the speech envelope,
and the springs that give the body inertia. Processing chunks independently
would restart the springs at every sentence and snap the head back to centre,
so `PoseTrackBuilder` carries all of it across chunk boundaries. Two details
make the result identical to a batch build rather than merely similar:

- the analysis window is **centred**, so a frame needs audio from after it and
  is held back until its window closes;
- an emotion span crossfades *in* before it starts, so frames within half a
  blend of the known audio horizon are held back too — the next chunk may
  still change their expression.

`buildPoseTrack` is implemented on top of the same builder, and a test asserts
the streamed and batch tracks match slot for slot. That equality is what makes
it safe to record a realtime session and get back the video you just watched.

### Head and body motion

`src/motion/body.ts` sums five layers into slots 37..38 and 39..43:

| Layer | Driven by | Role |
| --- | --- | --- |
| **orientation** | **a held, stepped heading** | turning left and right |
| sway | two octaves of value noise | never perfectly still |
| **gesture** | **the speech envelope** | nods on accents, lean with sustained speech |
| posture | the active emotion | a per-emotion bias |
| **thinking** | **sustained silence** | gaze drifts up and to one side during pauses |

**Horizontal is the expressive axis.** Vertical stays near neutral while
speaking — a head that keeps looking up and down reads as restless — and only
lifts during pauses, where it reads as thought. Measured over a talking clip,
vertical travel is about a fifth of horizontal.

Getting that balance right needed the nod to be centred. `accent` is
*signed*: clamping it to `>= 0` meant every syllable pushed the head down and
nothing ever pushed it back, so the head was held down for whole utterances
(travel was −0.220…+0.092, 2.4× more downward than up; it is now −0.108…+0.088).

Two sign conventions, both established by rendering a sweep rather than
assumed:

- negative `head_x` / `iris_rotation_x` face the viewer's left
- positive `head_y` raises the chin, but **negative** `iris_rotation_y` looks
  up — the two vertical axes are inverted relative to each other, so the eyes
  roll the wrong way if you assume they match

Orientation is what makes the character look somewhere on purpose. It is a
*step* function — pick a heading, hold it 1.6–4.2 s, pick another — and the
turn happens because the spring downstream cannot follow a step instantly.
Continuous noise at the same amplitude just looks adrift. The eyes are sprung
much faster than the head, so they arrive at the new heading first and the head
follows, which is what makes a turn read as intentional.

A sweep of the student model (`npx tsx scripts/sweep.ts`) shows `head_x` and
`body_y` stay artefact-free all the way to ±1; the first version used about 15%
of that. A twelve-second clip now covers 0.58–0.89 of `head_x`.

`heading` can be supplied directly instead of generated, so a compositor can
make the character turn towards the slide.

The gesture layer is what makes the character read as *talking* rather than
idling with a moving mouth. A sway-only first cut moved `head_x` through 0.060
of its range over three seconds — about 3% of travel, i.e. invisible.

Those three layers produce a *target* pose, never the pose itself. Each channel
is then integrated through a **damped spring**, so the head has mass: it cannot
reach a new pose within one frame, and it eases in and out of every move. The
head is light and answers quickly, the torso is heavy and lags behind it, which
is most of what makes the result read as a body rather than five independent
sliders.

That inertia is not cosmetic. Driving the pose straight from the speech
envelope gave `head_y` an acceleration of **0.145 per frame** against a total
travel of 0.320 — the head teleported on every syllable. With the spring it is
**0.0034**, a 43× reduction, while still covering 0.20 of travel:

| | travel | max acceleration / frame |
| --- | --- | --- |
| sway only | 0.042 | 0.0021 |
| + gesture, no spring | 0.320 | 0.1449 |
| + gesture, sprung | 0.202 | 0.0034 |

Tunable via `buildPoseTrack({ body: { swayScale, gestureScale, postureScale,
stiffness } })` — lower `stiffness` is heavier and smoother.

### WebGPU is required (no WASM fallback)

The body morpher's warp is a `GridSample`, and ONNX Runtime Web implements
that **only on the WebGPU EP**, for `ai.onnx(16-19)`. The WASM EP has no
GridSample kernel at all, so a browser without WebGPU fails to create the
session rather than running slowly.

That opset window also constrains the export. `tools/export_student.py`
targets **opset 17** and bakes out `affine_grid` (which `torch.onnx` only
lowers at opset ≥ 20) by precomputing its constant identity grid:

```bash
.venv-distill/bin/python tools/verify_affine_patch.py data/character_models/char  # bit-exact check
.venv-distill/bin/python tools/export_student.py     data/character_models/char
```

`verify_affine_patch.py` asserts the rewrite is bit-for-bit identical to stock
THA4 before the export is trusted.

---

## 🗺️ Roadmap

1. **PoC** — animate a standing illustration with THA4 and display it
2. **Slide loading** — PDF, then PPTX, page rendering
3. **Compositing & transitions** — merge character + slides with basic
   transition effects
4. **Operability** — slide-advance controls, expression-switching UI
5. **Streaming output** — chroma-key / virtual camera support
6. **More effects** — additional transition-animation variety

---

## 🙏 Credits / related projects

- [Talking Head Anime 4 (THA4)](https://github.com/pkhungurn/talking-head-anime-4-demo)
  — the foundation for character animation

Please follow THA4's license and usage terms from its distributor.

---

## 📝 Changelog

User-observable changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).

---

## 📄 License

The VPresentation source code is [MIT licensed](LICENSE).

**This does not cover THA4's pretrained model weights.** They're
distributed separately by the THA4 author under **CC BY-NC 4.0
(non-commercial)** — see [`tools/fetch_weights.sh`](tools/fetch_weights.sh).
Any use of this project that relies on those weights (directly via the
teacher, or via a per-character model distilled from them, see
[`DISTILL.md`](DISTILL.md)) is bound by that non-commercial restriction,
regardless of this repo's own MIT license. Commercial use would require
separate arrangement with THA4's author.
