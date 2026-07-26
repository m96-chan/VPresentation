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
- [ ] Lip-sync driven by microphone input / TTS audio
- [ ] Manual and automatic switching between expression presets
      (smile / surprise / trouble, etc.)

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
