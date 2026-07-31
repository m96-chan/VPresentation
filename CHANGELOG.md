# Changelog

Notable, user-observable changes. Following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project has not
cut a release yet, so everything is still under `Unreleased`.

Entries say **why** a change was needed. What changed is in the diff.

## [Unreleased]

### Added

- **Browser presenter (`web/`).** Load a PDF and a character reads it aloud
  while animating. The app lives in the browser because
  [VoxShot](https://github.com/m96-chan/voxshot) is browser-first (WebGPU +
  ONNX Runtime Web); putting THA4 there too lets synthesis and rendering share
  one runtime rather than straddling a process boundary.
- **Waveform-driven lip sync.** VoxShot exposes no phoneme timings and
  Chatterbox is autoregressive, so there is no alignment to borrow. The mouth
  is recovered from the audio instead: LPC per frame, F1/F2 matched in mel
  space against cardinal vowel targets. Swappable via `LipsyncSource` if a
  forced aligner is added later.
- **Live pose engine.** The character breathes, blinks and looks around from
  the moment the model loads, and speech is layered into that. Driving pose
  from an audio buffer instead meant nothing existed until something was
  spoken, and made playback depend on synthesis keeping up.
- **PDF decks with reading rules and a reading camera.** Page text is grouped
  into positioned blocks in reading order (two-column papers included), filtered
  so bibliographies, running headers, page numbers and extracted equations are
  not read aloud, and the viewport follows the passage being spoken — which is
  what makes an A4 paper legible at all in a 16:9 frame.
- **Compositor with layout presets.** `half`, `picture-in-picture`,
  `slide-only`, and a transparent `character-only` feed for chroma-keying into
  OBS. The presenter stands in a bottom corner and idles facing the deck.

### Changed

- **Student models export at opset 17.** ORT Web implements `GridSample` only
  on the WebGPU EP for `ai.onnx(16-19)`; at opset 20 the node drops to a CPU
  kernel and dominates the frame. `torch.onnx` only lowers `affine_grid` at
  opset 20+, so it is baked out as a constant grid — asserted bit-exact against
  stock THA4 by `tools/verify_affine_patch.py` before the export is trusted.

### Removed

- **In-page video recording.** A 34-page paper is roughly 106 minutes, or
  190,000 frames; no `MediaRecorder` or frame buffer survives that. Capture is
  left to OBS, with a clean-output mode and a transparent character feed.

### Notes

- **WebGPU is required.** ORT Web's WASM EP has no `GridSample` kernel at all,
  so a browser without WebGPU fails to create the session rather than running
  slowly.
- Nothing in `web/` has been verified in a real browser yet; the demos in
  `web/scripts/` run the same modules on native ORT under Node, which cannot
  exercise rasterisation, compositing or the camera.
