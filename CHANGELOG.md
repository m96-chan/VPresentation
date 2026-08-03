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

- **VoxShot 0.3.0.** A caret on a `0.x` version pins the minor, so this had to
  be asked for. What it changes here: the Chatterbox generation cap is now
  sized per chunk from its length instead of a fixed 256 tokens, so a long
  sentence is no longer cut off mid-word, and a `synthesize-truncated` event
  reports it when a cap is hit anyway. It also adds an `AbortSignal` to
  `speak()` / `stream()` / `play()` and a per-call `expressiveness`; neither is
  wired up yet (#21), so stopping still only takes effect at a chunk boundary
  while the render in flight runs to completion. The KV-cache patch below is
  untouched — that one is on `@huggingface/transformers`, which 0.3.0 still
  depends on at `^4.2.0`.

### Removed

- **In-page video recording.** A 34-page paper is roughly 106 minutes, or
  190,000 frames; no `MediaRecorder` or frame buffer survives that. Capture is
  left to OBS, with a clean-output mode and a transparent character feed.

### Fixed

- **Reading a deck aloud ran the GPU out of memory.** VRAM grew by 62 MiB per
  utterance and was never returned, so a long read ended in
  `vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY` mid-sentence —
  9.5 GB held by the browser when it fell over. The leak is in
  `@huggingface/transformers`: `ChatterboxModel.generate` passes
  `return_dict_in_generate`, which is the flag that suppresses the base
  `generate`'s KV-cache disposal, and then discards the cache. Patched locally
  with `patch-package` because no published version both supports Chatterbox
  and frees the cache, so pinning is not a way out. 20 utterances now move VRAM
  by 8 MiB in total.

- **Head and gaze axes were swapped.** THA4's pose names are rotation *axes*,
  not directions: `head_x` is rotation about x, which is pitch, and `head_y` is
  yaw. Reading them as positions drove turns into the nod axis, so the character
  looked up and down instead of left and right — and flipping the "turn" sign
  changed nothing about where it faced. The authority is THA4's own mocap
  converters, which map a tracked face onto the pose vector; the same applies to
  `iris_rotation_x` (gaze pitch) and `iris_rotation_y` (gaze yaw). Motion fields
  are now named `yaw` / `pitch` / `roll` so the confusion cannot recur, with the
  mapping onto axis-named slots in one place.

- **The thinking gaze is an occasional glance, not a held stare.** It ramped in
  after a moment's silence and never decayed, so with nothing being said the
  character sat with its chin up and its eyes pegged at the limit for as long as
  the silence lasted — vacant rather than thoughtful. Glances now occur on a
  held schedule, ease in and out, keep one direction for their whole duration,
  and stay clear of the limit. Over five minutes of silence the character is
  glancing 15% of the time and never saturates.
- **Breath is expressed as a body lean.** THA4's `breathing` parameter barely
  moves a distilled student — sweeping it from 0 to 1 changes the silhouette by
  0.19% — so the idle breath also drives `body_z`, a left/right bend worth an
  11 px centroid swing, which the model does render.

### Notes

- **WebGPU is required.** ORT Web's WASM EP has no `GridSample` kernel at all,
  so a browser without WebGPU fails to create the session rather than running
  slowly.
- `web/` now has a first pass on real hardware: headless Chrome on a real
  WebGPU device (Vulkan, RTX 5090), driven over the DevTools protocol, loads a
  34-page paper, renders the composite and reads it aloud with Chatterbox. That
  is what turned up the GPU leak above. The subjective checks in issue #18 —
  motion quality, camera pacing, lip-sync against real speech — are still
  open; a screenshot is not a substitute for watching it.
