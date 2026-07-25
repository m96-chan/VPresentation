# CLAUDE.md

Guidance for AI assistants (and humans) working in the **VPresentation** repository.

> **Read [`README.md`](README.md) and the relevant issue first — every task
> starts there (rules 4 & 6).** The design and direction are tracked in the
> issues and the README.

## Project summary

VPresentation turns a single 2D character illustration plus a slide deck
(PPTX / PDF) into a VTuber-style presentation stream — the character appears to
present the slides live.

- **One illustration is enough:** a single front-facing character image, no
  Live2D-style multi-layer rigging.
- **Character animation:** built on
  [Talking Head Anime 4 (THA4)](https://github.com/pkhungurn/talking-head-anime-4-demo)
  (PyTorch) — expression / head / mouth motion generated from the still image.
- **Slides as-is:** load existing **PPTX** and **PDF** decks directly and render
  them to frames.
- **Slide-advance animations:** multiple switchable transition effects
  (fade / slide / zoom / flip / page-turn, etc.), with character reactions.
- **Made for streaming:** output designed to drop into OBS etc. — chroma-key
  (transparent background), virtual camera, and window capture.
- **Status:** early / WIP — most features are still goals, not implemented.

> The tech stack is not finalized. Confirm THA4's runtime requirements
> (GPU / CUDA inference) before committing to dependencies. When in doubt, ask
> (rule 8) rather than locking in a choice.

## Development rules (must follow)

These are hard rules. Do not skip them.

### 1. Develop with TDD
Practice test-driven development. Write a failing test first, make it pass with
the simplest change, then refactor (red → green → refactor). New behavior should
be accompanied by tests; do not add functionality without a test that covers it.

### 2. Always run a demo before pushing
**Never push without first demonstrating the change actually works.** Running the
test suite is necessary but not sufficient — exercise the real behavior (run the
app / the relevant component and observe it) before every `git push`. If a
change cannot be demoed for some reason, say so explicitly instead of pushing
silently. See [Demo before push](#demo-before-push) for the procedure.

### 3. Treat the documentation on GitHub as the source of truth
For any API, library, or tool manual, consult the **documentation published on
GitHub** as the most up-to-date reference. Do not rely on memory or training
data for API details — verify against the current upstream docs. This matters
especially for fast-moving dependencies (THA4 and the ML/rendering stack).

### 4. Read `README.md` first
**Before starting any task, read [`README.md`](README.md).** It is the canonical
overview of the project's purpose, architecture, and current direction. Ground
your work in it before touching code or docs.

### 5. Update `README.md` when you finish
**When an implementation is done, update [`README.md`](README.md)** so it stays
accurate — features, roadmap checkboxes, usage, and any changed architecture or
commands. A change is not "done" until the README reflects it. Do this before
pushing (it is part of the demo/push checklist below).

### 6. Read the relevant issue before implementing
**Before writing any code, read the GitHub issue(s) covering the work**, plus any
linked/related issues and docs. Understand the scope, task list, acceptance
criteria, and dependencies first. If no issue exists for the work, create one (or
ask) before implementing.

### 7. Update the issue to mark the work complete
**The issue is the definition of done.** When the work is finished, update the
relevant issue — check off completed tasks, note what was implemented/decided,
and close it (or mark it complete). Implementation is **not** "done" until the
issue is updated. Do this before/at push time.

### 8. Ask instead of guessing when there's ambiguity or a design gap
**If multiple interpretations/approaches are plausible, or the issue/design is
incomplete or inconsistent, stop and ask — do not guess and implement.** Don't
silently pick one path or paper over a design deficiency. Surface the ambiguity,
lay out the options with a recommendation, and get a decision before proceeding.
When the answer matters, record it in the issue/docs so it isn't re-litigated.

## Demo before push

A "demo" means exercising the real behavior of your change and observing the
result — not just a green test suite. Because the output is visual (animated
character + rendered slides + transitions), demoing means looking at actual
rendered output.

Suggested forms of demo as components land:
- Character: render the illustration driven by THA4 and confirm the motion.
- Slides: render a sample PPTX / PDF page to an image and inspect it.
- Composite: render the combined character + slide + transition to frames /
  a short clip and inspect them.

> Concrete demo commands will be added here once an entry point exists. Until
> then, describe what you ran and observed.

When you push, briefly record **what you ran and what you observed** in the
commit message or PR description so the demo is auditable.

## Working with the codebase

Build / test / run commands will be documented here as the project takes shape.
Keep this section in sync with the actual tooling once it exists.

### Before pushing — checklist
- [ ] `README.md` read at the start of the task (rule 4)
- [ ] Relevant issue(s) read before implementing (rule 6)
- [ ] Ambiguities / design gaps raised and resolved, not guessed (rule 8)
- [ ] Tests written first / updated (TDD)
- [ ] Test suite passes
- [ ] Change demoed against real behavior, and what you ran/observed is recorded (rule 2)
- [ ] Any API usage verified against current GitHub docs (rule 3)
- [ ] `README.md` updated to reflect the change (rule 5)
- [ ] Issue updated / tasks checked off / closed (rule 7)

## Conventions

- Keep new code consistent with the surrounding style; use the project's
  formatter / linter once configured.
- Keep platform- and dependency-specific code (e.g. THA4 inference, slide
  rendering backends, output/virtual-camera integration) behind clear
  boundaries so alternatives can slot in without churn.
- Respect the licenses and usage terms of upstream projects — THA4 in
  particular. Verify before adding a dependency.

## License

TBD (to be decided). Mind the licenses of upstream dependencies (THA4, slide
rendering, etc.) when choosing one.
