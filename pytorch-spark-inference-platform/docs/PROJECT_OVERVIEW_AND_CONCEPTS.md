# Project Overview & Concepts

A conceptual map of this repo: what it is, why it's structured the way it
is, and the testing discipline that keeps changes safe to ship. Start here;
every linked doc goes deeper on one specific piece.

## What this is

A framework for running PyTorch inference **distributed across a Spark
cluster**, where the actual model(s) being run are a plug-in, not hardcoded.
"Bring your own model" (BYOM) is the core idea: someone adds a model by
writing a small adapter against a fixed contract, not by modifying the
framework itself. Two flavors of that contract exist, because two genuinely
different kinds of "model" show up in practice — see
`docs/BRING_YOUR_OWN_MODEL.md` for the authoring guide, and
`docs/REPO_STRUCTURE.md` for the exact directory layout.

| | `models/plugins/` (simple) | `models/pipelines/` (complex) |
|---|---|---|
| Shape | one `nn.Module` | any number of models/objects |
| Input | fixed-shape `float32` tensor | file paths / raw text, any format |
| Output | discarded (sample count only) | kept — real structured results |
| Example | `example_model.py` | `ner_translate/` (GLiNER + NLLB) |

## The execution layer: how a model actually runs across the cluster

`inference/cluster_engine.py` (tensor plugins) and
`inference/text_pipeline_engine.py` (pipeline plugins) both do the same
fundamental thing: use Spark's `mapPartitions` to run inference on each
partition of input data, on whichever executor happens to own that
partition. Everything downstream of that decision — which container image
an executor runs in, whether the model loads in-process or gets called over
HTTP — is a packaging choice layered on top of this, not a change to the
core execution model. See `docs/SPARK_WORKERS_EXECUTORS_EXPLAINED.md` and
`docs/ARCHITECTURE_CODE_FLOW.md` for the mechanics.

## Two ways to isolate a model's dependencies

The naive approach — one shared Python environment for every model — breaks
the first time two models need incompatible dependency versions (this
happened once already: installing `ner_translate`'s deps silently bumped
`numpy` for every other model too). `docs/MODEL_CONTAINER_ISOLATION.md` is
the full design doc; the short version:

- **Option A — one image per model, in-process.** Each pipeline gets its
  own Docker image (`FROM` a shared base), its own dependency set, its own
  small Spark cluster. Simple, Spark-native, no new moving parts. This is
  what `ner_translate` uses today via `deploy/Dockerfile.ner_translate`.
- **Option B — "waiter/kitchen" split.** Spark itself (the "waiter") runs
  on a completely model-agnostic image with zero ML dependencies. Each
  model lives in its own small HTTP server (the "kitchen") that Spark calls
  over the network instead of loading in-process. The waiter image is
  identical and reusable across every model; only the kitchen changes per
  model. Full walkthrough: `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`.

Neither is "correct" — which one to use depends on what you already have
running. If a target machine already has a compatible base image loaded,
Option A is usually less new infrastructure. If you're starting fresh and
want models to scale/update completely independently of Spark, Option B is
the better fit.

## Shipping dependency changes without re-shipping gigabytes

Both options share the same underlying trick for getting dependency updates
onto an airgapped machine cheaply: **never ship a rebuilt multi-GB image
for a small change.** Instead:

- **Python packages** → a *wheelhouse* (`deploy/scripts/build_*_wheelhouse.sh`):
  `pip download` targeting the container's exact platform/Python version,
  producing a folder of `.whl` files installed later via `pip install
  --no-index --find-links=...`. A few hundred MB instead of the packages'
  live resolution pulling in whatever's newest.
- **System packages** → a *`.deb` bundle*
  (`deploy/scripts/build_ner_translate_debs.sh`): same idea for `apt`
  packages (tesseract-ocr, poppler-utils), downloaded from inside the exact
  target base image (not a generic fresh one — version skew between the two
  breaks `dpkg`'s dependency resolution, a real bug hit and documented in
  `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`), installed via
  `dpkg -i` instead of `apt-get install`.

Both get fed into the Dockerfile via `RUN --mount=type=bind,source=...`,
not `COPY` — a bind mount is visible only to that one build step and never
becomes a permanent image layer, unlike `COPY ... && rm -rf ...`, which
looks like it cleans up but doesn't (the deleted layer's bytes still ship in
`docker save`). This exact mistake cost ~6GB of accidental bloat once before
being caught — see that doc's changelog for the full story.

**End result:** a dependency-only update is a wheelhouse + `.deb` bundle +
code tarball (tens to hundreds of MB), built into the final image **on the
target machine itself**, reusing whatever base image is already there — not
a new CD with a multi-GB image every time.

## Testing discipline: verify before you advance, not after

This project doesn't have a classical unit-test-first workflow, but follows
an equivalent discipline that serves the same purpose — catching a broken
assumption at the cheapest possible point instead of the most expensive
one. `docs/TESTING_BEFORE_EXPORT.md` lays out the concrete tiers for
`ner_translate`; the general shape applies to anything in this repo:

1. **Tier 0 — local, no cluster, no Docker.** `local[4]` Spark mode,
   whatever's fastest to iterate on. Catches logic bugs cheaply.
2. **Tier 1/2 — single Docker container, then a local compose cluster.**
   Catches packaging bugs (missing system package, wrong Python version)
   before they're expensive to debug on remote infra.
3. **Tier 3 — real multi-node cluster** (AWS, or a local multi-container
   compose cluster). Catches distributed-execution bugs — worker
   core-count mismatches, network/socket issues (the `py4j` global-timeout
   bug this session hit twice is a good example of something that only
   shows up here).
4. **Tier 4 — export and reload the actual artifact** that will ship
   (image tarball, wheelhouse, `.deb` bundle) on a *fresh* environment that
   didn't build it, before calling it done. Several image-bloat and
   path-fragility bugs this session were only visible at this tier — a
   Dockerfile that "works" on the machine that built it can still ship
   broken.

**The rule that makes this work in practice: don't skip a tier to save
time.** Every real bug traced through this repo's docs (the `gcc-12-base`
mismatch, the `py4j` socket-timeout poisoning, the wheelhouse layer bloat)
was invisible at an earlier tier and only surfaced at a later one — the
tiers exist specifically because "worked in my dev environment" and
"works when shipped" are different claims.

## Where to go next

- Adding a new model: `docs/BRING_YOUR_OWN_MODEL.md`
- Setting this up locally or in a new environment: `docs/SETUP_GUIDE.md`
- Full directory layout: `docs/REPO_STRUCTURE.md`
- Airgapped deployment (waiter/kitchen): `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`
- Airgapped deployment (reusing an existing base image):
  `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`
- Testing tiers in full: `docs/TESTING_BEFORE_EXPORT.md`
