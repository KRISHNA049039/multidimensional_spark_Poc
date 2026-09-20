# Complete Architecture Reference

One document, the full picture: what this framework is, how every piece
fits together, and where to go deeper on any one of them. Start here if
you're new to the repo or need to see the whole shape before diving into
a specific doc.

## 1. The core idea: BYOM on Spark

This is a framework for running PyTorch inference **distributed across a
Spark cluster**, where the model being run is a plug-in, not hardcoded.
Two contracts exist for "bring your own model," because two genuinely
different kinds of model show up in practice:

```
models/
├── plugins/     — SIMPLE: one nn.Module, fixed-shape float32 tensor in/out
│                  (image classifiers, detectors, signal models)
│                  → inference/cluster_engine.py, submit_job.py
└── pipelines/   — COMPLEX: multi-model, file/text in, structured out
                   (ner_translate: OCR → language-detect → translate → NER)
                   → inference/text_pipeline_engine.py, submit_pipeline_job.py
```

Execution, for both, is Spark's `mapPartitions` — **not**
`torch.distributed`/DDP, and deliberately so: every workload here is
forward-pass-only batch inference, never training, so there's no gradient
to synchronize across workers and nothing DDP's process-group machinery
would buy. A model's weights are broadcast once (`sc.broadcast`), each
partition loads its own copy and scores its own data slice — fully
independent, no cross-partition communication needed. Full reasoning:
`docs/PROJECT_OVERVIEW_AND_CONCEPTS.md`.

## 2. Dependency isolation — why every model doesn't share one environment

The naive approach (one shared Python environment for every model) broke
for real once: installing `ner_translate`'s deps silently bumped `numpy`
for every other model. Two fixes exist, and which to use depends on
context:

### Option A — one image per model, in-process (`docs/MODEL_CONTAINER_ISOLATION.md`)
```
multi-model-inference:latest (shared base: Spark+Java+torch/CUDA+core deps)
        │
        └─ FROM ──> ner-translate-worker:latest
                     (+ tesseract/poppler, + gliner/transformers, own cluster)
```
Simplest, stays Spark-native. Used when a compatible base image already
exists somewhere you want to reuse it.

### Option B — waiter/kitchen HTTP split (`docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`)
```
spark-lean:latest ("waiter")          ner-translate-server:latest ("kitchen")
Spark master/worker only,        <──   owns 100% of torch/CUDA/model deps,
ZERO ML dependencies                   answers HTTP /predict requests
```
The waiter image is identical and reusable across *every* future
kitchen-backed model — adding an 11th pipeline never touches it, only
builds one new kitchen. Fully verified end-to-end on real GPU infra (see
`docs/TEST_RESULTS_AND_TRADEOFFS.md` §1).

**Which to use**: Option A if a compatible base already exists where
you're deploying (cheaper to extend). Option B if starting fresh, or if
Spark and the model server need to scale/update completely
independently.

## 3. Shipping to airgapped systems without re-shipping gigabytes

Both isolation options share the same underlying trick for cheap
dependency updates — **never ship a rebuilt multi-GB image for a small
change**:

| Ingredient | What it replaces | Built by |
|---|---|---|
| Wheelhouse (`wheels/<pipeline>/`) | Live `pip install` | `deploy/scripts/build_*_wheelhouse.sh` — `pip download --platform manylinux... --only-binary=:all:`, works cross-platform from any host |
| `.deb` bundle (`debs/<pipeline>/`) | Live `apt-get install` | `deploy/scripts/build_ner_translate_debs.sh` — downloaded from *inside* the actual target base image, not a generic one (version-skew gotcha, see §6) |

Both get fed into a Dockerfile via `RUN --mount=type=bind`, never `COPY`
— a bind mount is visible only to that build step and never becomes a
permanent image layer, unlike `COPY ... && rm -rf ...`, which looks like
cleanup but isn't (the deleted layer's bytes still ship). This distinction
cost ~6GB of accidental image bloat once before being caught — see
`docs/TEST_RESULTS_AND_TRADEOFFS.md` §2 for the full bug list.

**End result**: a dependency-only update is a wheelhouse + `.deb` bundle +
code tarball (hundreds of MB, not GB), built into the final image **on
the target machine itself**, reusing whatever base is already there.
Full walkthrough: `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`.
Testing this claim locally before trusting it on a real airgapped box:
`docs/LOCAL_AIRGAPPED_SIMULATION_TEST.md`.

## 4. Concurrency & GPU scheduling layer

Four pieces, composing into one concurrent-GPU-inference story, all
**require zero image changes** — pure code/config against dependencies
already present in `multi-model-inference:latest`
(`docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md`):

```
Spark task scheduling  →  intra-process concurrency  →  cross-process sharing  →  the inference call itself
  spark.task.resource.     torch.cuda.Stream() —          NVIDIA MPS —              predict_batch_udf —
  gpu.amount (opt-in,       genuinely overlaps            multiple OS processes     Pandas-UDF path,
  NOT default — wrong       independent models'           sharing one GPU           returns real
  for the waiter/kitchen    kernels instead of             efficiently instead      predictions (RDD
  split, where Spark        the default stream             of driver time-slicing   path discards them)
  never touches a GPU)      serializing them                                        
```

All four verified working on real GPU infra (`docs/TEST_RESULTS_AND_TRADEOFFS.md`
§4) — **verified correct, not all verified faster**: the UDF engine was
actually ~3.4x slower than the RDD engine at the tested scale (trades
speed for returning real predictions), and streams/MPS were confirmed to
work without breaking anything but not benchmarked against a no-streams/
no-MPS baseline. Read the tradeoffs doc before assuming any of these four
is a performance win by default.

## 5. Testing discipline

Not classical TDD, but an equivalent discipline serving the same
purpose — catch a broken assumption at the cheapest point, not the most
expensive one (`docs/TESTING_BEFORE_EXPORT.md`, and the general version in
`docs/PROJECT_OVERVIEW_AND_CONCEPTS.md`):

```
Tier 0: local, no Docker      →  cheapest, catches logic bugs
Tier 1/2: single container,   →  catches packaging bugs
          then local cluster
Tier 3: real multi-node       →  catches distributed-execution bugs
        cluster (or AWS)         (the py4j socket-timeout bug only
                                   showed up here, twice)
Tier 4: export + reload on    →  catches "works on the machine that
        a FRESH environment      built it but not elsewhere" bugs
```

**The rule that makes this work**: don't skip a tier to save time. Every
real bug found this session (the `gcc-12-base` mismatch, the py4j
timeout, the wheelhouse layer bloat, the `docker build --target`
default-stage bug, the `pandas_udf`/postponed-annotations conflict) was
invisible at an earlier tier and only surfaced at a later one.

## 6. Known gotchas — read before rediscovering them the hard way

| Gotcha | Where it bites | Doc |
|---|---|---|
| `docker build` without `--target` builds the LAST stage in the file, not "the important one" | Silently built `spark-lean` instead of `multi-model-inference` after `lean` was added after `final` in the Dockerfile | `docs/TEST_RESULTS_AND_TRADEOFFS.md` §5 |
| Global `socket.setdefaulttimeout()` left unset poisons py4j's own gateway socket | `Py4JNetworkError`/raw `TimeoutError` deep inside `JavaSparkContext.__init__`, looks completely unrelated to sockets | `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md` |
| `.deb` bundle built from a generic base instead of the actual target image | `dpkg: dependency problems` — exact package version mismatch | `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md` |
| `from __future__ import annotations` breaks `pandas_udf`'s signature detection | `PySparkNotImplementedError: [UNSUPPORTED_SIGNATURE]` | `docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md` §2 |
| `spark.createDataFrame` schema inference over deeply-nested tensor rows | Hangs for minutes, not an error — looks like it's just slow | `docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md` §1 |
| CRLF line endings on `.sh`/Dockerfiles | Broken shebangs on Linux, only on a Windows dev machine without `.gitattributes` | `.gitattributes` at repo root |
| A prior `docker save` export sitting in the build context | Silently copied into the *next* build via `COPY .`/bind-mount-copy, ballooning size | `.dockerignore`'s `*.tar.gz` rule |

## 7. Map of every doc in this repo, by question

| Question | Doc |
|---|---|
| "What is this project, conceptually?" | `docs/PROJECT_OVERVIEW_AND_CONCEPTS.md` |
| "How do I set it up / run it?" | `docs/SETUP_GUIDE.md` |
| "How do I add a new model?" | `docs/BRING_YOUR_OWN_MODEL.md` |
| "Why two isolation options, which do I pick?" | `docs/MODEL_CONTAINER_ISOLATION.md` |
| "How do I deploy the waiter/kitchen split airgapped?" | `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md` |
| "How do I update deps airgapped without re-shipping an image?" | `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md` |
| "How do I test the offline path before trusting it on a real airgapped box?" | `docs/LOCAL_AIRGAPPED_SIMULATION_TEST.md` |
| "What's the concurrency/UDF/streams/MPS architecture?" | `docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md` |
| "What was actually tested, and what did it prove vs. not prove?" | `docs/TEST_RESULTS_AND_TRADEOFFS.md` |
| "What's the exact directory layout?" | `docs/REPO_STRUCTURE.md` |
| "What's the testing tier checklist in full?" | `docs/TESTING_BEFORE_EXPORT.md` |

This doc is the map; the others are the territory. When something here
goes stale (a doc gets renamed, an approach gets replaced), update this
table — it's the one place meant to stay a reliable index.
