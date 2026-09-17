# Updating `ner_translate` on Airgapped Systems Without a New CD

**Status: tested and working, 2026-09-17.** This is for the situation where
your airgapped system **already has `multi-model-inference:latest`**
loaded (Spark + torch/CUDA, no `ner_translate`-specific packages yet).
If that's true, you never need to burn a new CD with a multi-GB image again
for a `ner_translate` dependency change — only a few hundred MB of small
files, and the final image gets **built on the airgapped machine itself**.

## The idea in one picture

```
Internet-connected machine                    Airgapped machine
───────────────────────────                    ─────────────────
                                                 already has:
                                                 multi-model-inference:latest
                                                     (Spark + torch/CUDA)

build wheelhouse (~300MB)  ──┐
build .deb bundle (~25MB)  ──┼── small CD ──►   docker build (locally, offline)
package code (~100KB)      ──┘                       │
                                                       ▼
                                              ner-translate-worker:latest
                                              (built HERE, never shipped as
                                               a finished image)
```

The finished `ner-translate-worker:latest` image is never copied across the
airgap at all — only its three small ingredients are, and the airgapped
Docker daemon assembles it itself from the base image it already has.

## Why this works

`docker build` doesn't need the internet for two things it would normally
need it for, **as long as you feed it local files instead**:

1. **Python packages** (`pip install`) — normally hits PyPI. Fed from
   `wheels/ner_translate/` instead, via `pip install --no-index
   --find-links=...`. This part already existed before today.
2. **System packages** (`apt-get install tesseract-ocr ...`) — normally
   hits Ubuntu's package servers. Fed from `debs/ner_translate/` instead,
   via `dpkg -i`. **This part is new as of today.**

Everything else `Dockerfile.ner_translate` does (`FROM
multi-model-inference:latest`) just reuses the base image already sitting
on the airgapped machine — no download needed for that either.

## Step-by-step

### On the internet-connected machine

**1. Make sure you have a local copy of the exact same base image the
airgapped machine has.** Check with:
```bash
docker images multi-model-inference:latest
```
If it's not there, get it the same way the airgapped system originally
did (load it from whatever tarball/CD gave it that image). This step
matters more than it sounds — see the gotcha below.

**2. Build the Python wheelhouse** (skip if you already have an
up-to-date one — rerun only when `models/pipelines/ner_translate/requirements.txt`
changes):
```bash
bash deploy/scripts/build_ner_translate_wheelhouse.sh
```
Produces `wheels/ner_translate/` (~300MB).

**3. Build the system-package (`.deb`) bundle** — rerun whenever the
`apt-get install` package list in `Dockerfile.ner_translate` changes, or
whenever `multi-model-inference:latest` gets rebuilt:
```bash
bash deploy/scripts/build_ner_translate_debs.sh
```
Produces `debs/ner_translate/` (~25MB, ~50 `.deb` files: tesseract-ocr,
its language packs, poppler-utils, and everything they depend on).

**4. Package the code** (tiny — a few hundred KB):
```bash
tar czf ner-translate-code.tar.gz \
  inference/ models/__init__.py models/pipelines models/plugins \
  deploy/Dockerfile.ner_translate deploy/apply_wheels_hotfix.sh \
  wheels/ner_translate debs/ner_translate \
  models/pipelines/ner_translate/requirements.txt
```
(Or reuse `deploy/scripts/build_and_test_ner_translate_gpu.sh`'s packaging
step if you already have one — the point is: code + the two bundles above,
all together.)

**5. Verify it builds locally before burning anything**, exactly the way
the airgapped machine will do it:
```bash
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .
docker run --rm ner-translate-worker:latest tesseract --version
docker run --rm ner-translate-worker:latest python3 -c "import gliner, transformers, fastapi; print('OK')"
```

### Physical transfer

Burn/transfer: `wheels/ner_translate/` + `debs/ner_translate/` + the code
tarball. That's it — no `.tar.gz` Docker image. Total size: a few hundred
MB instead of multiple GB.

### On the airgapped machine

**1. Confirm the base image is there:**
```bash
docker images multi-model-inference:latest
```

**2. Extract the transferred files** into the repo checkout, so
`wheels/ner_translate/` and `debs/ner_translate/` land exactly where
`Dockerfile.ner_translate` expects them (sibling to `deploy/`).

**3. Build — entirely offline:**
```bash
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .
```

**4. Verify** (same commands as step 5 above) before relying on it.

## The one real gotcha (already hit and fixed)

**The `.deb` bundle must be built against the *exact same*
`multi-model-inference:latest` the airgapped machine has** — not a fresh,
generic `ubuntu:22.04`. `deploy/scripts/build_ner_translate_debs.sh`
already does this correctly (it downloads from inside a container `FROM
multi-model-inference:latest`), but it's worth understanding why, in case
you ever see this error again:

```
dpkg: dependency problems prevent configuration of libgomp1:amd64:
 libgomp1:amd64 depends on gcc-12-base (= 12.3.0-1ubuntu1~22.04.3); however:
  Version of gcc-12-base:amd64 on system is 12.3.0-1ubuntu1~22.04.
```

This happened the first time this was tried, using a plain `ubuntu:22.04`
container to download the `.deb`s: that fresh pull resolved against
*today's* Ubuntu package snapshot, which had a newer `gcc-12-base` than
whatever was baked into `multi-model-inference:latest` back when *that*
was built. `libgomp1` (needed by `tesseract-ocr`) demanded the exact newer
version, which wasn't present, and `dpkg` refuses to configure a package
against a dependency it can't find — even though every individual `.deb`
unpacked fine. Downloading from inside the actual target image sidesteps
this by construction: whatever versions are already there are what gets
resolved against.

**Practical rule:** if `multi-model-inference:latest` is ever rebuilt (new
base image, new CD), rerun `build_ner_translate_debs.sh` against the new
one before reusing an old `.deb` bundle — they're tied together.

## When you genuinely need a new full-image CD

Only when `multi-model-inference:latest` itself changes — a Spark/Java
version bump, a different torch/CUDA version, a new base OS. That's the
rare case; everything else about a `ner_translate` update (new Python
package, new tesseract language, a code fix) goes through the small
wheelhouse + `.deb` bundle + code path above instead.

## Relationship to the other architecture in this repo

This repo also has a **waiter/kitchen split**
(`docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`) that fully separates
Spark from CUDA into two independent images. That's the better fit when
you're starting from nothing (no compatible base image already on the
target machine) or want Spark and the model server to scale/update
completely independently. This doc's approach is the better fit when — like
here — you already have a compatible CUDA base sitting on the target
machine and just want to avoid re-shipping it.
