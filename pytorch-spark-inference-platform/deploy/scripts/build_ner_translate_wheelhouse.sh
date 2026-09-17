#!/usr/bin/env bash
# Build the ner_translate wheelhouse — downloads this pipeline's full
# dependency set (including transitive deps) as local files, targeting the
# Linux/cp311 container regardless of what platform runs this script. Two
# uses:
#
#   1. deploy/Dockerfile.ner_translate installs from this directory at BUILD
#      time via `pip install --no-index --find-links=wheels/ner_translate`,
#      instead of resolving live against PyPI on every build. Protects
#      against a pinned version disappearing from the index later (this
#      already happened once in this repo — see docs/CHANGELOG_20260913.md,
#      Dockerfile.worker's nightly torch pin).
#
#   2. The same file format can be copied into
#      wheels-hotfix/ner_translate/ on an already-deployed node to patch a
#      dependency without a full image rebuild — see
#      deploy/apply_wheels_hotfix.sh and wheels-hotfix/ner_translate/README.md.
#
# Run this on an INTERNET-CONNECTED machine before `docker build`, from the
# pytorch-spark-inference-platform/ directory:
#   bash deploy/scripts/build_ner_translate_wheelhouse.sh
#
# Rerun whenever models/pipelines/ner_translate/requirements.txt changes.
#
# torch/torchvision are NOT in that requirements.txt (they come from the
# base image) but gliner/transformers pull in torch>=2.0.0 transitively, so
# this download grabs SOME torch build too (whatever's newest under the
# platform below) even though it's never installed — the container's own
# `pip install` never touches an already-satisfied requirement without
# --upgrade, so an unused wheel here is just wasted bandwidth, not a
# correctness problem. Don't try to constrain this to the base image's
# exact torch==2.6.0 (a constraints file was tried and reverted — that
# exact version has no wheel under the platform tag below, so constraining
# to it made the whole download fail instead of saving bandwidth).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> pytorch-spark-inference-platform/

REQ_FILE=models/pipelines/ner_translate/requirements.txt
OUT=wheels/ner_translate

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: $REQ_FILE not found — run this from pytorch-spark-inference-platform/" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# odfpy/ebooklib ship no wheels at all (pure-Python, sdist-only) — pip
# refuses to cross-download sdists for a foreign platform when
# --platform/--python-version/etc are set (it can't guarantee they'd build
# the same way), so the top-level packages need a separate, unrestricted,
# --no-deps pass. Their sdists are pure Python either way, so they build
# fine later inside the Linux container at install time regardless of what
# platform ran this script.
#
# --no-deps is right for THEM but wrong for their OWN dependencies: odfpy
# needs defusedxml; ebooklib needs lxml + six (PyPI's JSON metadata reported
# requires_dist: None for BOTH — wrong/incomplete, discovered only by real
# build failures). Those go through the SAME cross-platform-targeted call
# as everything else below, not a second unrestricted download — lxml has
# real compiled extensions, and downloading it unrestricted on Windows
# previously grabbed a win_amd64 wheel that would have been silently
# useless inside the Linux container.
SDIST_ONLY="odfpy ebooklib"
sdist_pattern=$(echo "$SDIST_ONLY" | tr ' ' '|')
SDIST_ONLY_DEPS="defusedxml lxml six"

# huggingface-hub (pulled in transitively by gliner/transformers) requires
# hf-xet on x86_64/amd64/arm64/aarch64 — NOT gated behind an extra, a real
# Requires-Dist. `pip download`'s environment-marker evaluation for
# platform_machine uses the HOST machine's platform.machine(), same as
# `pip install` would — it's not overridden by --platform (that only
# constrains wheel *tags*, not marker evaluation). Windows reports
# "AMD64" for platform.machine(); the marker checks lowercase "amd64"/
# "x86_64", so building this wheelhouse from Windows silently evaluates
# the marker False and skips hf-xet entirely — pip install then fails
# inside the (correctly-marked, Linux x86_64) container at build time,
# not here. Force it in explicitly rather than relying on marker
# evaluation to agree with the target platform.
SDIST_ONLY_DEPS="$SDIST_ONLY_DEPS hf-xet"

echo "Downloading ner_translate's dependency wheelhouse to $OUT ..."
grep -viE "^($sdist_pattern)==" "$REQ_FILE" | grep -v '^#' | grep -v '^$' > /tmp/req_wheels_only.txt
for dep in $SDIST_ONLY_DEPS; do echo "$dep" >> /tmp/req_wheels_only.txt; done

# Platform/interpreter targeting for the container (Linux, Python 3.11),
# regardless of what platform builds this wheelhouse:
#   --platform manylinux_2_28_x86_64 — NOT the older manylinux2014
#     (glibc>=2.17): newer packages (onnxruntime 1.17+, needed for real
#     numpy-2.x support) only publish manylinux_2_28 wheels, and pip's
#     cross-platform match does NOT treat a newer target as a superset of
#     older tags — packages that only ship manylinux2014 wheels at their
#     currently-pinned version had to be bumped to a version that also
#     publishes manylinux_2_28 (sentencepiece 0.2.0->0.2.2, tiktoken
#     0.8.0->0.12.0). BUT other packages (tokenizers, pinned transitively by
#     transformers, among others) publish ONLY the older manylinux2014 tag
#     even in their latest release — no version bump fixes that. --platform
#     can be repeated to accept wheels matching ANY of multiple tags, so
#     pass both instead of picking one and chasing version bumps forever.
#     (Tried blaming --abi cp311 first — packages using Python's abi3
#     stable-ABI tag, like tokenizers' cp39-abi3, seemed like the
#     explanation. Verified wrong by testing in isolation: removing --abi
#     alone did NOT fix it; passing both --platform values did. Don't trust
#     a plausible-sounding diagnosis without testing it isolated from
#     everything else changing at the same time.)
#   --python-version 311 + --implementation cp, no --abi — letting pip
#     compute the full compatible tag set itself (cp311-cp311 through
#     cp39-abi3) rather than demanding one exact ABI tag.
pip download \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --python-version 311 \
    --implementation cp \
    --only-binary=:all: \
    -r /tmp/req_wheels_only.txt \
    -d "$OUT"

for pkg in $SDIST_ONLY; do
    spec=$(grep -iE "^${pkg}==" "$REQ_FILE" | sed 's/#.*//' | xargs || true)
    if [ -n "$spec" ]; then
        echo "Downloading sdist-only package: $spec"
        pip download --no-deps "$spec" -d "$OUT"
    fi
done
rm -f /tmp/req_wheels_only.txt

count=$(ls "$OUT" | wc -l)
echo "Done: $count files in $OUT"
echo "Next: docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate ."
