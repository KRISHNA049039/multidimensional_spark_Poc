#!/usr/bin/env bash
# Build the ner_translate .deb bundle — downloads tesseract-ocr + its
# language packs + poppler-utils AND their full apt dependency closure as
# local .deb files, targeting the exact Ubuntu 22.04 base every image in
# this repo builds from (deploy/Dockerfile's spark-base stage). Sibling to
# build_ner_translate_wheelhouse.sh, same reasoning, for system packages
# (apt) instead of Python packages (pip):
#
#   deploy/Dockerfile.ner_translate installs from this directory via
#   `dpkg -i debs/ner_translate/*.deb` at BUILD time, instead of
#   `apt-get install` hitting live apt repos. Combined with the existing
#   wheelhouse, this lets `docker build -f deploy/Dockerfile.ner_translate`
#   run ENTIRELY OFFLINE on a system that already has multi-model-inference:
#   latest cached locally (e.g. an airgapped box that received it in an
#   earlier CD) — no new pre-built image ever needs to cross the airgap
#   again for a dependency-only update, only this bundle + the wheelhouse +
#   code (tens of MB + a few hundred MB + KB, vs multiple GB).
#
# Run this on an INTERNET-CONNECTED machine with Docker, from the
# pytorch-spark-inference-platform/ directory:
#   bash deploy/scripts/build_ner_translate_debs.sh
#
# Rerun whenever Dockerfile.ner_translate's or
# Dockerfile.ner_translate_server's apt package list changes, OR whenever
# multi-model-inference:latest itself gets rebuilt (its already-installed
# package versions are what this bundle must match — see below). Downloads
# happen inside a throwaway container FROM multi-model-inference:latest
# itself (not this host, and not a generic ubuntu:22.04 — see below) so the
# .deb files are guaranteed compatible with the actual target regardless of
# what OS runs this script — same cross-platform-targeting idea as the
# wheelhouse script's --platform flags, just via a container instead of pip
# flags since apt has no native cross-platform download mode. Requires
# multi-model-inference:latest to already exist locally (`docker images` to
# check) — build/load it first if it doesn't.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=debs/ner_translate
rm -rf "$OUT"
mkdir -p "$OUT"

# Keep this list in sync with deploy/Dockerfile.ner_translate's and
# deploy/Dockerfile.ner_translate_server's apt-get install line.
PACKAGES="tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-kan tesseract-ocr-ben tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-mal tesseract-ocr-urd poppler-utils"

echo "Downloading .deb bundle (tesseract-ocr + language packs + poppler-utils, full dependency closure) to $OUT ..."
# Download from INSIDE multi-model-inference:latest itself, not a generic
# ubuntu:22.04 — the two drift apart over time (that image was built from
# whatever Ubuntu 22.04 apt snapshot existed at ITS build time, a fresh
# ubuntu:22.04 pull resolves against TODAY's snapshot). Discovered the hard
# way: a fresh ubuntu:22.04's libgomp1 pin (12.3.0-1ubuntu1~22.04.3) didn't
# match the older gcc-12-base (12.3.0-1ubuntu1~22.04, no patch suffix)
# already baked into multi-model-inference:latest, and dpkg refuses to
# configure a package against a dependency version it doesn't have —
# `tesseract-ocr` -> `libtesseract4` -> `libgomp1` -> exact `gcc-12-base`
# pin, so the whole chain failed to configure even though every .deb
# unpacked fine. Downloading against the ACTUAL target image's already-
# installed package versions avoids this by construction.
if ! docker image inspect multi-model-inference:latest >/dev/null 2>&1; then
    echo "ERROR: multi-model-inference:latest not found locally — build/load it first" >&2
    exit 1
fi
# `pwd -W` (not plain `pwd`) on Git Bash for Windows: Docker Desktop expects
# a Windows-style path (D:/...) for -v bind mounts, not MSYS's POSIX-style
# (/d/...) — the latter silently produces a mount that behaves like a
# non-directory inside the container (`cp: target '/out/' is not a
# directory`) rather than a clear error. `pwd -W` is a no-op (identical to
# plain `pwd`) on real Linux/macOS, so this line is safe on every platform
# this script runs on.
HOST_OUT="$(pwd -W 2>/dev/null || pwd)/$OUT"
docker run --rm -v "${HOST_OUT}:/out" multi-model-inference:latest bash -c "
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends --download-only $PACKAGES
    cp /var/cache/apt/archives/*.deb /out/
"

count=$(ls "$OUT"/*.deb 2>/dev/null | wc -l)
echo "Done: $count .deb files in $OUT"
echo "Next: docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate ."
