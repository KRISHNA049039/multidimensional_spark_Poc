#!/usr/bin/env bash
# Build the ner_translate wheelhouse — downloads this pipeline's full
# dependency set (including transitive deps) as local files, for two uses:
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
# Note: torch/torchvision are NOT in that requirements.txt (they come from
# the base image), but gliner/transformers declare torch as a dependency, so
# pip will download a torch wheel here too even though it'll go unused at
# install time (the base image's torch==2.6.0 already satisfies it, and pip
# install never re-installs an already-satisfied requirement). Wastes some
# local disk/bandwidth on the build machine in exchange for not having to
# hand-maintain a constraints file — an acceptable trade for a wheelhouse
# that isn't committed to git.
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
# refuses to cross-download sdists for a foreign platform (it can't
# guarantee they'd build the same way), so they need a separate, unrestricted
# pass. They're pure Python either way, so the sdist downloaded here builds
# fine later inside the Linux container at install time regardless of what
# platform ran this script.
SDIST_ONLY="odfpy ebooklib"
sdist_pattern=$(echo "$SDIST_ONLY" | tr ' ' '|')

echo "Downloading ner_translate's dependency wheelhouse to $OUT ..."
grep -viE "^($sdist_pattern)==" "$REQ_FILE" | grep -v '^#' | grep -v '^$' > /tmp/req_wheels_only.txt
pip download \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --implementation cp \
    --abi cp311 \
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
