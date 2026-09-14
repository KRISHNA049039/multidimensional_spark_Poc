#!/usr/bin/env bash
# Dependency hotfix installer — runs at CONTAINER START, before Spark
# master/worker launches. If /wheels-hotfix has any .whl files in it,
# installs them; otherwise it's a no-op and the container runs exactly as
# built.
#
# This exists for one specific case: patching a dependency on an
# ALREADY-DEPLOYED air-gapped node without a full rebuild + re-save +
# re-transfer cycle. It mirrors the code-update-without-rebuild pattern this
# repo already uses for source changes (docs/air_gapped_dep.md §7's bind
# mount) — same idea, applied to dependencies instead of code.
#
# This is NOT the primary way dependencies get into the image — that's
# still deploy/scripts/build_ner_translate_wheelhouse.sh baked in at build
# time via `pip install --no-index --find-links=wheels/ner_translate`. Using
# this as the default/primary path instead would mean every container start
# re-installs packages (slow — torch alone is ~2GB) and would make the
# running environment depend on host-side state the image tag can't
# describe, both of which matter more here than usual: this repo has
# already hit a real driver/worker Python-environment-mismatch bug once
# (docs/FRAMEWORK_OVERVIEW.md), and a wheels-hotfix mount that's present on
# one node but not another (or has drifted) reproduces exactly that failure
# mode. Use it deliberately, on one node at a time, and keep the SAME
# wheels dropped on every node in the cluster if the fix needs to apply
# cluster-wide — this script cannot detect or warn about that drift itself.
#
# Usage: exec this before the Spark start command, e.g. in a compose
# `command:` block:
#   bash -c "/usr/local/bin/apply_wheels_hotfix.sh && \$SPARK_HOME/sbin/start-master.sh ..."
set -euo pipefail

HOTFIX_DIR="${WHEELS_HOTFIX_DIR:-/wheels-hotfix}"

if [ -d "$HOTFIX_DIR" ] && [ -n "$(find "$HOTFIX_DIR" -maxdepth 1 -name '*.whl' -print -quit 2>/dev/null)" ]; then
    echo "[wheels-hotfix] Found wheel(s) in $HOTFIX_DIR — installing as an override:"
    ls -1 "$HOTFIX_DIR"/*.whl
    pip install --no-cache-dir --no-index --find-links="$HOTFIX_DIR" "$HOTFIX_DIR"/*.whl
    echo "[wheels-hotfix] Override install complete."
else
    echo "[wheels-hotfix] No wheels found in $HOTFIX_DIR — running with the image's baked-in dependencies."
fi
