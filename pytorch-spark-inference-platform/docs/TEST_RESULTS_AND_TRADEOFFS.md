# Test Results & Tradeoffs

Every test actually run in this session, with real numbers and honest
tradeoffs — including the ones that didn't come out flattering. Where a
test showed something worth thinking about (a slowdown, an unquantified
claim, an environment limit), that's called out explicitly rather than
smoothed over.

## 1. `ner_translate` pipeline correctness (waiter/kitchen split, Option B)

**Test**: full pipeline (OCR → language detection → translation → NER)
over 6 real input files, on AWS `g4dn.xlarge` (Tesla T4), routed entirely
through the HTTP kitchen — `spark-lean:latest` (waiter, no torch) calling
`ner-translate-server:latest` (kitchen, torch/CUDA) over `/predict`.

**Result** — all correct, all recognizable/sane for the input content:

| File | Language | Translated | Entities |
|---|---|---|---|
| `sample_scan6.png` (OCR) | en | No | 8 |
| `sample_text1.txt` | en | No | 28 |
| `sample_text2.txt` | en | No | 33 |
| `sample_text3.txt` | en | No | 6 |
| `sample_text4.txt` | hi | Yes | 14 |
| `sample_text5.txt` | mr | Yes | 24 |

**Tradeoff**: the HTTP hop (Spark → kitchen) adds network + (de)serialization
latency per batch, and loses Spark's data-locality benefit versus loading
the model directly in the executor. Not measured as a separate number in
this run — the tradeoff is architectural (see
`docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`), accepted because the
payoff (zero torch/CUDA in the Spark image, full dependency isolation) is
worth more than the per-batch latency cost for this batch-inference
workload.

## 2. Image size — three real bloat bugs found and fixed

**Before** (first successful build, before auditing size): **11.9GB**
combined (`spark-lean` 6.0GB + `ner-translate-server` 5.9GB).
**After**: **3.82GB** combined (837MB + 3.0GB) — a 68% reduction, found by
actually inspecting `docker history` layer-by-layer rather than assuming
the images were already lean.

| Bug | Root cause | Fix | Tradeoff introduced |
|---|---|---|---|
| Wheelhouse baked into "lean" image | `COPY . .` in the no-torch stage also copied `wheels/<pipeline>/` (GBs of unused `.whl` files) | `RUN --mount=type=bind` + explicit `find ... ! -name wheels` instead of `COPY . .` | Slightly more Dockerfile complexity than a plain `COPY`; the mount syntax needs `# syntax=docker/dockerfile:1` |
| Wheelhouse shipped twice in kitchen image | `COPY wheels/... + rm -rf` — Docker layers are additive, so `rm -rf` doesn't remove the bytes from what ships | Bind mount instead of `COPY`, so the wheelhouse never becomes a layer at all | None — strictly better, no downside found |
| A rebuild copied its own prior export | `docker save` output (`*.tar.gz`) sitting in the build context got swept into the next build via the same bind-mount-copy | Added `*.tar.gz` to `.dockerignore` | None |

**Tradeoff worth naming**: none of these three bugs were caught by manual
review — all three were only visible by actually running `docker history`
and inspecting layer sizes after a real build. This is a process gap, not
just a code gap: image-size auditing wasn't part of the normal workflow
before this session, despite avoiding large images being a stated priority
the whole time.

## 3. Offline dependency update via `.deb` bundle (reusing an existing base image)

**Test**: build `ner-translate-worker:latest` **entirely offline-style** —
using only a wheelhouse + a new `.deb` bundle + code, against an
already-present `multi-model-inference:latest`, with no live `apt-get`/pip
index access during the actual pipeline-image build.

**First attempt — failed, real bug**: `.deb` bundle built from a generic
`ubuntu:22.04` container →
```
dpkg: dependency problems prevent configuration of libgomp1:amd64:
 libgomp1:amd64 depends on gcc-12-base (= 12.3.0-1ubuntu1~22.04.3); however:
  Version of gcc-12-base:amd64 on system is 12.3.0-1ubuntu1~22.04.
```
Root cause: a fresh `ubuntu:22.04` pull resolves against *today's* apt
snapshot, which had drifted from whatever snapshot `multi-model-inference:latest`
was actually built against. **Fix**: download the `.deb` bundle from
*inside* `multi-model-inference:latest` itself, not a generic base —
guarantees the bundle matches what's already installed, by construction.

**Second attempt — succeeded**: 50 `.deb` files (~23MB, down from 66/~40MB
in the failed attempt — fewer needed since the real base already had many
shared libs). Verified:
```
tesseract 4.1.1, poppler 22.02.0, gliner/transformers/fastapi import OK,
torch 2.6.0+cu126 present
```

**Tradeoff**: the `.deb` bundle is now *tied to a specific base image
version* — if `multi-model-inference:latest` is ever rebuilt, the bundle
must be regenerated against the new one before reuse, or the same version
mismatch recurs. This is a real, ongoing maintenance coupling, not a
one-time cost — documented explicitly in
`docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`'s gotcha section so it
isn't rediscovered the hard way again.

## 4. Concurrency & UDF enhancements — all four tested on real GPU (Tesla T4)

### 4a. RDD engine vs. UDF engine (`predict_batch_udf`)

**Test**: `resnet18`, 64 samples, 2 partitions, `--mode gpu_only`, both
engines, same hardware, same run.

| Engine | Elapsed | Throughput | Returns real predictions? |
|---|---|---|---|
| RDD (`cluster_engine.py`, existing) | **5.36s** | 11.9/s | No — discards output, counts samples only |
| UDF (`cluster_engine_udf.py`, new) | **18.01s** | 3.6/s | **Yes** — full per-sample predictions |

**Honest tradeoff — the UDF path was ~3.4x slower here, not faster.** This
is a real result, not a caveat to bury: at this small a scale (64 samples),
Arrow schema setup, `pandas_udf` dispatch overhead, and — critically —
actually serializing real prediction values back to the driver (which the
RDD path never does at all) outweigh whatever per-batch serialization
efficiency Arrow provides over pickle. This does **not** mean the UDF path
is a mistake — it means the UDF path isn't a drop-in performance upgrade;
it trades speed at small scale for a genuine new capability (real
predictions) the RDD path doesn't have, and its actual throughput
advantage (if any) at larger, more realistic batch sizes was **not tested
here** — this comparison used a small sample count specifically to test
correctness quickly, not to benchmark at scale. Anyone deciding between the
two engines for a real workload should re-run this comparison at their
actual data volume before choosing based on speed alone.

### 4b. Real CUDA streams (`distributed_gpu.py`'s `run_models_on_streams`)

**Test**: all 10 registered tensor models, 2 partitions, real GPU, via
`run_benchmark.py --mode distributed`.

**Result**: all 10 models completed successfully per partition —
`model_load: 2.19–2.21s`, `inference: 1.24s` (combined, all 10 models, per
partition, ~5,350 samples/partition). Per-model throughput was sane and
non-degenerate for every model (ranging ~210/s for YOLO to ~240,964/s for
the lightweight signal denoiser) — no zero/negative/corrupted values,
which would indicate broken CUDA-event timing.

**Honest tradeoff — no before/after speed comparison exists.** The "before"
state wasn't a slower implementation of the same feature — it was a
docstring *claiming* stream-based concurrency with zero actual
`torch.cuda.Stream()` calls anywhere in the file. So this test confirms
the feature now genuinely exists and produces correct results; it does
**not** demonstrate a measured speedup over the old sequential loop,
because a true apples-to-apples timing comparison (same models, same data,
streams vs. no streams, isolated from everything else) wasn't run. Streams
help most when the GPU has spare capacity a sequential job isn't using
(small/medium models, memory-bound ops) — for 10 models with very
different sizes (some as small as `[128]` input, one as large as
`[3,640,640]`), the actual overlap achieved wasn't isolated/measured here.

### 4c. NVIDIA MPS (multi-process GPU sharing)

**Test**: started the MPS daemon on the host, then ran **two separate
containers concurrently** (not sequentially) against the same physical
GPU — `resnet18` (256 samples) and `mobilenetv3` (256 samples) — both
pointed at the shared `CUDA_MPS_PIPE_DIRECTORY`.

**Result**: both completed successfully, at nearly the same time
(container 1: 6.27s, container 2: 5.89s, both finishing within ~1 second
of each other in wall-clock terms) — confirming they ran **concurrently**,
not one blocking the other.

**Honest tradeoff — this confirms MPS doesn't break anything, not that it
measurably helps.** No comparison run was done *without* MPS to see
whether the two containers would have taken meaningfully longer sharing
the GPU via the driver's default time-slicing instead. The result rules
out "MPS causes conflicts/corruption" — it does not, by itself, prove the
efficiency gain MPS is supposed to provide. Quantifying that would need a
dedicated with/without-MPS throughput comparison, not run in this pass.

### 4d. Spark GPU discovery script

**Test**: `deploy/scripts/gpu_discovery.sh`, run directly on the GPU
instance.

**Result**: `{"name": "gpu", "addresses": ["0"]}` — valid, correctly
matches Spark's documented discovery-script JSON contract (string
addresses, not bare integers — a detail that's easy to get wrong and
silently break Spark's parsing of it).

**Tradeoff / scope gap**: this confirms the *script* is correct in
isolation. It does **not** confirm Spark's own scheduling config
(`spark.task.resource.gpu.amount`, fractional GPU sharing across tasks)
actually works end-to-end, since that requires a real multi-worker
standalone cluster invoking this script itself — not set up in this pass.
This is the one piece from `docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md`
that's "should work" rather than "verified."

## 5. Two build-process bugs caught during this session's testing (not the pipeline's own bugs)

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| `docker build` without `--target` | `multi-model-inference:latest` came out as an 879MB no-torch image instead of the expected ~11GB torch/CUDA image | Docker builds the **last** stage in a Dockerfile by default; adding `lean` *after* `final` earlier in this session silently changed what a target-less build produces | Always pass `--target final` explicitly for this image going forward |
| Local Docker Desktop/WSL2 crash | `docker run` for the UDF test died mid-execution with `unexpected EOF`, then the whole Docker Desktop engine became unresponsive, then even `wsl --shutdown` itself hung | Repeated OOM kills on a memory-constrained (7.85GB RAM) Windows dev machine running Spark+Arrow+torch in one container left WSL2's VM state wedged | Required a full machine reboot — no software-only fix recovered it |

## Summary: what these tests do and don't establish

**Established with real evidence:**
- The `ner_translate` pipeline produces correct results end-to-end, on real GPU infra, via the waiter/kitchen split.
- Three specific image-bloat bugs existed and are fixed, with before/after sizes.
- The offline `.deb`-bundle update path works, including the exact version-skew failure mode to watch for.
- All four concurrency/UDF enhancements function correctly (no crashes, correct outputs) on a real Tesla T4.

**Explicitly NOT established — decide with this in mind, not assumed:**
- That the UDF engine is faster than the RDD engine (it was slower at the tested scale).
- That real CUDA streams measurably speed up multi-model inference versus the old sequential loop (no isolated before/after timing).
- That MPS provides a measurable throughput benefit versus no MPS (only "doesn't break" was confirmed).
- That Spark's own GPU-aware task scheduling works against a real multi-worker cluster (only its file dependency was checked).
