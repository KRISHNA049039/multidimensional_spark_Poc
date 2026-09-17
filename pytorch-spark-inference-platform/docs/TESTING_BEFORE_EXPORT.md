# Testing `ner_translate` Before Exporting for Air-Gapped Transfer

An ordered checklist — each tier only worth running once the one before it
passes. Mirrors `docs/air_gapped_dep.md`'s own Phase 1 "build → verify →
export" structure, applied to the new per-pipeline image from
`docs/CHANGES_SUMMARY.md` Parts 2-3. Stop and fix before moving to the next
tier — no point building/exporting an image whose pipeline logic hasn't
even been proven locally.

## Internet-machine steps run here vs. their airgapped equivalent

Every "Update — 2026-09-16" block below records a real run on a Windows
laptop (D: drive, 7.85GB RAM — see each Tier's update for what that
constrained). Quick map of what happened on that internet-connected
machine against what the airgapped target does instead:

| Phase | Ran on this internet machine (2026-09-16) | Airgapped target does instead |
|---|---|---|
| Docker storage | Relocated Docker Desktop's WSL2 disk from C: (7.8GB free — too little) to D: via junction (Tier 2 update, below) | Not applicable — just needs enough local disk; see `docs/air_gapped_dep.md` §2 Storage row |
| Model weights | `snapshot_download()` for `gliner-multi` + `nllb-200-distilled-600M` into `models/weights/` (Tier 0 update, below) | Never downloads anything — weights are already baked into the image at `docker build` time on *this* machine; airgapped nodes only `docker load` |
| Wheelhouse | `bash deploy/scripts/build_ner_translate_wheelhouse.sh`, plus a real fix for a missing `hf-xet` wheel (Tier 0 update) | N/A — wheelhouse is a build-time-only artifact, never shipped to or used on the airgapped side directly (it's already inside the image) |
| Base image build | `docker build -f deploy/Dockerfile` — succeeded, 10.5GB (Tier 2 update) | `docker load` from the transferred `.tar.gz`, per `docs/internet_to_airgapped_transfer.md` §3.3 |
| Pipeline image build | `docker build -f deploy/Dockerfile.ner_translate` — **blocked by this host's RAM**, not completed (Tier 2 update) | Same build command works on a host meeting the 8GB+ minimum — once it succeeds here, export/transfer is identical to the base image's |
| Export + transfer | Not reached (pipeline image didn't finish building) | `docker save` → `gzip` → USB/data-diode/approved transfer → `gunzip \| docker load`, per Tier 4 below and `docs/air_gapped_dep.md` §4 |
| Cluster + job | Not reached | `docker compose -f deploy/docker-compose.ner_translate.yml up -d`, then `submit_pipeline_job.py` exactly as in Tier 3 below — no network needed since weights are already in the image |

## Tier 0 — Blockers (checked 2026-09-14, still true)

| Blocker | Status | Fix |
|---|---|---|
| Model weights not populated | `models/weights/` has only `README.md` | Run the two `from_pretrained(...).save_pretrained(...)` commands in `models/weights/README.md` — needs internet, ~2-3GB total |
| Docker Desktop not running | Confirmed via `docker info` | Start Docker Desktop before Tier 2+ |
| OCR/PDF path never exercised | `data/ner_samples/` was text-only | **Fixed** — added `data/ner_samples/sample_scan6.png`, a synthetic image with person/org/location/phone entities, verified legible |

Nothing below Tier 1 is worth running until weights are populated.

### Update — 2026-09-16, verified on Windows (D: drive, 7.85GB RAM host)

Weights populated on this machine: `models/weights/gliner-multi` and
`models/weights/nllb-200-distilled-600M`. `huggingface_hub`'s default
symlink-based cache fails on Windows without Developer Mode/admin
(`OSError: [WinError 1314]` — `GLiNER.from_pretrained().save_pretrained()`
hits this). Fix used: `snapshot_download(repo_id, local_dir=...)` directly
(no `local_dir_use_symlinks` needed on `huggingface_hub` 0.36.2 — that
param is deprecated and copies files, not symlinks, unconditionally now).
Both models verified to load offline afterward
(`GLiNER.from_pretrained('models/weights/gliner-multi')` /
`AutoModelForSeq2SeqLM.from_pretrained('models/weights/nllb-200-distilled-600M')`).

New real bug found and fixed: `deploy/scripts/build_ner_translate_wheelhouse.sh`
was silently dropping `hf-xet` (a hard `Requires-Dist` of `huggingface-hub`
0.36.2 on x86_64/amd64/arm64/aarch64, not gated behind an extra). `pip
download`'s environment-marker evaluation uses the *host* machine's
`platform.machine()`, not `--platform`'s wheel-tag target — Windows
reports `"AMD64"`, which doesn't match the marker's `"x86_64"`/`"amd64"`
checks, so the marker silently evaluates `False` when building the
wheelhouse from Windows and `pip install --no-index` fails inside the
(correctly Linux/x86_64) container instead. Fixed by forcing `hf-xet`
into the explicit download list alongside the existing sdist-only
handling — see the script's own comment for the full explanation.
`wheels/ner_translate` now has 51 files (was 50).

## Tier 1 — Local, no Docker (fastest feedback)

Exercises: language detection, direct-path (English), translated-path
(Hindi `sample_text4.txt`, Marathi `sample_text5.txt` — already non-English
in the existing samples, so translation was already covered), phone-number
regex, entity dedup. Does **not** exercise OCR (Windows has no tesseract by
default) or the Docker/Spark-cluster machinery.

```bash
pip install -r requirements.txt -r models/pipelines/ner_translate/requirements.txt
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1
```

**Pass criteria:** all 5 `.txt` files produce entities (check
`results/ner_translate_<timestamp>.json`); Hindi/Marathi samples show
`"translated": true`; `sample_text3.txt`'s phone number
(`1234567890`) appears as a `phone number` entity. `sample_scan6.png` will
either be skipped (no OCR locally) or error — that's expected here, not a
failure; it's covered in Tier 3.

### Update — 2026-09-16, verified on Windows (D: drive, 7.85GB RAM host)

`python submit_pipeline_job.py ...` as written does **not** work out of the
box on native Windows — two environment issues, neither a pipeline bug:

1. Driver/worker Python version mismatch: PySpark's `local[N]` spawns
   worker subprocesses via whichever `python` resolves first on `PATH`,
   which was a *different* installed Python (3.10) than the venv's
   interpreter (3.12) driving the job — `PYSPARK_PYTHON` /
   `PYSPARK_DRIVER_PYTHON` must both be pinned to `sys.executable`
   explicitly when more than one Python is installed system-wide.
2. `create_cluster_session`'s defaults (`driver_memory="6g"`,
   `executor_memory="4g"`) assume a real cluster node. On a 7.85GB-RAM
   host they starve the JVM enough that the Python worker subprocess (which
   also has to load GLiNER + NLLB-200 in-process) crashes with
   `Py4JJavaError: ... Python worker exited unexpectedly (crashed)` /
   `EOFException` — no Python traceback, because the process dies before
   it can report one. Lowering to `driver_memory="2g"`, `executor_memory="1g"`
   got further (JVM started, task actually ran) but still isn't reliably
   enough headroom for this pipeline's model footprint on a sub-8GB host.

**Isolated the two from each other** by calling `mod.load()` /
`mod.run(loaded, paths)` directly with no Spark involved at all (`inference/
text_pipeline_engine.py`'s `process_partition` closure, just called inline).
That succeeded cleanly on all samples — real GLiNER + NLLB weights, real
entities out (military units, weapon systems, coordinates, phone numbers,
dates from the EW-domain sample text). This confirms the pipeline *logic*
is correct; the Spark-local-mode crash is this host's RAM, not the code.
`docs/air_gapped_dep.md` §2's own prerequisite table lists **8GB RAM
minimum** — this host's 7.85GB is technically under that bar, which lines
up with what happened.

**Takeaway:** treat Tier 1 as informative-only on a <8GB host — a failure
here doesn't block moving to Tier 2/3, which run the same code inside
Docker containers with their own bounded memory instead of sharing the
host process space with a driver JVM.

## Tier 2 — Docker build verification

```bash
# Base image
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# Wheelhouse (skip if already built — verify with: ls wheels/ner_translate | wc -l, want 50)
bash deploy/scripts/build_ner_translate_wheelhouse.sh

# Pipeline image
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .
```

**Pass criteria — run each of these, all should succeed:**
```bash
docker run --rm ner-translate-worker:latest python --version
# Python 3.11.x

docker run --rm ner-translate-worker:latest python -c "import gliner, transformers, sentencepiece, py3langid; print('deps OK')"
# deps OK — proves the --no-index --find-links wheelhouse install actually worked

docker run --rm ner-translate-worker:latest tesseract --list-langs
# should list: eng, mar, hin, tel, tam, kan, ben, guj, pan, mal, urd

docker run --rm ner-translate-worker:latest python -c "import torch; print(torch.__version__)"
# 2.6.0 — confirms the base image's torch wasn't touched by the pipeline image's install
```

If the wheelhouse install step in the Dockerfile fails here, it's cheaper
to find out now than after `docker compose up`.

### Update — 2026-09-16, verified on Windows (D: drive, 7.85GB RAM host)

**Base image (`multi-model-inference:latest`): built and confirmed —
10.5GB.** Before building, Docker Desktop's WSL2 data disk (51GB, default
location `C:\Users\<user>\AppData\Local\Docker\wsl\disk\docker_data.vhdx`)
was relocated to `D:\DockerDesktopWSL\disk` via `Move-Item` + an NTFS
directory junction (`mklink /J`) at the original C: path — this machine's
C: drive had only 7.8GB free, nowhere near enough for a CUDA+PyTorch+Spark
image. See "Relocating Docker Desktop's disk to another drive" below for
the exact steps; this is the Windows-native equivalent of "build in D
drive."

**Pipeline image (`ner-translate-worker:latest`): build not completed on
this host.** Three attempts, each further than the last:
1. Failed on the `hf-xet` wheelhouse gap above (now fixed).
2. Docker Desktop's daemon died mid-build (`Docker Desktop is unable to
   start`) — this host was down to ~500MB available RAM at the time (two
   unrelated Cassandra containers were also running; stopping them and a
   clean Docker Desktop restart recovered ~2GB, but only while nothing else
   heavy was running).
3. After the crash in (2), buildkit's own metadata store came back
   **read-only** (`write /var/lib/docker/buildkit/containerd-overlayfs/
   metadata_v2.db: read-only file system`) — the WSL2 ext4 disk's safety
   response to the unclean shutdown, not lasting corruption. Recovered with
   `docker desktop stop` → `wsl --shutdown` → restart Docker Desktop (a
   fresh mount cleared it — verified via `docker run --rm hello-world`
   before retrying the real build). A 4th attempt then died again with the
   daemon RPC dropping (`failed to receive status: ... EOF`) partway
   through `pip install`, consistent with the same RAM ceiling rather than
   a new issue.

Root cause across all three: **this host has 7.85GB total RAM**, under
`docs/air_gapped_dep.md` §2's documented 8GB minimum, and that has to cover
Windows + the interactive IDE session driving this work + Docker
Desktop/WSL2 + the build's own `apt`/`pip`/`setup.py` subprocesses at the
same time. The base image (no pipeline-specific deps) built fine; the
pipeline image's build died specifically while installing OCR/NER
dependencies (`odfpy`'s `setup.py` metadata step) on top of everything else
already resident. **Not a pipeline defect** — the base image, the
wheelhouse, and (per Tier 1's direct-call test above) the pipeline logic
itself are all independently confirmed correct; only the *concurrent*
build-while-driving-the-IDE memory budget is the blocker on this
particular machine. Retry on a host with more free headroom (close other
apps first, or a machine that actually meets the 8GB+ minimum) to get a
clean Tier 2 pass, then continue to Tier 3.

#### Relocating Docker Desktop's disk to another drive (Windows)

```powershell
# 1. Stop Docker Desktop and shut down WSL2 so the vhdx isn't locked
docker desktop stop
wsl --shutdown

# 2. Move the data disk folder to the target drive
Move-Item -Path "C:\Users\<user>\AppData\Local\Docker\wsl\disk" `
          -Destination "D:\DockerDesktopWSL\disk" -Force

# 3. Junction the original path to the new location (no admin rights needed,
#    unlike mklink's symlink mode)
cmd /c mklink /J "C:\Users\<user>\AppData\Local\Docker\wsl\disk" "D:\DockerDesktopWSL\disk"

# 4. Start Docker Desktop and verify
Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
docker desktop status
docker run --rm hello-world
```

Everything Docker writes after this — image layers, build cache, the
`docker build` output for this pipeline — now lands on D: transparently;
no Dockerfile, compose file, or build command changes.

## Tier 3 — Cluster + real job submission

```bash
docker compose -f deploy/docker-compose.ner_translate.yml up -d
```

**Pass criteria:**
1. `docker compose -f deploy/docker-compose.ner_translate.yml logs ner-translate-master | grep wheels-hotfix` shows
   `No wheels found in /wheels-hotfix — running with the image's baked-in dependencies.`
   (proves the hotfix mechanism is correctly a no-op by default, not silently active)
2. Spark UI reachable at `http://localhost:8081` (offset port — the shared
   cluster's UI, if also running, stays on `8080`), worker shows as
   `ALIVE` under it
3. Submit the job — this now includes the OCR image. **Use an absolute
   path** (`/app/data/ner_samples`, not `data/ner_samples`): on a real
   distributed cluster each executor task runs from its own per-application
   work directory, not `/app` — a relative path here "succeeds" (no
   exception) while producing nothing but a `[Errno 2] No such file or
   directory` error for every single file. Caught only by reading the job's
   actual per-file results, not by it merely completing without crashing.
   ```bash
   docker exec -it ner-translate-master bash -c \
     "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
   ```
4. Check `_resolve_master_url()` actually picked the dedicated cluster
   automatically (no `--master` flag was passed above) — the job output's
   `partition_details` should show `hostname` values matching the
   container, and it should complete without trying to reach `local[4]`
5. `sample_scan6.png` in the results JSON shows non-empty `entities_unique`
   containing something like `Rakesh Verma` (person), `Indian Army` /
   `5th Mountain Brigade` (organization/military unit), `Leh` (location),
   `9876543210` (phone number) — this is the actual proof tesseract +
   poppler + pytesseract are wired correctly end-to-end, not just installed
6. Nothing in the master/worker logs mentions a Python or package version
   mismatch between them (the failure mode `docs/FRAMEWORK_OVERVIEW.md`
   already documented once)

```bash
docker compose -f deploy/docker-compose.ner_translate.yml down
```

### Update — 2026-09-16

Not attempted on the Windows/D-drive host: Tier 2's `ner-translate-worker`
image never finished building there (see Tier 2's update above), and this
Tier needs it. Once Tier 2 passes on a host that actually clears the 8GB
RAM minimum, Tier 3 is the next thing to run — nothing about it changes.

## Tier 4 — only now, export

Once Tiers 1-3 all pass, follow `docs/air_gapped_dep.md` §3.5's pattern,
applied to this image:

```bash
docker save ner-translate-worker:latest -o ner-translate-worker.tar
gzip ner-translate-worker.tar
```

Transfer `ner-translate-worker.tar.gz` + `deploy/docker-compose.ner_translate.yml`
+ `wheels-hotfix/ner_translate/` (empty — the hotfix mechanism, present in
case it's needed later, not because it's needed now) to the air-gapped
target, same transfer method as the base image. `docs/air_gapped_dep.md`
itself still documents the base/shared image's own export — this doc only
covers the new `ner_translate`-specific image on top of it.
