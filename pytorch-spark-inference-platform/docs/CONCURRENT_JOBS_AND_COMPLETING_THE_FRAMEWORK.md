# Concurrent Job Submission & Completing the BYOM Framework

Two related questions this doc answers:
1. How do multiple Spark jobs run *at the same time* against one cluster (master
   + workers), with this framework, today?
2. What's still missing before this is a genuinely complete "submit any model,
   get real results back, from multiple people/jobs at once" framework — not
   just a benchmarking harness?

---

## 1. How concurrent jobs work in Spark Standalone (the mechanics)

Every `submit_job.py` run creates its own **Spark Application** — a fresh
`SparkSession` (`create_cluster_session()`, `inference/cluster_engine.py:43-86`)
that registers with the master (`spark://<master>:7077`) and gets torn down
(`spark.stop()`) when the run finishes (`submit_job.py:60-63`).

Spark Standalone mode **natively supports multiple Applications registered
with the same master at once** — nothing needs to be built for that part.
What determines whether they actually run *concurrently* (as opposed to one
starving the other) is how cluster cores get divided up:

- Standalone's default scheduler is **FIFO across Applications** for core
  allocation, and — critically — **an Application with no `spark.cores.max`
  set claims every free core on the cluster** until it finishes. Our
  `create_cluster_session()` sets `spark.executor.cores` (cores *per
  executor process*) but **never sets `spark.cores.max`** (the cap on total
  cores an Application may hold across the whole cluster) — confirmed by
  grepping the codebase, there is no `cores.max`/`cores_max` anywhere in
  `inference/cluster_engine.py` or `submit_job.py`.
- Net effect **today**: if you launch two `submit_job.py` runs back to back,
  the first one to register with the master takes all available worker
  cores, and the second sits registered-but-starved (0 active tasks) until
  the first releases cores. They're not blocked from *submitting*
  concurrently — they just won't get **simultaneous execution** unless the
  cluster happens to have more free cores than the first job asked for.

This is the one fact that matters most: **concurrent submission already
works; concurrent *execution* needs a cores cap per job, which doesn't exist
yet.**

---

## 2. Triggering concurrent jobs today (no code changes)

Works right now, with the caveat above in mind:

```powershell
# From two separate terminals / SSM sessions / background processes:
python submit_job.py --model example_mlp --mode hybrid --master spark://<master-ip>:7077 &
python submit_job.py --model ner_plugin  --mode hybrid --master spark://<master-ip>:7077 &
```

To actually get them running *in parallel* rather than queued behind each
other, check free cores first and keep each job's footprint under that:

```powershell
# On the master, check current free cores across the cluster:
Invoke-RestMethod http://localhost:8080/json/ | Select-Object -ExpandProperty workers |
    Select-Object host, cores, coresfree
```

If a worker shows `coresfree: 2`, a concurrent job's `--partitions`/executor
footprint needs to fit inside that — but again, there's currently no CLI flag
to cap a job's total core claim, so in practice today the *first* job you
launch should be the one you expect to dominate, and any second job you
launch concurrently will only get scheduled onto whatever that first job
didn't already grab (which, per §1, is often "nothing" until it finishes).

---

## 3. What's missing to make concurrent jobs actually fair

### 3.1 `--max-cores` on `submit_job.py` → `spark.cores.max`

**File to edit:** `submit_job.py` (add a `--max-cores` argument) and
`inference/cluster_engine.py`'s `create_cluster_session()` (add a
`cores_max: Optional[int] = None` parameter that, when set, adds
`.config("spark.cores.max", str(cores_max))` to the builder). This is the
single highest-leverage, smallest change for real concurrency — without it,
"concurrent jobs" degrades to "concurrent submissions, sequential execution."

### 3.2 A capacity-aware job queue (new file)

**New file: `scripts/job_queue.py`** — a small dispatcher that:
- accepts job specs (model name, input, mode, desired cores) from a simple
  queue (a watched directory of `.json` job files, or a Python list for a
  first pass),
- polls `http://<master>:8080/json/` for `coresfree` before launching each
  one,
- launches accepted jobs as `submit_job.py` subprocesses with `--max-cores`
  set to fit the currently-free capacity, queuing the rest until room frees
  up.

Without this, "concurrent jobs" means "whoever launches `submit_job.py` by
hand, whenever they feel like it" — fine for a couple of engineers testing,
not fine for anything with multiple simultaneous users.

### 3.3 Pre-flight plugin validation (new function, existing file)

**File to edit:** `models/plugin_loader.py` — add a `validate_plugin(name,
registry)` that instantiates the model on CPU, runs one dummy batch through
`forward()`, and checks the output shape/dtype *before* a Spark job is ever
submitted. Today, a broken plugin (wrong `forward()` signature, mismatched
`input_shape`) only fails once it's already running distributed across
executors, surfacing as a Py4J stack trace that's much harder to debug than a
local pre-flight error would be.

---

## 4. The bigger gap: results are currently thrown away

This is the most important finding in this doc, and it directly affects the
NER model test you mentioned wanting to run next.

Look at `process_partition()` inside `run_cluster_inference()`
(`inference/cluster_engine.py`, around the batch loop):

```python
with torch.no_grad():
    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        batch_np = chunk[start:end]
        batch_tensor = torch.from_numpy(batch_np).float().to(device)
        _ = model(batch_tensor)          # <-- output is discarded
        num_outputs += (end - start)     # <-- only a count is kept
```

**The model's actual output tensor is thrown away — `run_cluster_inference()`
only returns a count of how many samples were processed, never the
predictions themselves.** This makes complete sense for what the platform
was originally built for (`benchmark/cluster_benchmark.py` — pure throughput
benchmarking: "how many samples/sec can this cluster process"), but it means
**today, running any model through this framework — including the demo
plugin we already tested — gives you a sample count, never the model's
actual predictions.**

For a throughput test, that's fine. For an actual NER run where the point is
*getting entity predictions back*, this is a hard blocker, not a nice-to-have.

### 4.1 What needs to change

**File to edit: `inference/cluster_engine.py`** — `process_partition()` needs
to actually keep `output = model(batch_tensor)` and do something with it.
Returning raw tensors to the driver via `.collect()` doesn't scale (driver
memory blows up on real datasets), so the correct pattern is **each executor
writes its own partition's predictions directly to disk**, not through the
driver:

**New file: `inference/output_writer.py`** — a small helper,
`write_partition_predictions(output_tensor, partition_idx, job_id, output_dir)`,
called from inside `process_partition()` after each batch, writing to e.g.
`results/predictions/<job_id>/partition_<idx>.npy` (or `.parquet` if you want
something more queryable). `run_cluster_inference()`'s return value then adds
an `output_dir` field pointing at where the driver can find all the
partition files afterward — no change to the `.collect()` aggregation path,
which stays as the lightweight per-partition timing/count summary it already
is.

### 4.2 A `postprocess()` convention for plugins (new optional hook)

Raw model output usually isn't the end product — NER needs logits decoded
into entity spans/labels, classification needs an argmax, detection needs
NMS. Right now the plugin contract (`docs/BRING_YOUR_OWN_MODEL.md`) only
defines `forward()`. Extend the convention: a plugin module may *optionally*
define a top-level `postprocess(output_tensor) -> Any` function; if present,
`plugin_loader.py`'s class map lookup also returns the postprocess function
alongside the class, and `process_partition()` calls it after `forward()`
before writing to disk (§4.1). If absent, raw tensor output is written as-is
(today's implicit behavior, made explicit instead of discarded).

---

## 5. The other gap that blocks the NER test specifically: input isn't text

`submit_job.py --input` only accepts a `.npy` file, shape `(N, *input_shape)`,
dtype `float32` (`submit_job.py:44-48`). **A NER model takes raw text, not a
fixed-shape float array** — this framework has no tokenization/text-input
path today at all.

### 5.1 What needs to be added

**New file: `data/generic_loader.py`** (this was sketched as a stub in the
original BYOM design but never actually built — worth building now that a
real text model is the next test case). It should:
- read a plain `.txt` (one document per line) or `.jsonl` file of raw
  strings,
- partition the raw strings across Spark the same way `run_cluster_inference()`
  already partitions numpy arrays (index-slicing into `num_partitions`
  chunks) — this part is easy to mirror.

**Extend the plugin manifest contract** (`models/plugins/manifest.json`):
add an optional `"tokenizer_module"` field pointing at a function (in the
plugin's own file, or a shared one) that turns a batch of raw strings into
the tensor `forward()` expects. This function needs to run **on the
executor**, not the driver — tokenizer vocab files/config must ship the same
way model weights do (either baked into `project.zip`, or, if using
`transformers`, downloaded once and cached — which reintroduces the
airgapped/no-runtime-download constraint from `docs/BRING_YOUR_OWN_MODEL.md`
§4, so for an airgapped target the tokenizer files need to be vendored
locally too, not `from_pretrained()`'d at runtime).

**Extend `models/plugins/manifest.json`** with an optional
`"extra_requirements"` list (e.g. `["transformers==4.44.0"]`) — a NER model
built on HuggingFace `transformers` needs a package not in the base
`requirements.txt`. `plugin_loader.py`'s `register_plugins()` should pip-install
these automatically (or at minimum print what needs manual installing) rather
than the user finding out via an `ImportError` on a remote executor.

---

## 6. Summary: files to add/edit for a "complete" framework

| # | Priority | File | Change |
|---|---|---|---|
| 1 | High | `submit_job.py`, `inference/cluster_engine.py` | Add `--max-cores` → `spark.cores.max`, so concurrent jobs can actually co-execute instead of starving each other |
| 2 | High | `inference/cluster_engine.py` (edit) + `inference/output_writer.py` (new) | Stop discarding model outputs — write per-partition predictions to disk instead of only counting samples |
| 3 | High (blocks NER specifically) | `data/generic_loader.py` (new) | Text/`.jsonl` input support — today only fixed-shape `.npy` float32 arrays are accepted |
| 4 | Medium | `models/plugins/manifest.json` (schema) + `models/plugin_loader.py` | Optional `tokenizer_module` + `postprocess` hooks per plugin, `extra_requirements` list for plugin-specific pip packages |
| 5 | Medium | `models/plugin_loader.py` | `validate_plugin()` pre-flight check (catch broken plugins locally, not on a remote executor) |
| 6 | Lower (nice-to-have) | `scripts/job_queue.py` (new) | Capacity-aware dispatcher so concurrent submissions from multiple users don't have to be manually core-budgeted by hand |

Items 2 and 3 are the ones that actually block a real NER run — worth doing
those first. Items 1 and 6 are what turn "you can technically launch two
`submit_job.py` at once" into "concurrent jobs behave sanely." Items 4 and 5
are what make onboarding a new model type (like NER) safe and less
trial-and-error.
