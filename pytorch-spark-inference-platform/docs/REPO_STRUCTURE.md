# Recommended Repo Structure

This documents the structure applied when integrating the `ner_translate`
pipeline (NLLB + GLiNER), and the convention to follow for anything added
after it. Two kinds of "bring your own model" exist now, and they don't share
a folder because they don't share a contract.

```
pytorch-spark-inference-platform/
├── models/
│   ├── model_registry.py          # built-in tensor models (unchanged)
│   ├── plugin_loader.py           # dynamic loader for models/plugins/
│   ├── plugins/                   # SIMPLE plugins: one nn.Module, fixed-shape
│   │   │                          # float32 tensor in, tensor out (see
│   │   │                          # docs/BRING_YOUR_OWN_MODEL.md)
│   │   ├── manifest.json
│   │   ├── example_model.py
│   │   └── weights/                (gitignored, .pt state_dict files)
│   ├── pipelines/                 # COMPLEX plugins: multi-model, file/text
│   │   │                          # in, structured results out (this doc)
│   │   ├── manifest.json          # {name: {module, weights, requirements_file, master_url}}
│   │   └── ner_translate/
│   │       ├── pipeline.py        # load()/run() wrapper — the only file
│   │       │                      # that talks to the framework
│   │       ├── mt_ner_all_formats.py  # the actual pipeline, untouched logic
│   │       └── requirements.txt   # THIS pipeline's deps only — never merged
│   │                              # into the shared requirements.txt (see
│   │                              # docs/MODEL_CONTAINER_ISOLATION.md)
│   └── weights/                   # gitignored — ALL large model binaries,
│       ├── .gitignore             # regardless of which plugin owns them
│       ├── README.md              # how to (re)populate this dir
│       ├── gliner-multi/
│       └── nllb-200-distilled-600M/
├── data/
│   ├── image_generator.py, signal_generator.py   # synthetic data for the
│   │                                              # built-in demo models
│   └── ner_samples/               # small real input files for pipelines
│       └── sample_text*.txt
├── inference/
│   ├── cluster_engine.py          # tensor engine: numpy array -> mapPartitions
│   │                              # -> model(batch_tensor) -> sample count
│   └── text_pipeline_engine.py    # pipeline engine: file paths -> mapPartitions
│                                  # -> load()/run() -> REAL results merged back
├── submit_job.py                  # CLI for models/plugins/ (tensor models)
├── submit_pipeline_job.py         # CLI for models/pipelines/ (pipeline plugins)
├── deploy/
│   ├── Dockerfile, docker-compose*.yml   # Linux/Docker cluster (unchanged)
│   └── aws-cdk/
│       └── spark_cluster/
│           ├── spark_cluster_stack.py         # Linux EC2 cluster
│           ├── gpu_benchmark_stack.py         # single GPU benchmark instance
│           └── windows_spark_cluster_stack.py # native Windows EC2 cluster
└── docs/                          # all *.md documentation lives here, never
                                    # mixed with model code or sample data
```

## Why two plugin folders instead of one

`models/plugins/` and `models/pipelines/` exist because they solve genuinely
different problems, and forcing one contract onto both would make both worse:

| | `models/plugins/` | `models/pipelines/` |
|---|---|---|
| Model shape | one `nn.Module` | any number of models/objects |
| Load contract | `ModelClass()` + `load_state_dict(weights.pt)` | `load()` returns whatever the pipeline needs (HF `from_pretrained()`, tokenizers, language-id, ...) |
| Input | fixed-shape `float32` numpy array | file paths / raw text, any format |
| Output today | **discarded** — only a sample count (see `docs/CONCURRENT_JOBS_AND_COMPLETING_THE_FRAMEWORK.md` §4) | **kept** — real structured results merged back to the driver |
| Execution engine | `inference/cluster_engine.py` | `inference/text_pipeline_engine.py` |
| CLI | `submit_job.py` | `submit_pipeline_job.py` |

If a future model fits the simple tensor shape (classifier, detector,
regressor on a fixed-size input), it belongs in `models/plugins/` and gets
the existing, lighter-weight contract. If it needs its own loading logic,
multiple models, or text/file input — like translation+NER — it belongs in
`models/pipelines/`, following exactly the `ner_translate/` layout: the
original implementation untouched in its own file, plus a thin `pipeline.py`
that's the only thing the framework actually imports.

## Rules that keep this from rotting

1. **No model weights in git, ever, from either folder.** They all live
   under `models/weights/`, which is fully gitignored except its own
   `README.md`. A plugin's manifest entry points at a path under here; the
   path is never itself committed.
2. **`docs/` is documentation only.** Sample input data goes in `data/`
   (`data/ner_samples/`, not `docs/sample_text*.txt` — the original NER
   upload had this backwards, which is why it got moved).
3. **One pipeline = one subfolder under `models/pipelines/`.** Even if it's
   just one file today, this leaves room for a pipeline to grow its own
   helper modules (as `ner_translate/` already has two: `pipeline.py` +
   `mt_ner_all_formats.py`) without spilling into its siblings.
4. **The original implementation stays untouched wherever possible.**
   `mt_ner_all_formats.py`'s only edits were making its two path constants
   robust to being imported from a different working directory (relative to
   `__file__`, not `os.getcwd()`) — none of its extraction/translation/NER
   logic changed. `pipeline.py` is the adapter layer; it should always be the
   thing that changes to fit the framework, never the vendored implementation.
