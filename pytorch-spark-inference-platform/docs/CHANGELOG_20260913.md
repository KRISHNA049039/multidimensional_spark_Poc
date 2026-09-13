# Changelog — September 13, 2026

Fixing a Docker image / NER pipeline version mismatch that broke
`ner_translate` (GLiNER + NLLB) both locally and in the Spark cluster
images, plus a `deploy: nvidia/cuda` base-image decision.

---

## Root Cause Summary

| # | Bug | Where | Symptom |
|---|-----|-------|---------|
| 1 | `torchvision==0.17.0` pin silently downgraded torch | `requirements.txt` | `pip install -r requirements.txt` after `torch==2.6.0 torchvision==0.21.0` install replaced both with `torch==2.2.0` (torchvision 0.17.0 hard-requires `torch==2.2.0`) — undoes the CUDA 12.6 / sm_120 (RTX 5060) fix from `CHANGELOG_20260720.md` |
| 2 | `ner_translate` deps never installed | `models/pipelines/manifest.json` `extra_requirements` vs `requirements.txt` | `ModuleNotFoundError: No module named 'gliner'` on every executor — the manifest field was declarative only, nothing consumed it |
| 3 | `numpy==1.26.4` incompatible with current `py3langid` | `requirements.txt` | `py3langid>=0.3.0` requires `numpy>=2.0.0`; old pin forced `py3langid` back to `0.2.2`, and `pyarrow==16.1.0` has no wheel for Python ≥3.13 |
| 4 | `Dockerfile.worker` couldn't build on either target | `Dockerfile.worker` (orphaned — no compose file references it) | CPU stage: `torch==2.5.1` has no `cp314` wheel on `python:3.14-slim` base. GPU stage: pinned nightly `2.9.0.dev20250115+cu128` does not exist on the nightly index |
| 5 | No OCR/document-extraction system deps | `deploy/Dockerfile` | PDF/image/Office-doc inputs to `mt_ner_all_formats.py::extract_text()` silently produced empty text (missing `tesseract-ocr`, `poppler-utils`, and their Python bindings) |
| 6 | `models/weights/` empty | n/a (data, not code) | `GLiNER.from_pretrained(..., local_files_only=True)` fails — needs manual population, documented but not automated |

All version pins below were checked against PyPI/Apache/Ubuntu-archive metadata and dry-run resolved together before committing (not guessed).

---

## Modified Files

### `requirements.txt`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| `torchvision` pin | `torchvision==0.17.0` | **removed** | Was force-downgrading torch 2.6.0 → 2.2.0; torchvision is installed by `deploy/Dockerfile` directly from the cu126 wheel index instead |
| `numpy` | `1.26.4` | `2.1.3` | Required by `py3langid>=0.3.0`; old pin also has no wheel for the Python versions in play |
| `pandas` | `2.2.2` | `2.2.3` | Keep in step with numpy 2 |
| `pyarrow` | `16.1.0` | `18.1.0` | `16.1.0` has no `cp313`/`cp314` wheel |
| `ultralytics` | `8.2.0` | `8.3.40` | numpy-2 compatible |
| `matplotlib` | `3.9.0` | `3.9.2` | Compatibility bump alongside numpy 2 |
| `gliner`, `transformers`, `sentencepiece`, `py3langid` | *(not present — only listed in manifest `extra_requirements`, never installed)* | `gliner==0.2.13`, `transformers==4.57.6`, `sentencepiece==0.2.0`, `py3langid==0.4.0` | `ner_translate` pipeline deps now actually installed; `transformers` capped `<5.0` because `mt_ner_all_formats.py`'s NLLB tokenizer/`generate()` calls target the 4.x API — 5.x is a breaking rewrite |
| Document-extraction libs | *(not present)* | `pypdf==5.1.0`, `pdf2image==1.17.0`, `pillow==11.0.0`, `pytesseract==0.3.13`, `python-docx==1.1.2`, `openpyxl==3.1.5`, `python-pptx==1.0.2`, `odfpy==1.4.1`, `ebooklib==0.18`, `beautifulsoup4==4.12.3`, `striprtf==0.0.28`, `xlrd==2.0.1` | Cover every format branch in `extract_text()` (PDF/image/DOCX/PPTX/XLSX/XLS/ODS/ODT/RTF/EPUB/HTML) |

**Verification:** dry-run resolved the full set for `python3.11`/`manylinux2014_x86_64` with `torch==2.6.0` pre-installed as a constraint — torch stayed at `2.6.0` (no downgrade), all 73 packages resolved with no conflicts.

---

### `deploy/Dockerfile`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| apt packages | `software-properties-common wget gnupg curl`, `python3.11* openjdk-17-jre-headless procps` | added `tesseract-ocr` + language packs `tesseract-ocr-mar tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-kan tesseract-ocr-ben tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-mal tesseract-ocr-urd`, and `poppler-utils` | Matches `OCR_LANG = "eng+mar+hin+tel+tam+kan+ban+guj+pan+mal+urd"` in `mt_ner_all_formats.py:86` exactly; `poppler-utils` backs `pdf2image` for scanned-PDF OCR. All package names verified against the Ubuntu 22.04 (jammy) archive |
| torch/torchvision install order | unchanged (`torch==2.6.0 torchvision==0.21.0 --index-url .../cu126` then `pip install -r requirements.txt`) | unchanged, but now safe | With the `torchvision` pin removed from `requirements.txt` (see above), the second `pip install` no longer has anything that conflicts with the first — no code change needed here beyond the requirements.txt fix |

---

### `Dockerfile.worker` *(currently orphaned — not built by any compose file; kept consistent anyway)*

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| CPU stage torch/torchvision | `torch==2.5.1 torchvision==0.20.1 --index-url .../whl/cpu` | `torch==2.9.1 torchvision==0.24.1 --index-url .../whl/cpu` | `2.5.1` has no `cp314` wheel (base image is `python:3.14-slim`) — install was failing outright. `2.9.1` is the first stable torch release published for `cp314`; `0.24.1` is the torchvision release that pins `torch==2.9.1` exactly |
| GPU stage torch/torchvision | `torch==2.9.0.dev20250115+cu128 torchvision==0.24.0.dev20250115+cu128 --index-url .../nightly/cu128` | `torch==2.9.1 torchvision==0.24.1 --index-url .../whl/cu128` | The pinned nightly build does not exist on the nightly index — never worked. Replaced with the same stable pair as the CPU stage (real, released, `cp314` wheels, `sm_120`/Blackwell support — first landed in the 2.7 series) |

Both stages now share one real Python (3.14) / torch (2.9.1) pairing, satisfying the file's own "MUST match Master and CPU worker exactly" comment.

---

## Design Decision: `nvidia/cuda:*-runtime` vs `*-devel`

**Kept `runtime`, did not switch to `devel`.**

- torch/torchvision are installed as pip wheels in every Dockerfile here, and those wheels bundle their own CUDA/cuDNN shared libraries (as `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc. pip deps) — the base image's own CUDA toolkit is not exercised at runtime.
- `devel` images add `nvcc` + headers + static libs (~3-4GB extra), which only pays off when something needs to *compile* CUDA code at build or import time (custom ops, `flash-attn`/`apex`/`xformers` built from source, DeepSpeed JIT kernels). Nothing in this repo does that.
- Given the project's whole deployment story is `docker save` / `docker load` tarballs for airgapped machines (`AIRGAPPED_DEPLOYMENT.md`), the smaller `runtime` image is strictly better here.

---

## Not Fixed (deliberately out of scope / needs manual action)

| Item | Why left alone |
|------|-----------------|
| `models/weights/` still empty | Gitignored by design (`models/weights/README.md`) — populate via the `GLiNER.from_pretrained(...).save_pretrained(...)` / NLLB snippets in that README, or copy the folders from an internet-connected machine |
| `deploy/docker-compose.yml` (dev mode) sets `runtime: nvidia` unconditionally | Will refuse to start without an NVIDIA GPU + `nvidia-container-toolkit`; intentional for its purpose (dev-mode GPU box), not something to "fix" for CPU-only testing |
| `Dockerfile.worker` is not referenced by any compose file | Appears to be a standalone/future artifact; fixed its version pins for correctness but did not wire it into the compose stack (out of scope of the reported mismatch) |

---

## How to Verify Without an NVIDIA GPU

The pipeline auto-detects and falls back to CPU (`mt_ner_all_formats.py:441`,
`device = "cuda" if torch.cuda.is_available() else "cpu"`), so most of this
is testable on any machine:

1. **No Docker, fastest feedback:**
   ```bash
   pip install -r requirements.txt
   # populate models/weights/ per models/weights/README.md, then:
   python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1
   ```
2. **Multi-container Spark cluster, still CPU-only** (its GPU-worker service
   already has `CUDA_VISIBLE_DEVICES=` blanked and no `runtime: nvidia`):
   ```bash
   docker compose -f deploy/docker-compose.cluster.yml build
   docker compose -f deploy/docker-compose.cluster.yml up
   docker exec -it spark-master bash -c "python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 2 --master spark://spark-master:7077"
   ```
3. **`Dockerfile.worker` GPU stage:** `docker build --target gpu -f Dockerfile.worker .` builds fine without a GPU (building never requires one) and proves the pip resolution is real; `torch.cuda.is_available()` will just print `False` without driver passthrough. Actual `sm_120` execution needs real Blackwell hardware — the repo's `deploy/aws-cdk/` + `run_gpu_cdk.ps1` already provision a GPU EC2 instance for that.

---

## File Tree After Changes

```
pytorch-spark-inference-platform/
├── requirements.txt                 [MODIFIED] torchvision pin removed, numpy/pyarrow/pandas/
│                                                ultralytics bumped, ner_translate + doc-extraction
│                                                deps added
├── deploy/
│   └── Dockerfile                   [MODIFIED] tesseract-ocr + language packs + poppler-utils added
├── Dockerfile.worker                [MODIFIED] both stages repinned to torch==2.9.1/torchvision==0.24.1
│                                                (real, stable, cp314 + sm_120 support)
└── docs/
    └── CHANGELOG_20260913.md        [NEW] This file
```
