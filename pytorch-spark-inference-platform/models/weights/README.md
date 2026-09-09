# Model Weights (not committed to git)

This directory holds local model artifacts referenced by `models/plugins/manifest.json`
(`weights_path`, single `.pt` files) and `models/pipelines/manifest.json`
(`weights`, whole HuggingFace-style model directories). Everything here is
gitignored except this file — weights are large binaries and must never go
into source control, matching how the rest of the platform handles them
(the Linux Docker image bakes weights in at build time; see
`docs/BRING_YOUR_OWN_MODEL.md` §4 and §7 for the airgapped-transfer story).

## Current contents

- `gliner-multi/` — GLiNER zero-shot NER model (`urchade/gliner_multi-v2.1`),
  used by `models/pipelines/ner_translate/`.
- `nllb-200-distilled-600M/` — Facebook NLLB translation model, used by the
  same pipeline.

## Populating this directory on a new machine

Since this directory isn't in git, copy these folders over separately
(matching the project's general airgapped-transfer pattern: prepare on an
internet-connected machine, transfer the folder), or regenerate them:

```bash
pip install gliner transformers torch sentencepiece py3langid pyarrow

python -c "from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_multi-v2.1').save_pretrained('models/weights/gliner-multi')"

python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
m='facebook/nllb-200-distilled-600M'; \
AutoTokenizer.from_pretrained(m).save_pretrained('models/weights/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained(m).save_pretrained('models/weights/nllb-200-distilled-600M')"
```
