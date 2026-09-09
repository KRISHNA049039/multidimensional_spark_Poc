"""
Thin BYOM "pipeline" wrapper around mt_ner_all_formats.py.

Exposes the load()/run(loaded, paths) contract that
inference/text_pipeline_engine.py expects, without modifying the underlying
pipeline's own extraction/translation/NER logic at all.
"""
from . import mt_ner_all_formats as _impl


def load():
    """Load GLiNER + NLLB + language-id once per executor."""
    return _impl.load_models()


def run(loaded, paths, labels=None):
    """Run the full extract -> detect -> translate -> NER pipeline over `paths`.

    Returns {filename: {language, translated, entities_unique, ...}} —
    the same shape mt_ner_all_formats.py itself writes to ner_output.json.
    """
    return _impl.process_paths_batched(loaded, paths, labels or _impl.DEFAULT_LABELS)
