"""
predict_batch_udf — the batch-prediction Pandas UDF for the tensor-plugin
execution path (models/plugins/), wired into a real Spark job by
cluster_engine_udf.py. See docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md §2 for
the design rationale (why this exists alongside cluster_engine.py's
mapPartitions path, not instead of it).

Not for models/pipelines/ (ner_translate etc.) — file-based, nested-output
pipelines don't fit Arrow's columnar schema requirement; those stay on
text_pipeline_engine.py's mapPartitions path.

Scope note: the UDF below returns ArrayType(FloatType()) — a flat float
vector per row, which covers classifiers/regressors like ResNet18/
MobileNetV3/EfficientNet-B0 (models/image_models.py) but NOT a
variable-structure output like YOLO's per-image detection list
(models/yolo_model.py). Extending this to structured outputs would need a
richer return schema (StructType with nested ArrayType) — not attempted
here since it's a different, larger design question than what this file
answers.

Deliberately NOT using `from __future__ import annotations` here (unlike
cluster_engine.py) — PySpark's pandas_udf decorator inspects the actual
`Iterator[pd.Series]` type object via `inspect.signature()` to pick the
right calling convention; postponed evaluation turns that into a bare
string and pandas_udf fails with `PySparkNotImplementedError:
[UNSUPPORTED_SIGNATURE]`. Hit this for real while testing — not a
theoretical caveat.
"""

import io
import logging
from typing import Iterator, Type

logger = logging.getLogger("predict_batch_udf")


def predict_batch_udf(model_bytes: bytes, model_class: "Type",
                       sample_shape: tuple, device: str = "cuda",
                       inference_batch_size: int = 256):
    """
    Scalar-iterator Pandas UDF factory: the model loads ONCE per partition
    (same optimization cluster_engine.py's mapPartitions docstring already
    calls out — "loads models ONCE, processes all items" — this is that
    same idea expressed as a Pandas UDF instead of a raw mapPartitions
    closure), then every Arrow batch Spark hands this partition streams
    through the already-loaded model.

    Args:
        model_bytes: serialized state_dict, from the same
            sc.broadcast(_serialize_model(...)) pattern cluster_engine.py
            already uses — no new serialization mechanism introduced.
        model_class: the nn.Module subclass to instantiate before loading
            model_bytes into it — generic across any models/plugins/ entry,
            not hardcoded to one model.
        sample_shape: the model's expected per-sample shape, e.g.
            (3, 224, 224) for ResNet18. Rows travel through Spark as FLAT
            1-D float arrays (see cluster_engine_udf.py's row-building
            code) and get reshaped back to this on the way into the model
            — flattening on the way in is what keeps Spark's schema
            trivial (see the caller-side note below); a naive first version
            of this that sent nested (3,224,224) lists straight through
            `spark.createDataFrame` without an explicit schema hung for
            minutes on schema inference over deeply nested Python lists —
            caught by real local testing, not a hypothetical.
        device: "cuda" or "cpu" — resolved once per partition, same
            fallback logic cluster_engine.py's process_partition already
            has (falls back to "cpu" if torch.cuda.is_available() is False).
        inference_batch_size: the MODEL's preferred batch size. Distinct
            from Arrow's own batch size (spark.sql.execution.arrow.
            maxRecordsPerBatch, default 10,000, decided by Spark, not this
            function) — an Arrow batch gets re-chunked to this size before
            each model() call, so a 10k-row Arrow batch can't blow VRAM on
            a large model just because Spark happened to hand it all at
            once. This is the batching-size reconciliation
            docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md §2 flagged as an open
            decision — resolved here by always re-chunking, never assuming
            the two sizes match.

    Returns: a pandas_udf usable as
        df.select(predict_batch_udf(...)(df.input_col))
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType
    import pandas as pd
    import numpy as np
    import torch

    expected_flat_len = 1
    for d in sample_shape:
        expected_flat_len *= d

    @pandas_udf(ArrayType(FloatType()))
    def _predict(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        model = model_class()
        model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location="cpu"))
        resolved_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        model = model.to(resolved_device).eval()

        for arrow_batch in batches:
            rows = list(arrow_batch)
            results: list = [None] * len(rows)
            valid_idx: list[int] = []
            valid_arrays: list = []

            for i, row in enumerate(rows):
                arr = _row_to_array(row, expected_flat_len, sample_shape)
                if arr is None:
                    # Bad-input decision (docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md
                    # §2): skip-and-log, not fail-the-partition. Matches
                    # cluster_engine.py's mapPartitions path, which also
                    # doesn't isolate one bad row from the rest of a batch
                    # today — this isn't a regression, just made explicit.
                    logger.warning("predict_batch_udf: skipping malformed row %d", i)
                    continue
                valid_idx.append(i)
                valid_arrays.append(arr)

            if valid_arrays:
                stacked = np.stack(valid_arrays)
                with torch.no_grad():
                    for start in range(0, len(stacked), inference_batch_size):
                        end = min(start + inference_batch_size, len(stacked))
                        chunk = torch.from_numpy(stacked[start:end]).float().to(resolved_device)
                        out = model(chunk).cpu().numpy()
                        for j, res in zip(valid_idx[start:end], out):
                            results[j] = res.tolist()

            yield pd.Series(results)

    return _predict


def _row_to_array(row, expected_flat_len, sample_shape):
    """Convert one Arrow/pandas row (a flat list/array of expected_flat_len
    floats) back to a numpy float32 array of sample_shape. Returns None
    (never raises) for malformed input — the caller treats None as
    skip-and-log."""
    import numpy as np
    try:
        arr = np.asarray(row, dtype=np.float32)
        if arr.size != expected_flat_len:
            return None
        return arr.reshape(sample_shape)
    except (TypeError, ValueError):
        return None
