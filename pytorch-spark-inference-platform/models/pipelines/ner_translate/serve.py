"""
ner_translate model server — the "kitchen" in the waiter/kitchen split
(see docs/MODEL_CONTAINER_ISOLATION.md Option B). Wraps this pipeline's
existing load()/run(loaded, paths) contract (pipeline.py) behind one HTTP
endpoint, so Spark workers never need torch/CUDA/gliner/transformers in
their own Python environment — they just POST a list of file paths (already
on the same mounted volume this server reads from) and get back the same
{filename: result} shape pipeline.run() always returned.

Model loads exactly once, at process startup, regardless of how many Spark
executors are calling in — the whole point of moving inference behind a
service instead of loading it once per executor.

Run directly (inside the server container):
    uvicorn models.pipelines.ner_translate.serve:app --host 0.0.0.0 --port 8000
"""
import logging
import time

from fastapi import FastAPI
from pydantic import BaseModel

from . import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ner_translate.serve")

app = FastAPI(title="ner_translate model server")
_state = {"loaded": None}


class PredictRequest(BaseModel):
    paths: list[str]
    labels: list[str] | None = None


@app.on_event("startup")
def _load_once() -> None:
    start = time.time()
    logger.info("Loading GLiNER + NLLB + language-id...")
    _state["loaded"] = pipeline.load()
    logger.info("Models loaded in %.1fs", time.time() - start)


@app.get("/health")
def health():
    return {"status": "ok" if _state["loaded"] is not None else "loading"}


@app.post("/predict")
def predict(req: PredictRequest):
    if _state["loaded"] is None:
        return {"error": "models still loading, retry shortly"}
    start = time.time()
    results = pipeline.run(_state["loaded"], req.paths, labels=req.labels)
    logger.info("Processed %d path(s) in %.2fs", len(req.paths), time.time() - start)
    return results
