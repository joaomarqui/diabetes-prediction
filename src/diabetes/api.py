"""
FastAPI Layer - Diabetes Prediction
=====================================
Exposes Kedro pipelines as REST endpoints.

Endpoints:
    POST /train          — Roda data_engineering + modelling em background
    GET  /train/{run_id} — Consulta status do treino
    POST /inference      — Inferência online síncrona (recebe JSON, retorna predições)
    POST /batch-inference — Inferência batch assíncrona (lê do catalog)

Architecture (from slides):
    - Plain def endpoints -> FastAPI dispatches to thread pool
    - Training uses daemon threads (long-running jobs)
    - Thread-safe shared state with threading.Lock
    - KedroSession created per pipeline run (isolated catalog)
    - Bootstrap runs once at startup (lifespan)
"""

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
PROJECT_PATH = Path(__file__).parent.parent.parent.parent.resolve()

# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
_bootstrapped: bool = False
_bootstrap_lock = threading.Lock()
_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Kedro bootstrap (runs ONCE per process)
# ---------------------------------------------------------------------------
def _ensure_bootstrap() -> None:
    """Double-checked locking: bootstrap Kedro exactly once."""
    global _bootstrapped
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            configure_project("diabetes")
            _bootstrapped = True
            logger.info("Kedro bootstrap complete.")


# ---------------------------------------------------------------------------
# Run state helpers
# ---------------------------------------------------------------------------
def _set_run(run_id: str, status: str, **kwargs) -> None:
    with _runs_lock:
        _runs[run_id] = {"run_id": run_id, "status": status, **kwargs}


def _get_run(run_id: str) -> dict:
    with _runs_lock:
        return _runs.get(run_id, {}).copy()


# ---------------------------------------------------------------------------
# Background training job
# ---------------------------------------------------------------------------
def _run_training_background(run_id: str) -> None:
    """Runs data_engineering + modelling pipelines sequentially in a daemon thread."""
    try:
        _ensure_bootstrap()
        _set_run(run_id, "running", pipeline="data_engineering + modelling")
        logger.info("[%s] Training started", run_id)

        for pipeline_name in ["data_engineering", "modelling"]:
            with KedroSession.create(project_path=PROJECT_PATH) as session:
                session.run(pipeline_name=pipeline_name)
            logger.info("[%s] Pipeline '%s' complete", run_id, pipeline_name)

        _set_run(run_id, "completed")
        logger.info("[%s] Training completed successfully", run_id)

    except Exception as exc:
        _set_run(run_id, "failed", error=str(exc))
        logger.exception("[%s] Training failed: %s", run_id, exc)


# ---------------------------------------------------------------------------
# Background batch-inference job
# ---------------------------------------------------------------------------
def _run_batch_inference_background(run_id: str) -> None:
    """Runs inference pipeline from catalog in a daemon thread."""
    try:
        _ensure_bootstrap()
        _set_run(run_id, "running", pipeline="inference")

        with KedroSession.create(project_path=PROJECT_PATH) as session:
            session.run(pipeline_name="inference")

        _set_run(run_id, "completed")
        logger.info("[%s] Batch inference completed", run_id)

    except Exception as exc:
        _set_run(run_id, "failed", error=str(exc))
        logger.exception("[%s] Batch inference failed: %s", run_id, exc)


# ---------------------------------------------------------------------------
# Lifespan: bootstrap on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_bootstrap()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Prediction API",
    description="Kedro pipelines exposed as REST endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models (API contracts)
# ---------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    """
    Body for online inference.
    instances: list of dicts, each dict is one patient record.
    """
    instances: list[dict[str, Any]]


class InferenceResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]


class TrainResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    pipeline: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints — all plain def so FastAPI dispatches to thread pool
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train():
    """
    Trigger async training: data_engineering + modelling pipelines.
    Returns run_id immediately — poll GET /train/{run_id} for status.
    """
    _ensure_bootstrap()
    run_id = str(uuid.uuid4())[:8]
    _set_run(run_id, "pending")

    thread = threading.Thread(
        target=_run_training_background,
        args=(run_id,),
        daemon=True,
    )
    thread.start()

    return TrainResponse(
        run_id=run_id,
        status="pending",
        message=f"Training started. Poll GET /train/{run_id} for status.",
    )


@app.get("/train/{run_id}", response_model=RunStatusResponse)
def get_train_status(run_id: str):
    """Poll training status by run_id."""
    run = _get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found.")
    return RunStatusResponse(**run)


@app.post("/batch-inference", response_model=TrainResponse)
def batch_inference():
    """
    Trigger async batch inference from catalog (reads CSV from data/01_raw).
    Returns run_id — poll GET /train/{run_id} for status.
    """
    _ensure_bootstrap()
    run_id = str(uuid.uuid4())[:8]
    _set_run(run_id, "pending")

    thread = threading.Thread(
        target=_run_batch_inference_background,
        args=(run_id,),
        daemon=True,
    )
    thread.start()

    return TrainResponse(
        run_id=run_id,
        status="pending",
        message=f"Batch inference started. Poll GET /train/{run_id} for status.",
    )


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    """
    Synchronous online inference.
    Accepts JSON with patient data, returns predictions and probabilities.

    Uses MemoryDataset trick to inject HTTP data into Kedro catalog —
    same inference pipeline as batch mode, no code duplication.
    """
    _ensure_bootstrap()

    if not request.instances:
        raise HTTPException(status_code=400, detail="'instances' list is empty.")

    input_df = pd.DataFrame(request.instances)

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        catalog = context._get_catalog()

        # Inject HTTP data into catalog (MemoryDataset override)
        catalog["raw_inference_data"] = MemoryDataset(data=input_df)
        catalog["inference_predictions"] = MemoryDataset()

        # Run the SAME inference pipeline as batch mode
        from kedro.framework.project import pipelines
        inf_pipeline = pipelines["inference"]
        SequentialRunner().run(inf_pipeline, catalog)

        result_df = catalog.load("inference_predictions")

    return InferenceResponse(
        predictions=result_df["prediction"].tolist(),
        probabilities=result_df["probability_diabetes"].tolist(),
    )
