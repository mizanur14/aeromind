"""AeroMind REST API.

Run: ``uv run aeromind serve`` (or ``uvicorn aeromind.api.app:app``). Configuration via
environment variables (see ``.env.example``):

- ``AEROMIND_REGISTRY``  registry JSON (default ``reports/registry/model_registry.json``)
- ``AEROMIND_ROOT``      directory that registry bundle paths are relative to (default ``.``)

Every response that contains a prediction includes a ``provenance`` object. Predictions are
computed by the registered bundle; the API never fabricates values.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from aeromind import __version__
from aeromind.data.cmapss import SENSOR_COLUMNS, SETTING_COLUMNS
from aeromind.serving.service import InvalidInput, ModelService, ServiceError

API_VERSION = "1"
MAX_BATCH_ENGINES = 500
ModelId = Annotated[str, Path(max_length=80, description="Registry model id, e.g. fd001-s2-xgboost")]

# ----------------------------------------------------------------------------- schemas
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)

CycleRecord = create_model(  # one row of the raw C-MAPSS file (engine_id given separately)
    "CycleRecord",
    __config__=_STRICT,
    cycle=(int, Field(ge=1, description="Operating cycle number (1-based, strictly increasing)")),
    **{c: (float, Field(strict=True, description="Operating setting")) for c in SETTING_COLUMNS},
    **{c: (float, Field(strict=True, description="Sensor reading (C-MAPSS units)")) for c in SENSOR_COLUMNS},
)


class EngineInput(BaseModel):
    """One engine's observed history."""

    model_config = _STRICT
    engine_id: str | int = Field(default="engine", description="Caller's label, echoed back")

    @field_validator("engine_id")
    @classmethod
    def _short_label(cls, v: str | int) -> str | int:
        if isinstance(v, str) and len(v) > 64:
            raise ValueError("engine_id strings are limited to 64 characters")
        return v
    cycles: list[CycleRecord] = Field(min_length=1, max_length=5000)  # type: ignore[valid-type]


class PredictOptions(BaseModel):
    """Common prediction options."""

    model_config = _STRICT
    interval_levels: list[float] = Field(default_factory=list, max_length=4, description="e.g. [0.8, 0.9]")
    include_trajectory: bool = Field(default=False, description="Also return the prediction at every cycle")


class PredictRequest(PredictOptions):
    """Single-engine prediction request."""

    engine: EngineInput


class BatchPredictRequest(PredictOptions):
    """Multi-engine prediction request."""

    engines: list[EngineInput] = Field(min_length=1, max_length=MAX_BATCH_ENGINES)


class Interval(BaseModel):
    level: float
    lower: float
    upper: float
    half_width: float
    method: str
    n_calibration_engines: int
    validated_coverage: dict[str, Any] | None
    guarantee: str


class Prediction(BaseModel):
    model_id: str
    engine_id: str | int
    n_cycles: int
    last_cycle: int
    predicted_rul: float
    target: str
    intervals: list[Interval]
    inference_ms: float
    warnings: list[str]
    provenance: dict[str, Any]
    trajectory: dict[str, list[float]] | None = None


class BatchPrediction(BaseModel):
    model_id: str
    n_engines: int
    total_inference_ms: float
    predictions: list[Prediction]


def _frame(engine: EngineInput) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in engine.cycles])


# ----------------------------------------------------------------------------- app
def create_app(service_factory: Callable[[], ModelService] | None = None) -> FastAPI:
    """Build the app. ``service_factory`` is injectable for tests."""

    def default_factory() -> ModelService:
        return ModelService(
            os.environ.get("AEROMIND_REGISTRY", "reports/registry/model_registry.json"),
            os.environ.get("AEROMIND_ROOT", "."),
        )

    factory = service_factory or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = factory()
        yield

    app = FastAPI(
        title="AeroMind RUL API",
        version=__version__,
        lifespan=lifespan,
        description=(
            "Remaining-useful-life inference for NASA C-MAPSS turbofan simulation data. Research "
            "demonstration only: not certified for operational aviation maintenance decisions. "
            "Independent project; not endorsed by NASA."
        ),
    )

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:  # no stack traces to clients
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "Unexpected server error", "details": None}})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()[:20]]
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Request body failed schema validation", "details": details}})

    def svc(request: Request) -> ModelService:
        return request.app.state.service

    @app.get("/health", tags=["service"])
    def health(request: Request) -> dict[str, Any]:
        s = svc(request)
        avail = [m for m in s.model_ids if s.availability(m)[0]]
        return {
            "status": "ok" if avail else "degraded",
            "api_version": API_VERSION,
            "package_version": __version__,
            "registry_built_utc": s.registry["built_utc"],
            "models_registered": len(s.model_ids),
            "models_available": len(avail),
            "default_model": s.registry.get("default_model"),
        }

    @app.get("/models", tags=["models"])
    def list_models(request: Request) -> dict[str, Any]:
        s = svc(request)
        return {"default_model": s.registry.get("default_model"), "models": [s.summary(m) for m in s.model_ids]}

    @app.get("/models/{model_id}", tags=["models"])
    def model_metadata(request: Request, model_id: ModelId) -> dict[str, Any]:
        s = svc(request)
        e = s.entry(model_id)
        ok, reason = s.availability(model_id)
        meta = {k: v for k, v in e.items() if k not in ("training",)}
        meta["training"] = {k: v for k, v in e["training"].items() if k != "engine_ids"}
        meta["available"], meta["availability_reason"] = ok, reason
        return meta

    @app.post("/models/{model_id}/predict", tags=["inference"], response_model=Prediction)
    def predict(request: Request, model_id: ModelId, body: PredictRequest) -> dict[str, Any]:
        return svc(request).predict_engine(
            model_id, _frame(body.engine), body.engine.engine_id, body.interval_levels, body.include_trajectory
        )

    @app.post("/models/{model_id}/predict/batch", tags=["inference"], response_model=BatchPrediction)
    def predict_batch(request: Request, model_id: ModelId, body: BatchPredictRequest) -> dict[str, Any]:
        s = svc(request)
        ids = [e.engine_id for e in body.engines]
        if len(set(ids)) != len(ids):
            raise InvalidInput("engine_id values must be unique within a batch")
        preds = [
            s.predict_engine(model_id, _frame(e), e.engine_id, body.interval_levels, body.include_trajectory)
            for e in body.engines
        ]
        return {
            "model_id": model_id,
            "n_engines": len(preds),
            "total_inference_ms": float(sum(p["inference_ms"] for p in preds)),
            "predictions": preds,
        }

    return app


app = create_app()
