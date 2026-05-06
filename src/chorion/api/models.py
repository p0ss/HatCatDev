"""
Chorion model registry API.

Wraps the existing MODEL_CANDIDATES from Thalamos with MongoDB persistence,
adding lifecycle status tracking for the ontology pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from chorion.auth import ActorClaims, require_role, validate_service_jwt
from chorion.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chorion/models", tags=["models"])


# --- Schemas ---

class ModelRegistration(BaseModel):
    """Request body for registering a new model."""
    model_key: str = Field(..., description="Short identifier, e.g. 'gemma-4-E4B-it'")
    model_id: str = Field(..., description="HuggingFace model ID, e.g. 'google/gemma-4-E4B-it'")
    name: str = Field(..., description="Human-readable name")
    model_class: str = Field("AutoModelForCausalLM", description="Transformers model class")
    dtype: str = Field("bfloat16")
    trust_remote_code: bool = False
    quantization: Optional[str] = Field(None, description="None, '4bit', or '8bit'")
    params_billions: float = 0
    architecture: str = "transformer"
    is_multimodal: bool = False
    reasoning_mode: Optional[str] = None
    vram_gb_estimate: float = 0
    notes: str = ""


class PipelineStatus(BaseModel):
    """Tracks which stages of the ontology pipeline are complete."""
    pillars: Optional[str] = None        # "complete", "running", "failed", or None
    skeleton: Optional[str] = None
    qualification: Optional[str] = None
    melds: Optional[str] = None
    lens_training: Optional[str] = None
    lens_pack: Optional[str] = None
    calibration: Optional[str] = None
    deployed: bool = False


class ModelRecord(BaseModel):
    """Full model record as stored in MongoDB."""
    model_key: str
    model_id: str
    name: str
    model_class: str = "AutoModelForCausalLM"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    quantization: Optional[str] = None
    params_billions: float = 0
    architecture: str = "transformer"
    is_multimodal: bool = False
    reasoning_mode: Optional[str] = None
    vram_gb_estimate: float = 0
    notes: str = ""
    pipeline: PipelineStatus = Field(default_factory=PipelineStatus)
    registered_by: str = ""
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Audit helper ---

async def _audit(actor: ActorClaims, action: str, resource_id: str, detail: str = ""):
    db = get_db()
    await db.chorion_audit.insert_one({
        "timestamp": datetime.now(timezone.utc),
        "actor_id": actor.user_id,
        "actor_role": actor.role,
        "action": action,
        "resource_type": "model",
        "resource_id": resource_id,
        "detail": detail,
    })


# --- Routes ---

@router.get("")
async def list_models(actor: ActorClaims = Depends(validate_service_jwt)):
    """List all registered models with pipeline status."""
    db = get_db()
    cursor = db.chorion_models.find({}, {"_id": 0})
    models = await cursor.to_list(length=200)
    return {"models": models}


@router.get("/{model_key}")
async def get_model(model_key: str, actor: ActorClaims = Depends(validate_service_jwt)):
    """Get a single model by key."""
    db = get_db()
    doc = await db.chorion_models.find_one({"model_key": model_key}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    return doc


@router.post("", status_code=status.HTTP_201_CREATED)
async def register_model(
    body: ModelRegistration,
    actor: ActorClaims = Depends(require_role("operator", "researcher")),
):
    """Register a new model candidate."""
    db = get_db()

    existing = await db.chorion_models.find_one({"model_key": body.model_key})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{body.model_key}' already registered",
        )

    record = ModelRecord(
        **body.model_dump(),
        registered_by=actor.user_id,
    )
    await db.chorion_models.insert_one(record.model_dump())
    await _audit(actor, "model:register", body.model_key, f"Registered {body.model_id}")
    logger.info("Model registered: %s (%s) by %s", body.model_key, body.model_id, actor.user_id)

    return record.model_dump()


@router.put("/{model_key}")
async def update_model(
    model_key: str,
    body: ModelRegistration,
    actor: ActorClaims = Depends(require_role("operator")),
):
    """Update an existing model's configuration."""
    db = get_db()

    result = await db.chorion_models.update_one(
        {"model_key": model_key},
        {"$set": {
            **body.model_dump(),
            "model_key": model_key,  # key is immutable
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")

    await _audit(actor, "model:update", model_key, f"Updated config")
    return {"status": "updated", "model_key": model_key}


@router.delete("/{model_key}")
async def deregister_model(
    model_key: str,
    actor: ActorClaims = Depends(require_role("operator")),
):
    """Soft-delete a model (marks as deregistered, does not remove data)."""
    db = get_db()

    result = await db.chorion_models.update_one(
        {"model_key": model_key},
        {"$set": {"deregistered": True, "deregistered_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")

    await _audit(actor, "model:deregister", model_key)
    return {"status": "deregistered", "model_key": model_key}


@router.get("/{model_key}/status")
async def get_pipeline_status(
    model_key: str,
    actor: ActorClaims = Depends(validate_service_jwt),
):
    """Get the full pipeline lifecycle status for a model."""
    db = get_db()
    doc = await db.chorion_models.find_one(
        {"model_key": model_key},
        {"_id": 0, "pipeline": 1, "model_key": 1, "name": 1},
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    return doc
