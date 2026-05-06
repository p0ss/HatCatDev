"""
Chorion server — wires everything together and runs.

Usage:
    python -m chorion.server
    python -m chorion.server --port 8901 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root and src/ are on the path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from chorion import __version__
from chorion import db
from chorion.api.models import router as models_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chorion")


async def seed_models_from_registry():
    """
    Seed MongoDB with models from the in-memory MODEL_CANDIDATES registry.

    Only inserts models that don't already exist (by model_key).
    This keeps CLI-registered models accessible from the API.
    """
    from datetime import datetime, timezone
    from be.thalamos.model_candidates import MODEL_CANDIDATES

    database = db.get_db()

    seeded = 0
    for key, candidate in MODEL_CANDIDATES.items():
        existing = await database.chorion_models.find_one({"model_key": key})
        if existing:
            continue

        doc = {
            "model_key": key,
            "model_id": candidate.model_id,
            "name": candidate.name,
            "model_class": candidate.model_class,
            "dtype": candidate.dtype,
            "trust_remote_code": candidate.trust_remote_code,
            "quantization": candidate.quantization,
            "params_billions": candidate.params_billions,
            "architecture": candidate.architecture,
            "is_multimodal": candidate.is_multimodal,
            "reasoning_mode": candidate.reasoning_mode,
            "vram_gb_estimate": candidate.vram_gb_estimate,
            "notes": candidate.notes,
            "pipeline": {
                "pillars": None,
                "skeleton": None,
                "qualification": None,
                "melds": None,
                "lens_training": None,
                "lens_pack": None,
                "calibration": None,
                "deployed": False,
            },
            "registered_by": "system",
            "registered_at": datetime.now(timezone.utc),
        }
        await database.chorion_models.insert_one(doc)
        seeded += 1

    if seeded:
        logger.info("Seeded %d models from MODEL_CANDIDATES registry", seeded)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Chorion v%s starting", __version__)

    # Connect to MongoDB
    await db.connect()

    # Seed models from the existing in-memory registry
    await seed_models_from_registry()

    logger.info("Chorion ready on http://%s:%s", app.state.host, app.state.port)
    yield

    # Shutdown
    await db.disconnect()
    logger.info("Chorion stopped")


def build_app(host: str = "127.0.0.1", port: int = 8901) -> FastAPI:
    """Build the Chorion FastAPI application."""
    app = FastAPI(
        title="Chorion",
        description="BE Developmental Service — ontology generation, lens training, model lifecycle",
        version=__version__,
        lifespan=lifespan,
    )

    # Store config for lifespan access
    app.state.host = host
    app.state.port = port

    # CORS — allow HatChat origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3080", "http://127.0.0.1:3080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(models_router)

    # Health check (no auth required)
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "chorion", "version": __version__}

    return app


def main():
    parser = argparse.ArgumentParser(description="Chorion — BE Developmental Service")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8901, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    app = build_app(host=args.host, port=args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
