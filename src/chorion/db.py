"""
Chorion database — async MongoDB via motor.

Shares the MongoDB instance with HatChat, using chorion_* prefixed collections.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("CHORION_DB_NAME", "LibreChat")

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect() -> AsyncIOMotorDatabase:
    """Connect to MongoDB and return the database handle."""
    global _client, _db
    if _db is not None:
        return _db

    logger.info("Connecting to MongoDB at %s (db: %s)", MONGO_URI, DB_NAME)
    _client = AsyncIOMotorClient(MONGO_URI)
    _db = _client[DB_NAME]

    # Ensure indexes for chorion collections
    await _db.chorion_models.create_index("model_key", unique=True)
    await _db.chorion_jobs.create_index([("status", 1), ("created_at", -1)])
    await _db.chorion_jobs.create_index("model_key")
    await _db.chorion_audit.create_index([("timestamp", -1)])
    await _db.chorion_audit.create_index("actor_id")

    logger.info("MongoDB connected, indexes ensured")
    return _db


async def disconnect():
    """Close the MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB disconnected")


def get_db() -> AsyncIOMotorDatabase:
    """Get the current database handle. Call connect() first."""
    if _db is None:
        raise RuntimeError("Database not connected — call connect() first")
    return _db
