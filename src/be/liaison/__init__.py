"""
Liaison layer for BE Diegesis.

This package provides API wiring (routers/handlers) that mediate external
actors to a BEDFrame without owning server lifecycle.
"""

from .api import create_bed_router

__all__ = ["create_bed_router"]
