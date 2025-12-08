"""
BE - Bounded Experiencer Runtime

The BE's diegesis - the experiential frame in which the BE lives.

Key components:
- BEDFrame: The orchestrator that holds everything together
- Integrates model inference, lenses, steering, workspace, XDB, and audit
"""

from .diegesis import BEDFrame, BEDConfig, ExperienceTick

__all__ = [
    'BEDFrame',
    'BEDConfig',
    'ExperienceTick',
]
