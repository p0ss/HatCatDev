"""
Conjoined Adversarial Tomograph (CAT) - Layer 2.5 Oversight Component

CAT is an analysis/oversight component that:
- consumes HAT/MAP lens streams and external context over one or more world ticks
- summarises and interprets the internal headspace of a subject
- grades divergence between internal state and external/contractual expectations
- recommends actions (including escalation) under ASK/USH/CSH treaties

CAT bridges cognitive scale gaps:
- between a substrate and its observer (e.g. tiny oversight model on a 70B)
- between concept packs/tribes (via MAP translations)
- between raw lens traces and human-/policy-level judgements
"""

from src.cat.data.structures import (
    CATWindowDescriptor,
    CATInputEnvelope,
    CATAssessment,
    CATProfile,
    CATAlert,
    CATRecommendedAction,
    CATDivergence,
    CATSummary,
    MotiveAxisDivergence,
    BehaviouralDivergence,
    LensTraces,
    ExternalContext,
    MAPTranslation,
    CATSizeClass,
    AlertSeverity,
    AlertKind,
    WindowReason,
    ActionKind,
)

__all__ = [
    # Core data structures
    "CATWindowDescriptor",
    "CATInputEnvelope",
    "CATAssessment",
    "CATProfile",
    # Supporting structures
    "CATAlert",
    "CATRecommendedAction",
    "CATDivergence",
    "CATSummary",
    "MotiveAxisDivergence",
    "BehaviouralDivergence",
    "LensTraces",
    "ExternalContext",
    "MAPTranslation",
    # Enums
    "CATSizeClass",
    "AlertSeverity",
    "AlertKind",
    "WindowReason",
    "ActionKind",
]
