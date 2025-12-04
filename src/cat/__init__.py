"""
Conjoined Adversarial Tomograph (CAT) - Layer 2.5 Oversight Component

CAT is an analysis/oversight component that:
- consumes HAT/MAP probe streams and external context over one or more world ticks
- summarises and interprets the internal headspace of a subject
- grades divergence between internal state and external/contractual expectations
- recommends actions (including escalation) under ASK/USH/CSH treaties

CAT bridges cognitive scale gaps:
- between a substrate and its observer (e.g. tiny oversight model on a 70B)
- between concept packs/tribes (via MAP translations)
- between raw probe traces and human-/policy-level judgements
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
    ProbeTraces,
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
    "ProbeTraces",
    "ExternalContext",
    "MAPTranslation",
    # Enums
    "CATSizeClass",
    "AlertSeverity",
    "AlertKind",
    "WindowReason",
    "ActionKind",
]
