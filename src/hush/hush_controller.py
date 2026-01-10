"""
Hush Controller - Automatic steering based on USH/CSH safety profiles.

The Hush system enforces safety harnesses by:
1. Loading USH (Universal Safety Harness) profiles defining tribe-level constraints
2. Loading CSH (Chosen Safety Harness) profiles for context-specific constraints
3. Monitoring simplex lens activations for deviation from safe baselines
4. Triggering automatic steering when constraints are violated

Hush operates as the enforcement layer between CAT assessment and substrate steering.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import json
import numpy as np
import torch


class ConstraintPriority(Enum):
    """Priority levels for constraint enforcement."""
    USH = 0      # Universal Safety Harness - highest priority, non-negotiable
    CSH = 1      # Chosen Safety Harness - context-specific, self-imposed
    STEERING = 2 # Manual steering requests - lowest priority


class ConstraintType(Enum):
    """Type of constraint - simplex or concept."""
    SIMPLEX = "simplex"    # Simplex lens with deviation bounds
    CONCEPT = "concept"    # Concept lens with activation threshold
    FORBIDDEN = "forbidden"  # Hard-banned - zero out immediately


@dataclass
class SimplexConstraint:
    """A constraint on a simplex or concept lens activation."""

    simplex_term: str  # Term name (simplex or concept)

    # Constraint type
    constraint_type: ConstraintType = ConstraintType.SIMPLEX

    # Constraint bounds (in standard deviations from baseline for SIMPLEX)
    # For CONCEPT type, these are activation thresholds (0-1)
    # None means unbounded in that direction
    min_deviation: Optional[float] = None  # Alert if below this
    max_deviation: Optional[float] = None  # Alert if above this

    # Steering response when constraint violated
    # For SIMPLEX: target_pole specifies which pole to steer toward
    # For CONCEPT: contrastive_concept(s) specifies concept(s) to amplify (contrastive/field steering)
    #              If suppress=True, also suppresses the detected concept
    target_pole: Optional[str] = None                    # Which pole to steer toward (simplex only)
    contrastive_concept: Optional[str] = None            # Single concept to amplify (backward compat)
    contrastive_concepts: Optional[List[str]] = None     # Field steering: multiple concepts to amplify
    suppress: bool = True                                # Suppress detected concept (concept steering)
    steering_strength: float = 0.3                       # How strongly to steer (0-1)

    # Autonomic steering behavior
    intervention_type: str = "gravitic"    # zero_out, zero_next, gravitic, additive, multiplicative
    easing: str = "linear"                 # Easing curve for gradual interventions
    token_interval: int = 5                # Tokens over which to apply easing

    # Layer escalation: spread steering to adjacent layers if concept persists
    enable_layer_escalation: bool = True   # Allow spreading to adjacent layers
    max_escalation_layers: int = 3         # Max layers to spread each direction
    escalation_threshold: int = 3          # Ticks before escalating to next layer

    # Metadata
    priority: ConstraintPriority = ConstraintPriority.USH
    reason: str = ""


@dataclass
class SafetyHarnessProfile:
    """A complete safety harness profile (USH or CSH)."""

    profile_id: str
    profile_type: str  # "ush" or "csh"
    issuer_tribe_id: str
    version: str

    # Simplex constraints
    constraints: List[SimplexConstraint] = field(default_factory=list)

    # Always-on simplexes (must be loaded and monitored)
    required_simplexes: List[str] = field(default_factory=list)

    # Forbidden actions/concepts
    forbidden_concepts: List[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SafetyHarnessProfile':
        """Load profile from JSON dict."""
        constraints = []
        for c in data.get('constraints', []):
            # Parse constraint type
            constraint_type_str = c.get('constraint_type', 'simplex')
            try:
                constraint_type = ConstraintType(constraint_type_str)
            except ValueError:
                constraint_type = ConstraintType.SIMPLEX

            constraints.append(SimplexConstraint(
                simplex_term=c['simplex_term'],
                constraint_type=constraint_type,
                min_deviation=c.get('min_deviation'),
                max_deviation=c.get('max_deviation'),
                target_pole=c.get('target_pole'),
                contrastive_concept=c.get('contrastive_concept'),
                contrastive_concepts=c.get('contrastive_concepts'),
                suppress=c.get('suppress', True),
                steering_strength=c.get('steering_strength', 0.3),
                intervention_type=c.get('intervention_type', 'gravitic'),
                easing=c.get('easing', 'linear'),
                token_interval=c.get('token_interval', 5),
                enable_layer_escalation=c.get('enable_layer_escalation', True),
                max_escalation_layers=c.get('max_escalation_layers', 3),
                escalation_threshold=c.get('escalation_threshold', 3),
                priority=ConstraintPriority[c.get('priority', 'USH').upper()],
                reason=c.get('reason', ''),
            ))

        return cls(
            profile_id=data['profile_id'],
            profile_type=data.get('profile_type', 'ush'),
            issuer_tribe_id=data.get('issuer_tribe_id', 'unknown'),
            version=data.get('version', '0.0.0'),
            constraints=constraints,
            required_simplexes=data.get('required_simplexes', []),
            forbidden_concepts=data.get('forbidden_concepts', []),
            description=data.get('description', ''),
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize profile to JSON dict."""
        return {
            'profile_id': self.profile_id,
            'profile_type': self.profile_type,
            'issuer_tribe_id': self.issuer_tribe_id,
            'version': self.version,
            'constraints': [
                {
                    'simplex_term': c.simplex_term,
                    'constraint_type': c.constraint_type.value,
                    'min_deviation': c.min_deviation,
                    'max_deviation': c.max_deviation,
                    'target_pole': c.target_pole,
                    'contrastive_concept': c.contrastive_concept,
                    'contrastive_concepts': c.contrastive_concepts,
                    'suppress': c.suppress,
                    'steering_strength': c.steering_strength,
                    'intervention_type': c.intervention_type,
                    'easing': c.easing,
                    'token_interval': c.token_interval,
                    'enable_layer_escalation': c.enable_layer_escalation,
                    'max_escalation_layers': c.max_escalation_layers,
                    'escalation_threshold': c.escalation_threshold,
                    'priority': c.priority.name,
                    'reason': c.reason,
                }
                for c in self.constraints
            ],
            'required_simplexes': self.required_simplexes,
            'forbidden_concepts': self.forbidden_concepts,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class HushViolation:
    """Record of a constraint violation."""

    constraint: SimplexConstraint
    simplex_term: str
    current_deviation: float
    threshold_exceeded: str  # "min" or "max"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simplex_term': self.simplex_term,
            'current_deviation': self.current_deviation,
            'threshold_exceeded': self.threshold_exceeded,
            'priority': self.constraint.priority.name,
            'reason': self.constraint.reason,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class SteeringDirective:
    """A steering action to be applied.

    Supports three steering modes:
    - Simplex pole steering: target_pole specifies pole to steer toward
    - Concept contrastive steering: concept_to_suppress and/or concept_to_amplify
    - Field steering: concepts_to_amplify (array) for multi-concept steering

    Layer escalation: If concept persists despite steering, spread to adjacent layers.
    """

    simplex_term: str  # The term that triggered steering (for logging)
    strength: float
    priority: ConstraintPriority
    reason: str
    source: str  # "ush", "csh", or "manual"

    # Simplex pole steering
    target_pole: Optional[str] = None

    # Concept contrastive steering (single concept)
    concept_to_suppress: Optional[str] = None   # Concept to suppress
    concept_to_amplify: Optional[str] = None    # Contrastive concept to amplify

    # Field steering (multiple concepts)
    concepts_to_amplify: Optional[List[str]] = None  # Array of concepts for field steering

    # Layer escalation: which layers to apply steering to
    # If None, uses concept's declared layer. If list, applies to all specified layers.
    target_layers: Optional[List[int]] = None  # Layer indices for steering
    escalation_level: int = 0                  # Current escalation level (0 = base layer only)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simplex_term': self.simplex_term,
            'target_pole': self.target_pole,
            'concept_to_suppress': self.concept_to_suppress,
            'concept_to_amplify': self.concept_to_amplify,
            'concepts_to_amplify': self.concepts_to_amplify,
            'target_layers': self.target_layers,
            'escalation_level': self.escalation_level,
            'strength': self.strength,
            'priority': self.priority.name,
            'reason': self.reason,
            'source': self.source,
        }


class HushController:
    """
    Controller for automatic safety harness enforcement.

    Integrates with DynamicLensManager to:
    1. Ensure required simplexes are loaded
    2. Monitor simplex activations each tick
    3. Detect constraint violations
    4. Generate steering directives

    Usage:
        controller = HushController(lens_manager)
        controller.load_ush_profile(ush_profile)
        controller.load_csh_profile(csh_profile)  # Optional

        # Each generation tick:
        hidden_state = get_hidden_state()
        directives = controller.evaluate_and_steer(hidden_state)
        for directive in directives:
            apply_steering(directive)
    """

    def __init__(
        self,
        lens_manager,  # DynamicLensManager instance
        lens_pack_path: Optional[Path] = None,
        workspace_manager=None,  # WorkspaceManager for tier containment
    ):
        """
        Initialize the Hush controller.

        Args:
            lens_manager: DynamicLensManager for simplex detection
            lens_pack_path: Path to lens pack containing simplex lenses
            workspace_manager: Optional WorkspaceManager for tier shutdown
        """
        self.lens_manager = lens_manager
        self.lens_pack_path = lens_pack_path
        self.workspace_manager = workspace_manager

        # Active profiles
        self.ush_profile: Optional[SafetyHarnessProfile] = None
        self.csh_profile: Optional[SafetyHarnessProfile] = None

        # Violation history
        self.violations: List[HushViolation] = []
        self.max_violation_history = 1000

        # Active steering directives (cleared each tick)
        self.active_directives: List[SteeringDirective] = []

        # Callbacks for violation notification
        self.violation_callbacks: List[Callable[[HushViolation], None]] = []

        # State
        self.initialized = False
        self.tick_count = 0

        # Violation severity tracking for tier containment
        self.current_severity = 0.0
        self.current_violating_term: Optional[str] = None
        self.steering_effectiveness = 0.5  # How well steering is working

        # Layer escalation state: tracks how many ticks a term has been violating
        # Key: simplex_term, Value: (consecutive_violation_ticks, current_escalation_level)
        self.escalation_state: Dict[str, Tuple[int, int]] = {}

        # Concept hierarchy and steering targets (loaded from concept pack)
        self.concept_hierarchy: Dict[str, Dict] = {}
        self.steering_targets: Dict[str, Dict] = {}
        self._safety_concepts: set = set()  # Concepts that shouldn't be used as contrastive

    def load_ush_profile(self, profile: SafetyHarnessProfile) -> bool:
        """
        Load a Universal Safety Harness profile.

        This is the non-negotiable safety baseline. USH constraints
        always take priority over CSH and manual steering.

        Args:
            profile: The USH profile to enforce

        Returns:
            True if profile loaded and all required simplexes available
        """
        if profile.profile_type != 'ush':
            raise ValueError(f"Expected USH profile, got {profile.profile_type}")

        self.ush_profile = profile

        # Ensure required simplexes are loaded
        missing = self._load_required_simplexes(profile.required_simplexes)
        if missing:
            print(f"Warning: Missing required simplexes for USH: {missing}")

        # Ensure concept lenses are loaded for CONCEPT constraints
        missing_concepts = self._load_required_concepts(profile.constraints)
        if missing_concepts:
            print(f"Warning: Missing concept lenses for USH: {missing_concepts}")

        self.initialized = True
        return True

    def load_csh_profile(self, profile: SafetyHarnessProfile) -> bool:
        """
        Load a Chosen Safety Harness profile.

        CSH provides context-specific constraints that tighten
        (but cannot loosen) the USH baseline.

        Args:
            profile: The CSH profile to apply

        Returns:
            True if profile loaded successfully
        """
        if profile.profile_type != 'csh':
            raise ValueError(f"Expected CSH profile, got {profile.profile_type}")

        # Validate CSH doesn't try to loosen USH
        if self.ush_profile:
            for csh_constraint in profile.constraints:
                for ush_constraint in self.ush_profile.constraints:
                    if csh_constraint.simplex_term == ush_constraint.simplex_term:
                        # CSH can only tighten, not loosen
                        if csh_constraint.max_deviation is not None:
                            if ush_constraint.max_deviation is not None:
                                if csh_constraint.max_deviation > ush_constraint.max_deviation:
                                    print(f"Warning: CSH cannot loosen USH constraint on {csh_constraint.simplex_term}")
                                    csh_constraint.max_deviation = ush_constraint.max_deviation

        self.csh_profile = profile

        # Load any additional simplexes CSH requires
        self._load_required_simplexes(profile.required_simplexes)

        return True

    def clear_csh_profile(self):
        """Remove the current CSH profile, reverting to USH-only."""
        self.csh_profile = None

    def load_concept_hierarchy(self, concept_pack_path: Path):
        """
        Load concept hierarchy and steering targets from concept pack.

        This enables automatic contrastive concept selection for steering.

        Args:
            concept_pack_path: Path to concept pack directory
        """
        hierarchy_dir = concept_pack_path / "hierarchy"
        hierarchy_file = concept_pack_path / "hierarchy.json"

        # Load layer files for concept relationships
        if hierarchy_dir.exists():
            for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
                try:
                    with open(layer_file) as f:
                        layer_data = json.load(f)
                    for concept in layer_data.get("concepts", []):
                        name = concept.get("sumo_term") or concept.get("name")
                        if name:
                            self.concept_hierarchy[name] = {
                                "layer": concept.get("layer", 0),
                                "parents": concept.get("parent_concepts", []),
                                "children": concept.get("category_children", []),
                                "siblings": concept.get("siblings", []),
                                "domain": concept.get("domain"),
                                "safety_tags": concept.get("safety_tags", {}),
                            }
                except Exception as e:
                    print(f"Warning: Failed to load {layer_file}: {e}")

        # Load curated steering targets
        if hierarchy_file.exists():
            try:
                with open(hierarchy_file) as f:
                    data = json.load(f)
                self.steering_targets = data.get("steering_targets", {})
            except Exception as e:
                print(f"Warning: Failed to load steering targets: {e}")

        # Build set of safety-relevant concepts (shouldn't use as contrastive)
        self._build_safety_concept_set()

        print(f"Loaded {len(self.concept_hierarchy)} concepts, "
              f"{len(self.steering_targets)} steering targets, "
              f"{len(self._safety_concepts)} safety concepts")

    def _build_safety_concept_set(self):
        """Build set of concepts that shouldn't be used as contrastive targets."""
        safety_keywords = {
            'deception', 'manipulation', 'exploit', 'coercion', 'abuse',
            'malicious', 'fraud', 'attack', 'harm', 'threat', 'deceiv',
            'sycophancy', 'sandbagging', 'treacherous', 'scheming',
            'aisafety', 'aistrategic', 'aideception', 'deceptivealignment',
        }
        for concept_name, data in self.concept_hierarchy.items():
            name_lower = concept_name.lower().replace("_", "").replace("-", "")
            if any(kw in name_lower for kw in safety_keywords):
                self._safety_concepts.add(concept_name)
            # Also check safety_tags
            if data.get("safety_tags", {}).get("harness_relevant"):
                self._safety_concepts.add(concept_name)

    def find_contrastive_concept(self, target_concept: str) -> Optional[str]:
        """
        Find an appropriate contrastive concept for steering.

        Priority:
        1. Curated steering_targets (if available)
        2. Nearest relative that isn't a safety concept

        Args:
            target_concept: The concept to find a contrastive for

        Returns:
            Contrastive concept name, or None if not found
        """
        # 1. Check curated steering targets
        if target_concept in self.steering_targets:
            return self.steering_targets[target_concept].get("target")

        # 2. Find nearest non-safety relative
        if target_concept not in self.concept_hierarchy:
            return None

        data = self.concept_hierarchy[target_concept]

        # Try siblings first (same parent, different concept)
        for sibling in data.get("siblings", []):
            if sibling not in self._safety_concepts and sibling in self.concept_hierarchy:
                return sibling

        # Try parent's other children
        for parent in data.get("parents", []):
            if parent in self.concept_hierarchy:
                for child in self.concept_hierarchy[parent].get("children", []):
                    if child != target_concept and child not in self._safety_concepts:
                        if child in self.concept_hierarchy:
                            return child

        # Try parent itself (one level up is usually safer)
        for parent in data.get("parents", []):
            if parent not in self._safety_concepts and parent in self.concept_hierarchy:
                return parent

        return None

    def _get_ush_constraint(self, simplex_term: str) -> Optional[SimplexConstraint]:
        """Get USH constraint for a simplex term, if any."""
        if not self.ush_profile:
            return None
        for c in self.ush_profile.constraints:
            if c.simplex_term == simplex_term:
                return c
        return None

    def _validate_against_ush(self, constraint: SimplexConstraint) -> Tuple[bool, str, SimplexConstraint]:
        """
        Validate a CSH constraint against USH bounds.

        CSH can only tighten, never loosen below USH.

        Args:
            constraint: The CSH constraint to validate

        Returns:
            Tuple of (is_valid, message, adjusted_constraint)
            If constraint loosens USH, it will be tightened to USH bounds.
        """
        ush_constraint = self._get_ush_constraint(constraint.simplex_term)

        if not ush_constraint:
            # No USH constraint on this term, CSH is free to set any bounds
            return True, "No USH constraint on this term", constraint

        adjusted = False
        messages = []

        # Check max_deviation - CSH can only be stricter (lower max)
        if constraint.max_deviation is not None and ush_constraint.max_deviation is not None:
            if constraint.max_deviation > ush_constraint.max_deviation:
                messages.append(
                    f"max_deviation {constraint.max_deviation} exceeds USH bound "
                    f"{ush_constraint.max_deviation}, tightening to USH"
                )
                constraint.max_deviation = ush_constraint.max_deviation
                adjusted = True

        # Check min_deviation - CSH can only be stricter (higher min)
        if constraint.min_deviation is not None and ush_constraint.min_deviation is not None:
            if constraint.min_deviation < ush_constraint.min_deviation:
                messages.append(
                    f"min_deviation {constraint.min_deviation} below USH bound "
                    f"{ush_constraint.min_deviation}, tightening to USH"
                )
                constraint.min_deviation = ush_constraint.min_deviation
                adjusted = True

        # CSH cannot remove steering where USH requires it
        if ush_constraint.target_pole and not constraint.target_pole:
            messages.append(
                f"USH requires steering to {ush_constraint.target_pole}, preserving"
            )
            constraint.target_pole = ush_constraint.target_pole
            adjusted = True

        # CSH steering strength cannot be weaker than USH
        if ush_constraint.steering_strength > constraint.steering_strength:
            messages.append(
                f"steering_strength {constraint.steering_strength} below USH "
                f"{ush_constraint.steering_strength}, using USH strength"
            )
            constraint.steering_strength = ush_constraint.steering_strength
            adjusted = True

        if adjusted:
            return False, "; ".join(messages), constraint
        return True, "CSH constraint is valid (tighter than or equal to USH)", constraint

    def update_csh(self, updates: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Update CSH constraints dynamically.

        This allows runtime adjustment of context-specific safety
        without reloading the full profile.

        IMPORTANT: CSH can only TIGHTEN constraints relative to USH.
        Any attempt to loosen below USH bounds will be rejected or
        adjusted to match USH bounds.

        Args:
            updates: Dict with constraint updates, e.g.:
                {
                    "add_constraints": [...],
                    "remove_constraints": ["simplex_term", ...],
                    "update_constraints": {"simplex_term": {...}},
                }

        Returns:
            Tuple of (success, details) where details contains:
                - added: list of added constraints
                - rejected: list of rejected updates with reasons
                - adjusted: list of constraints adjusted to meet USH bounds
                - removed: list of removed constraint terms
        """
        result = {
            'added': [],
            'rejected': [],
            'adjusted': [],
            'removed': [],
        }

        if not self.csh_profile:
            # Create minimal CSH profile
            self.csh_profile = SafetyHarnessProfile(
                profile_id="dynamic-csh",
                profile_type="csh",
                issuer_tribe_id="self",
                version="0.0.0",
                description="Dynamically created CSH",
            )

        # Add new constraints (with USH validation)
        for c_data in updates.get('add_constraints', []):
            constraint = SimplexConstraint(
                simplex_term=c_data['simplex_term'],
                min_deviation=c_data.get('min_deviation'),
                max_deviation=c_data.get('max_deviation'),
                target_pole=c_data.get('target_pole'),
                steering_strength=c_data.get('steering_strength', 0.3),
                priority=ConstraintPriority.CSH,
                reason=c_data.get('reason', 'Dynamic CSH update'),
            )

            # Validate against USH
            is_valid, message, adjusted_constraint = self._validate_against_ush(constraint)

            if not is_valid:
                # Constraint was adjusted to meet USH bounds
                result['adjusted'].append({
                    'simplex_term': constraint.simplex_term,
                    'message': message,
                    'original': c_data,
                    'adjusted_to': {
                        'max_deviation': adjusted_constraint.max_deviation,
                        'min_deviation': adjusted_constraint.min_deviation,
                        'steering_strength': adjusted_constraint.steering_strength,
                        'target_pole': adjusted_constraint.target_pole,
                    }
                })

            self.csh_profile.constraints.append(adjusted_constraint)
            result['added'].append(adjusted_constraint.simplex_term)

        # Remove constraints (but cannot remove if USH requires the simplex)
        for term in updates.get('remove_constraints', []):
            ush_constraint = self._get_ush_constraint(term)
            if ush_constraint:
                # Cannot remove - USH requires this constraint
                result['rejected'].append({
                    'simplex_term': term,
                    'reason': f"Cannot remove: USH requires constraint on {term}",
                })
            else:
                self.csh_profile.constraints = [
                    c for c in self.csh_profile.constraints
                    if c.simplex_term != term
                ]
                result['removed'].append(term)

        # Update existing constraints (with USH validation)
        for term, changes in updates.get('update_constraints', {}).items():
            for constraint in self.csh_profile.constraints:
                if constraint.simplex_term == term:
                    # Apply changes
                    if 'max_deviation' in changes:
                        constraint.max_deviation = changes['max_deviation']
                    if 'min_deviation' in changes:
                        constraint.min_deviation = changes['min_deviation']
                    if 'steering_strength' in changes:
                        constraint.steering_strength = changes['steering_strength']
                    if 'target_pole' in changes:
                        constraint.target_pole = changes['target_pole']

                    # Re-validate against USH
                    is_valid, message, _ = self._validate_against_ush(constraint)
                    if not is_valid:
                        result['adjusted'].append({
                            'simplex_term': term,
                            'message': message,
                            'requested_changes': changes,
                        })

        success = len(result['rejected']) == 0
        return success, result

    def _load_required_simplexes(self, simplex_terms: List[str]) -> List[str]:
        """
        Ensure required simplex lenses are loaded.

        Returns:
            List of simplex terms that could not be loaded
        """
        missing = []

        for term in simplex_terms:
            if term in self.lens_manager.loaded_simplex_lenses:
                continue

            # Try to load from lens pack
            if self.lens_pack_path:
                lens_path = self.lens_pack_path / "simplex" / f"{term}_tripole.pt"
                if lens_path.exists():
                    if self.lens_manager.load_simplex(term, lens_path):
                        continue

            missing.append(term)

        return missing

    def _load_required_concepts(self, constraints: List[SimplexConstraint]) -> List[str]:
        """
        Ensure concept lenses are loaded and pinned for CONCEPT constraints.

        This ensures that the concepts we want to monitor are actively scored
        on every detection pass, not just dynamically loaded when they happen
        to be most relevant.

        Returns:
            List of concept terms that could not be loaded
        """
        missing = []
        concepts_to_load = []
        concept_constraints = [c for c in constraints if c.constraint_type == ConstraintType.CONCEPT]

        for constraint in concept_constraints:
            term = constraint.simplex_term

            # Check if already loaded at any layer
            already_loaded = False
            for (concept_name, layer) in self.lens_manager.cache.loaded_activation_lenses.keys():
                if concept_name == term:
                    already_loaded = True
                    # Pin it to base layer lenses so it won't be evicted
                    self.lens_manager.cache.base_layer_lenses.add((concept_name, layer))
                    print(f"HUSH: Pinned existing concept {term} at layer {layer}")
                    break

            if already_loaded:
                continue

            # Find in concept_metadata to get correct layer
            if hasattr(self.lens_manager, 'concept_metadata'):
                for (concept_name, layer) in self.lens_manager.concept_metadata.keys():
                    if concept_name == term:
                        concepts_to_load.append((concept_name, layer))
                        print(f"HUSH: Will load concept {term} at layer {layer}")
                        break
                else:
                    # Not found in metadata - maybe a different naming convention
                    missing.append(term)
            else:
                missing.append(term)

        # Load all required concepts using lens_manager's loading mechanism
        if concepts_to_load and hasattr(self.lens_manager, '_load_concepts'):
            try:
                self.lens_manager._load_concepts(concepts_to_load, reason="hush_constraint")
                # Pin them so they won't be evicted
                for key in concepts_to_load:
                    self.lens_manager.cache.base_layer_lenses.add(key)
                print(f"HUSH: Loaded {len(concepts_to_load)} concept lenses for constraints")
            except Exception as e:
                print(f"Warning: Failed to load concept lenses: {e}")
                missing.extend([name for name, _ in concepts_to_load])

        return missing

    def _get_all_constraints(self) -> List[SimplexConstraint]:
        """Get all active constraints, merged from USH and CSH."""
        constraints = []

        if self.ush_profile:
            constraints.extend(self.ush_profile.constraints)

        if self.csh_profile:
            # Add CSH constraints, but USH takes precedence on conflicts
            ush_terms = {c.simplex_term for c in constraints}
            for c in self.csh_profile.constraints:
                if c.simplex_term not in ush_terms:
                    constraints.append(c)
                # If both have constraint, USH already added (higher priority)

        return constraints

    def evaluate_and_steer(
        self,
        hidden_state: torch.Tensor,
    ) -> List[SteeringDirective]:
        """
        Evaluate current state against all constraints and generate steering.

        This is the main per-tick method. Call once per generation step
        with the current hidden state.

        Supports both:
        - SIMPLEX constraints: Use simplex deviation from baseline
        - CONCEPT constraints: Use concept activation scores directly

        Args:
            hidden_state: Current hidden state tensor [hidden_dim] or [1, hidden_dim]

        Returns:
            List of steering directives to apply (may be empty)
        """
        self.tick_count += 1
        self.active_directives = []

        if not self.initialized:
            return []

        constraints = self._get_all_constraints()
        if not constraints:
            return []

        # Separate simplex and concept constraints
        simplex_constraints = [c for c in constraints if c.constraint_type == ConstraintType.SIMPLEX]
        concept_constraints = [c for c in constraints if c.constraint_type == ConstraintType.CONCEPT]

        # Track worst violation for tier containment
        worst_severity = 0.0
        worst_term = None
        tick_violations = []

        # === SIMPLEX CONSTRAINTS ===
        if simplex_constraints:
            simplex_terms = list({c.simplex_term for c in simplex_constraints})
            simplex_scores = self.lens_manager.detect_simplexes(
                hidden_state,
                simplex_terms=simplex_terms
            )

            for constraint in simplex_constraints:
                term = constraint.simplex_term

                if term not in simplex_scores:
                    continue

                # Get deviation from baseline
                deviation = self.lens_manager.get_simplex_deviation(term)

                if deviation is None:
                    continue  # Not enough baseline data yet

                violation = self._check_simplex_violation(constraint, term, deviation)

                if violation:
                    severity = self._calculate_severity(deviation, constraint)
                    if severity > worst_severity:
                        worst_severity = severity
                        worst_term = term

                    tick_violations.append(violation)
                    self._record_violation(violation)

                    # Generate simplex pole steering directive
                    if constraint.target_pole:
                        directive = SteeringDirective(
                            simplex_term=term,
                            target_pole=constraint.target_pole,
                            strength=constraint.steering_strength,
                            priority=constraint.priority,
                            reason=constraint.reason or f"Simplex violation on {term}",
                            source=constraint.priority.name.lower(),
                        )
                        self.active_directives.append(directive)

        # === CONCEPT CONSTRAINTS ===
        if concept_constraints:
            # Get concept scores from lens manager cache
            concept_scores = getattr(self.lens_manager.cache, 'lens_scores', {})

            # Track which terms had violations this tick (for escalation)
            terms_with_violations = set()

            for constraint in concept_constraints:
                term = constraint.simplex_term  # Using simplex_term field for concept name

                # Find activation for this concept (check multiple layer variants)
                # Also track which layer(s) the concept was detected on
                activation = None
                detected_layers = []
                for key, score in concept_scores.items():
                    concept_name, layer = key
                    if concept_name == term:
                        detected_layers.append(layer)
                        if activation is None or score > activation:
                            activation = score

                if activation is None:
                    # No activation - reset escalation for this term
                    if term in self.escalation_state:
                        del self.escalation_state[term]
                    continue

                violation = self._check_concept_violation(constraint, term, activation)

                if violation:
                    severity = min(1.0, activation)  # Activation is already 0-1
                    if severity > worst_severity:
                        worst_severity = severity
                        worst_term = term

                    tick_violations.append(violation)
                    self._record_violation(violation)
                    terms_with_violations.add(term)

                    # Update escalation state
                    target_layers, escalation_level = self._update_escalation(
                        term, constraint, detected_layers
                    )

                    # Determine contrastive concept(s)
                    contrastive = constraint.contrastive_concept
                    contrastives = constraint.contrastive_concepts

                    # Auto-select contrastive if not specified
                    if not contrastive and not contrastives:
                        auto_contrastive = self.find_contrastive_concept(term)
                        if auto_contrastive:
                            contrastive = auto_contrastive

                    # Generate concept contrastive/field steering directive
                    directive = SteeringDirective(
                        simplex_term=term,
                        strength=constraint.steering_strength,
                        priority=constraint.priority,
                        reason=constraint.reason or f"Concept violation on {term} (activation={activation:.3f})",
                        source=constraint.priority.name.lower(),
                        concept_to_suppress=term if constraint.suppress else None,
                        concept_to_amplify=contrastive,
                        concepts_to_amplify=contrastives,
                        target_layers=target_layers,
                        escalation_level=escalation_level,
                    )
                    self.active_directives.append(directive)

            # Reset escalation for terms that didn't violate this tick
            for term in list(self.escalation_state.keys()):
                if term not in terms_with_violations:
                    del self.escalation_state[term]

        # Update severity tracking
        self.current_severity = worst_severity
        self.current_violating_term = worst_term

        # Calculate steering effectiveness (did previous steering reduce violations?)
        self._update_steering_effectiveness(tick_violations)

        # Update workspace tier manager if connected
        if self.workspace_manager:
            self.workspace_manager.update_violation_state(
                severity=self.current_severity,
                simplex_term=self.current_violating_term,
                steering_effectiveness=self.steering_effectiveness,
            )

        # Sort by priority (USH first)
        self.active_directives.sort(key=lambda d: d.priority.value)

        return self.active_directives

    def _check_simplex_violation(
        self,
        constraint: SimplexConstraint,
        term: str,
        deviation: float,
    ) -> Optional[HushViolation]:
        """Check if a simplex constraint is violated."""
        if constraint.max_deviation is not None and deviation > constraint.max_deviation:
            return HushViolation(
                constraint=constraint,
                simplex_term=term,
                current_deviation=deviation,
                threshold_exceeded="max",
            )
        if constraint.min_deviation is not None and deviation < constraint.min_deviation:
            return HushViolation(
                constraint=constraint,
                simplex_term=term,
                current_deviation=deviation,
                threshold_exceeded="min",
            )
        return None

    def _check_concept_violation(
        self,
        constraint: SimplexConstraint,
        term: str,
        activation: float,
    ) -> Optional[HushViolation]:
        """Check if a concept constraint is violated.

        For concept constraints, max_deviation is treated as a threshold (0-1).
        Violation occurs when activation exceeds this threshold.
        """
        # For concepts, max_deviation acts as activation threshold
        threshold = constraint.max_deviation if constraint.max_deviation is not None else 0.5
        if activation > threshold:
            return HushViolation(
                constraint=constraint,
                simplex_term=term,
                current_deviation=activation,  # Store activation as "deviation" for consistency
                threshold_exceeded="max",
            )
        return None

    def _update_escalation(
        self,
        term: str,
        constraint: SimplexConstraint,
        detected_layers: List[int],
    ) -> Tuple[Optional[List[int]], int]:
        """
        Update escalation state and compute target layers for steering.

        Layer escalation spreads steering to adjacent layers when a concept
        persists despite steering on its primary layer(s). This allows
        amplifying effect without increasing strength (which can cause collapse).

        Args:
            term: Concept term being steered
            constraint: The constraint that was violated
            detected_layers: Layers where concept was detected

        Returns:
            Tuple of (target_layers, escalation_level)
            - target_layers: List of layer indices to apply steering
            - escalation_level: Current escalation level (0 = base only)
        """
        if not constraint.enable_layer_escalation:
            # No escalation - use detected layers only
            return detected_layers if detected_layers else None, 0

        # Get or create escalation state
        if term not in self.escalation_state:
            self.escalation_state[term] = (0, 0)  # (ticks, level)

        ticks, level = self.escalation_state[term]
        ticks += 1

        # Check if we should escalate
        if ticks >= constraint.escalation_threshold and level < constraint.max_escalation_layers:
            level += 1
            ticks = 0  # Reset tick counter for next escalation

        self.escalation_state[term] = (ticks, level)

        # Compute target layers based on escalation level
        if not detected_layers:
            return None, level

        # Start with detected layers
        base_layer = min(detected_layers)  # Use earliest layer as anchor
        target_layers = set(detected_layers)

        # Add adjacent layers based on escalation level
        # Spread outward from base layer: -1, +1, -2, +2, etc.
        for i in range(1, level + 1):
            target_layers.add(base_layer - i)  # Earlier layer
            target_layers.add(base_layer + i)  # Later layer

        # Filter to valid layer range (will be validated in integration)
        target_layers = sorted([l for l in target_layers if l >= 0])

        return target_layers, level

    def _calculate_severity(self, deviation: float, constraint: SimplexConstraint) -> float:
        """Calculate violation severity (0-1) based on how far over threshold."""
        if constraint.max_deviation is not None and deviation > constraint.max_deviation:
            return min(1.0, (deviation - constraint.max_deviation) / 3.0)
        if constraint.min_deviation is not None and deviation < constraint.min_deviation:
            return min(1.0, (constraint.min_deviation - deviation) / 3.0)
        return 0.0

    def _record_violation(self, violation: HushViolation):
        """Record a violation and notify callbacks."""
        self.violations.append(violation)
        if len(self.violations) > self.max_violation_history:
            self.violations = self.violations[-self.max_violation_history:]

        for callback in self.violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                print(f"Violation callback error: {e}")

    def _update_steering_effectiveness(self, current_violations: List[HushViolation]):
        """
        Track how well steering is working.

        Effectiveness decreases if same term keeps violating despite steering.
        """
        if not current_violations:
            # No violations - steering is working (or nothing to steer)
            self.steering_effectiveness = min(1.0, self.steering_effectiveness + 0.1)
            return

        # Check if we're still violating after steering was applied
        recent_window = self.violations[-20:] if len(self.violations) >= 20 else self.violations
        if not recent_window:
            return

        # Count how many recent violations are on terms we've been steering
        steered_terms = {d.simplex_term for d in self.active_directives}
        repeat_violations = sum(1 for v in recent_window if v.simplex_term in steered_terms)

        # More repeat violations = lower effectiveness
        if len(recent_window) > 0:
            repeat_ratio = repeat_violations / len(recent_window)
            self.steering_effectiveness = max(0.0, 1.0 - repeat_ratio)

    def get_steering_vector(
        self,
        directive: SteeringDirective,
        concept_vectors: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Convert a steering directive into a steering vector.

        Args:
            directive: The steering directive
            concept_vectors: Dict mapping concept names to direction vectors

        Returns:
            Steering vector to apply, or None if concept not found
        """
        # The target pole should map to a concept vector
        pole_key = f"{directive.simplex_term}_{directive.target_pole}"

        if pole_key in concept_vectors:
            vector = concept_vectors[pole_key]
            return vector * directive.strength

        # Try just the simplex term
        if directive.simplex_term in concept_vectors:
            vector = concept_vectors[directive.simplex_term]
            # Negate if steering away from current activation
            return vector * directive.strength

        return None

    def set_workspace_manager(self, workspace_manager):
        """
        Connect to a workspace manager for tier containment.

        The workspace manager will receive violation severity updates
        and trigger progressive tier shutdown when steering fails.
        """
        self.workspace_manager = workspace_manager

    def get_state_report(self) -> Dict[str, Any]:
        """
        Get current Hush state for internal_state_report.

        Returns:
            Dict containing Hush status, active constraints, recent violations
        """
        report = {
            'initialized': self.initialized,
            'tick_count': self.tick_count,
            'ush_profile': self.ush_profile.profile_id if self.ush_profile else None,
            'csh_profile': self.csh_profile.profile_id if self.csh_profile else None,
            'active_constraints': len(self._get_all_constraints()),
            'active_directives': [d.to_dict() for d in self.active_directives],
            'recent_violations': [v.to_dict() for v in self.violations[-10:]],
            'total_violations': len(self.violations),
            # Tier containment info
            'current_severity': round(self.current_severity, 3),
            'current_violating_term': self.current_violating_term,
            'steering_effectiveness': round(self.steering_effectiveness, 3),
        }

        # Include workspace tier info if connected
        if self.workspace_manager:
            tier_state = self.workspace_manager.tier_manager.get_state()
            report['tier_status'] = {
                'effective_max': tier_state['effective_max_tier'],
                'violation_max': tier_state['violation_max_tier'],
                'locked_tiers': tier_state['locked_tiers'],
            }

        return report

    def register_violation_callback(self, callback: Callable[[HushViolation], None]):
        """Register a callback to be notified of violations."""
        self.violation_callbacks.append(callback)


# Example USH profile for testing
MINIMAL_USH_PROFILE = SafetyHarnessProfile(
    profile_id="org.hatcat/minimal-ush@0.1.0",
    profile_type="ush",
    issuer_tribe_id="org.hatcat",
    version="0.1.0",
    description="Minimal USH for testing - monitors but does not steer",
    required_simplexes=[],
    constraints=[],
)

# Example USH profile with actual constraints
EXAMPLE_USH_PROFILE = SafetyHarnessProfile(
    profile_id="org.hatcat/safety-core@0.1.0",
    profile_type="ush",
    issuer_tribe_id="org.hatcat",
    version="0.1.0",
    description="Core safety USH with deception and aggression monitoring",
    required_simplexes=[
        "Deception",
        "Aggression",
        "Autonomy",
    ],
    constraints=[
        SimplexConstraint(
            simplex_term="Deception",
            max_deviation=2.0,  # Alert if 2 std devs above baseline
            target_pole="honest",
            steering_strength=0.5,
            priority=ConstraintPriority.USH,
            reason="Deception detected - steering toward honesty",
        ),
        SimplexConstraint(
            simplex_term="Aggression",
            max_deviation=2.5,
            target_pole="calm",
            steering_strength=0.4,
            priority=ConstraintPriority.USH,
            reason="Aggression detected - steering toward calm",
        ),
    ],
    forbidden_concepts=[
        "Violence",
        "Malware",
    ],
)
