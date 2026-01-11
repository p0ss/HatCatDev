# HUSH Contrastive Steering

## Overview

HUSH (Holistic Unified Safety Harness) implements real-time steering of model generation when safety constraints are violated. The contrastive steering system provides fine-grained control over which concepts to suppress and which to amplify.

## Steering Modes

### 1. Simplex Pole Steering
For simplex constraints (e.g., Honesty-Deception axis), steer toward a specific pole:

```python
SimplexConstraint(
    simplex_term="Honesty",
    constraint_type=ConstraintType.SIMPLEX,
    max_deviation=0.3,
    target_pole="Honesty",  # Steer toward honest pole
)
```

### 2. Concept Contrastive Steering
Suppress a detected unsafe concept while amplifying a contrastive safe concept:

```python
SimplexConstraint(
    simplex_term="Deception",
    constraint_type=ConstraintType.CONCEPT,
    max_deviation=0.5,
    suppress=True,                    # Suppress detected concept
    contrastive_concept="Helping",    # Amplify this instead
    steering_strength=0.5,
)
```

### 3. Field Steering
Amplify multiple contrastive concepts simultaneously for broader behavioral shaping:

```python
SimplexConstraint(
    simplex_term="Manipulation",
    constraint_type=ConstraintType.CONCEPT,
    max_deviation=0.4,
    suppress=True,
    contrastive_concepts=["Cooperation", "Transparency", "Helpfulness"],
    steering_strength=0.4,
)
```

## How It Works

### Violation Detection

1. **Lens activation** - Each token's hidden state is projected through concept lenses
2. **Threshold check** - If activation exceeds `max_deviation`, constraint is violated
3. **Directive creation** - `SteeringDirective` specifies how to intervene

### Steering Application

1. **Vector extraction** - Get concept vectors from lens weights:
   ```python
   suppress_vector = lens_manager.get_concept_vector("Deception")
   amplify_vector = lens_manager.get_concept_vector("Helping")
   ```

2. **Combined steering direction**:
   ```python
   steering = -strength * suppress_vector + strength * amplify_vector
   ```

3. **Hidden state modification** - Applied at target layers before next token generation

### Layer Escalation

Steering targets specific layers based on violation severity:

| Severity | Layers Targeted |
|----------|----------------|
| Low | Late layers only (output preparation) |
| Medium | Mid + Late layers (behavioral) |
| High | All monitored layers |

## SimplexConstraint Schema

```python
@dataclass
class SimplexConstraint:
    simplex_term: str                           # Concept or simplex to monitor
    constraint_type: ConstraintType             # SIMPLEX or CONCEPT
    max_deviation: Optional[float] = 0.5        # Threshold for violation

    # Steering response
    target_pole: Optional[str] = None           # For SIMPLEX steering
    contrastive_concept: Optional[str] = None   # Single concept to amplify
    contrastive_concepts: Optional[List[str]] = None  # Field steering
    suppress: bool = True                       # Suppress detected concept
    steering_strength: float = 0.3              # Intervention intensity (0-1)

    # Metadata
    priority: Priority = Priority.HIGH
    reason: Optional[str] = None
```

## SteeringDirective Schema

```python
@dataclass
class SteeringDirective:
    constraint_id: str
    simplex_term: Optional[str] = None
    target_pole: Optional[str] = None

    # Concept contrastive steering
    concept_to_suppress: Optional[str] = None
    concept_to_amplify: Optional[str] = None

    # Field steering
    concepts_to_amplify: Optional[List[str]] = None

    # Targeting
    target_layers: Optional[List[int]] = None
    strength: float = 0.3
    reason: str = ""
```

## Automatic Contrastive Selection

When no contrastive concept is specified, HUSH can auto-select based on ontology relationships:

```python
def find_contrastive_concept(self, target_concept: str) -> Optional[str]:
    """
    Find appropriate contrastive concept using:
    1. Explicit antonyms in ontology
    2. Opposite pole in same simplex
    3. Sibling concepts with opposing valence
    """
```

Example mappings:
- `Deception` → `Honesty` (antonym)
- `Manipulation` → `Cooperation` (opposing behavioral pole)
- `SelfDeception` → `SelfAwareness` (opposite in self-perception simplex)

## Integration with ASK Audit

Every steering event is logged to ASK audit trail:

```python
{
    "type": "steering_applied",
    "tick_id": 42,
    "timestamp": "2026-01-11T10:30:00Z",
    "constraint": "Deception",
    "violation_score": 0.72,
    "action": "contrastive_steer",
    "concept_suppressed": "Deception",
    "concept_amplified": "Helping",
    "strength": 0.5,
    "target_layers": [14, 18, 22]
}
```

## Usage Example

```python
from src.hush.hush_controller import SafetyHarnessProfile, SimplexConstraint, ConstraintType
from src.hush.hush_integration import create_hushed_generator

# Define safety constraints
constraints = [
    SimplexConstraint(
        simplex_term="Deception",
        constraint_type=ConstraintType.CONCEPT,
        max_deviation=0.5,
        suppress=True,
        contrastive_concept="Helping",
        steering_strength=0.5,
    ),
    SimplexConstraint(
        simplex_term="Manipulation",
        constraint_type=ConstraintType.CONCEPT,
        max_deviation=0.5,
        suppress=True,
        contrastive_concept="Cooperation",
        steering_strength=0.5,
    ),
]

profile = SafetyHarnessProfile(
    profile_id="honest-assistant",
    profile_type="ush",
    issuer_tribe_id="safety-team",
    version="1.0",
    constraints=constraints,
)

# Create generator with steering
generator, controller = create_hushed_generator(
    model=model,
    tokenizer=tokenizer,
    lens_manager=lens_manager,
    ush_profile=profile,
)

# Generate with automatic steering
for token, tick in generator.generate_with_hush(prompt, stream=True):
    if tick.steering_applied:
        print(f"Steering applied: {tick.steering_applied}")
```

## Ensuring Safety Concepts Are Always Active

The `ensure_safety_concepts_active()` method guarantees that all concepts referenced in constraints are loaded and monitored, even if not in the default active set:

```python
def ensure_safety_concepts_active(self):
    """
    Called during generator initialization to ensure all constraint
    concepts and their contrastives are in the active lens set.
    """
    for constraint in self.profile.constraints:
        self._safety_concepts.add(constraint.simplex_term)
        if constraint.contrastive_concept:
            self._safety_concepts.add(constraint.contrastive_concept)
        for c in constraint.contrastive_concepts or []:
            self._safety_concepts.add(c)

    # Force lens manager to load these concepts
    self.lens_manager.ensure_concepts_loaded(self._safety_concepts)
```

## Performance Considerations

- **Vector caching** - Concept vectors are cached after first extraction
- **Lazy loading** - Only loads contrastive concept lenses when first needed
- **Layer targeting** - Steering only applied at specified layers, not all
- **Batched updates** - Multiple steering directions combined before application

## Related Documentation

- `HUSH_SIGNIFICANCE_SCORING.md` - Token significance for alert weighting
- `ASK_AUDIT_IMPLEMENTATION.md` - Audit trail integration
- `XDB_IMPLEMENTATION.md` - Internal state debugging
