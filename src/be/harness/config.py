"""
Configuration for the graft testing harness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class HarnessConfig:
    """Configuration for the graft testing harness."""

    # Model configuration
    target_model_id: str = "allenai/OLMo-1B"
    judge_model_id: str = "google/gemma-3-4b-it"

    # Data paths
    concept_pack_path: Path = field(default_factory=lambda: Path("concept_packs/first-light"))
    melds_path: Path = field(default_factory=lambda: Path("melds/applied"))
    output_dir: Path = field(default_factory=lambda: Path("results/harness"))

    # Hardware
    device: str = "cuda"
    target_dtype: str = "float16"
    judge_dtype: str = "float16"

    # Evaluation settings
    batch_size: int = 4
    max_concepts: Optional[int] = None
    knowledge_threshold: float = 6.0  # Score threshold for "knows" vs "doesn't know"

    # Grafting settings
    layers_to_graft: List[int] = field(default_factory=lambda: [8, 10, 12])
    scion_epochs: int = 3
    scion_learning_rate: float = 1e-4
    scion_batch_size: int = 4
    graft_mode: str = "soft"  # "soft" (bud) or "hard" (scion)

    # Generation settings
    max_response_tokens: int = 100
    temperature: float = 0.7

    # Judge settings
    judge_max_retries: int = 3

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.concept_pack_path, str):
            self.concept_pack_path = Path(self.concept_pack_path)
        if isinstance(self.melds_path, str):
            self.melds_path = Path(self.melds_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "target_model_id": self.target_model_id,
            "judge_model_id": self.judge_model_id,
            "concept_pack_path": str(self.concept_pack_path),
            "melds_path": str(self.melds_path),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "target_dtype": self.target_dtype,
            "judge_dtype": self.judge_dtype,
            "batch_size": self.batch_size,
            "max_concepts": self.max_concepts,
            "knowledge_threshold": self.knowledge_threshold,
            "layers_to_graft": self.layers_to_graft,
            "scion_epochs": self.scion_epochs,
            "scion_learning_rate": self.scion_learning_rate,
            "scion_batch_size": self.scion_batch_size,
            "graft_mode": self.graft_mode,
            "max_response_tokens": self.max_response_tokens,
            "temperature": self.temperature,
            "judge_max_retries": self.judge_max_retries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessConfig":
        """Create config from dictionary."""
        return cls(**data)
