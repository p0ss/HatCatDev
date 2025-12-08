#!/usr/bin/env python3
"""
Lens pack version manifest for diff-based distribution.

The version_manifest.json tracks:
- Which concept pack version each lens was trained at
- When each lens was last trained
- Training quality metrics for each lens
- Which lenses changed between versions (for diff distribution)

Per MAP_MELD_PROTOCOL.md §8:
- Lens packs are rebuilt on any concept pack version change
- Unchanged lenses can persist across versions
- Diff-based distribution only sends changed lenses

Usage:
    from src.data.version_manifest import LensManifest

    manifest = LensManifest.load(lens_pack_dir)
    manifest.update_lens("Deception", "4.1.0", metrics={"f1": 0.92})
    manifest.save()

    # Get diff between versions
    diff = manifest.compute_diff("4.0.0", "4.1.0")
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any


@dataclass
class LensEntry:
    """Entry for a single lens in the manifest."""
    concept: str
    trained_at_version: str  # Concept pack version when trained
    trained_timestamp: str  # ISO timestamp
    layer: int
    metrics: Dict[str, float] = field(default_factory=dict)  # f1, precision, recall
    training_samples: int = 0
    lens_file: str = ""  # Relative path to .pt file

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LensEntry':
        return cls(**data)


@dataclass
class VersionSnapshot:
    """Snapshot of lens states at a specific version."""
    version: str
    timestamp: str
    lenses_trained: List[str] = field(default_factory=list)  # New/retrained lenses
    lenses_unchanged: List[str] = field(default_factory=list)  # Persisted from prior
    total_lenses: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionSnapshot':
        return cls(**data)


@dataclass
class VersionDiff:
    """Diff between two version snapshots for distribution."""
    from_version: str
    to_version: str
    added_lenses: List[str] = field(default_factory=list)
    retrained_lenses: List[str] = field(default_factory=list)
    removed_lenses: List[str] = field(default_factory=list)
    unchanged_lenses: List[str] = field(default_factory=list)

    @property
    def lenses_to_download(self) -> List[str]:
        """Lenses that need to be downloaded for upgrade."""
        return self.added_lenses + self.retrained_lenses

    @property
    def download_count(self) -> int:
        return len(self.lenses_to_download)

    @property
    def savings_percent(self) -> float:
        """Percentage of lenses that don't need downloading."""
        total = len(self.unchanged_lenses) + self.download_count + len(self.removed_lenses)
        if total == 0:
            return 0.0
        return (len(self.unchanged_lenses) / total) * 100

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["lenses_to_download"] = self.lenses_to_download
        d["download_count"] = self.download_count
        d["savings_percent"] = round(self.savings_percent, 1)
        return d


class LensManifest:
    """
    Manages version manifest for a lens pack.

    Structure:
    {
        "lens_pack_id": "org.hatcat/apertus-8b__sumo-wordnet-v4",
        "model": "swiss-ai/Apertus-8B-2509",
        "current_version": "4.1.0",
        "created": "2025-11-29T00:00:00Z",
        "updated": "2025-11-30T12:00:00Z",
        "lenses": {
            "Deception": {
                "concept": "Deception",
                "trained_at_version": "4.1.0",
                "trained_timestamp": "2025-11-30T12:00:00Z",
                "layer": 3,
                "metrics": {"f1": 0.92, "precision": 0.90, "recall": 0.94},
                "training_samples": 100,
                "lens_file": "layer3/Deception.pt"
            }
        },
        "version_history": [
            {
                "version": "4.0.0",
                "timestamp": "2025-11-29T00:00:00Z",
                "lenses_trained": ["Entity", "Object", ...],
                "lenses_unchanged": [],
                "total_lenses": 7000
            },
            {
                "version": "4.1.0",
                "timestamp": "2025-11-30T12:00:00Z",
                "lenses_trained": ["Deception", "DeceptionDetector", ...],
                "lenses_unchanged": ["Entity", "Object", ...],
                "total_lenses": 7050
            }
        ]
    }
    """

    def __init__(
        self,
        lens_pack_id: str = "",
        model: str = "",
        source_pack: str = "",
    ):
        self.lens_pack_id = lens_pack_id
        self.model = model
        self.source_pack = source_pack
        self.current_version = "0.0.0"
        self.created = datetime.now().isoformat() + "Z"
        self.updated = self.created
        self.lenses: Dict[str, LensEntry] = {}
        self.version_history: List[VersionSnapshot] = []
        self._path: Optional[Path] = None

    @classmethod
    def load(cls, lens_pack_dir: Path) -> 'LensManifest':
        """Load manifest from lens pack directory."""
        manifest_path = lens_pack_dir / "version_manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)

            manifest = cls(
                lens_pack_id=data.get("lens_pack_id", ""),
                model=data.get("model", ""),
                source_pack=data.get("source_pack", ""),
            )
            manifest.current_version = data.get("current_version", "0.0.0")
            manifest.created = data.get("created", manifest.created)
            manifest.updated = data.get("updated", manifest.updated)

            for name, lens_data in data.get("lenses", {}).items():
                manifest.lenses[name] = LensEntry.from_dict(lens_data)

            for snap_data in data.get("version_history", []):
                manifest.version_history.append(VersionSnapshot.from_dict(snap_data))
        else:
            # Initialize from pack_info.json if available
            pack_info_path = lens_pack_dir / "pack_info.json"
            if pack_info_path.exists():
                with open(pack_info_path) as f:
                    info = json.load(f)
                manifest = cls(
                    lens_pack_id=f"{info.get('source_pack', '')}_{info.get('model', '').replace('/', '_')}",
                    model=info.get("model", ""),
                    source_pack=info.get("source_pack", ""),
                )
                manifest.current_version = info.get("pack_version", "0.0.0")
            else:
                manifest = cls()

        manifest._path = manifest_path
        return manifest

    def save(self, path: Optional[Path] = None):
        """Save manifest to file."""
        save_path = path or self._path
        if not save_path:
            raise ValueError("No path specified for saving manifest")

        self.updated = datetime.now().isoformat() + "Z"

        data = {
            "lens_pack_id": self.lens_pack_id,
            "model": self.model,
            "source_pack": self.source_pack,
            "current_version": self.current_version,
            "created": self.created,
            "updated": self.updated,
            "lenses": {name: p.to_dict() for name, p in self.lenses.items()},
            "version_history": [s.to_dict() for s in self.version_history],
            "summary": {
                "total_lenses": len(self.lenses),
                "versions_tracked": len(self.version_history),
                "layers": sorted(set(p.layer for p in self.lenses.values()))
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_lens(
        self,
        concept: str,
        version: str,
        layer: int,
        metrics: Optional[Dict[str, float]] = None,
        training_samples: int = 0,
        lens_file: str = ""
    ):
        """Update or add a lens entry."""
        self.lenses[concept] = LensEntry(
            concept=concept,
            trained_at_version=version,
            trained_timestamp=datetime.now().isoformat() + "Z",
            layer=layer,
            metrics=metrics or {},
            training_samples=training_samples,
            lens_file=lens_file or f"layer{layer}/{concept}.pt"
        )
        self.current_version = version

    def record_training_run(
        self,
        version: str,
        trained_concepts: List[str],
        all_concepts: Optional[List[str]] = None
    ):
        """Record a training run in version history."""
        trained_set = set(trained_concepts)

        # Determine unchanged lenses
        if all_concepts:
            all_set = set(all_concepts)
            unchanged = sorted(all_set - trained_set)
        else:
            # Infer from existing lenses that weren't retrained
            unchanged = sorted([
                name for name, p in self.lenses.items()
                if name not in trained_set and p.trained_at_version != version
            ])

        snapshot = VersionSnapshot(
            version=version,
            timestamp=datetime.now().isoformat() + "Z",
            lenses_trained=sorted(trained_concepts),
            lenses_unchanged=unchanged,
            total_lenses=len(self.lenses)
        )

        self.version_history.append(snapshot)
        self.current_version = version

    def get_lenses_at_version(self, version: str) -> Dict[str, LensEntry]:
        """Get all lenses that were current at a specific version."""
        result = {}
        for name, lens in self.lenses.items():
            # A lens is "at" a version if it was trained at that version
            # or at any prior version (and not retrained since)
            if self._version_lte(lens.trained_at_version, version):
                result[name] = lens
        return result

    def compute_diff(self, from_version: str, to_version: str) -> VersionDiff:
        """
        Compute the diff between two versions.

        Returns lenses that were added, retrained, removed, or unchanged.
        This enables efficient diff-based distribution.
        """
        from_lenses = self.get_lenses_at_version(from_version)
        to_lenses = self.get_lenses_at_version(to_version)

        from_names = set(from_lenses.keys())
        to_names = set(to_lenses.keys())

        added = sorted(to_names - from_names)
        removed = sorted(from_names - to_names)

        # Check which existing lenses were retrained
        common = from_names & to_names
        retrained = []
        unchanged = []

        for name in sorted(common):
            from_lens = from_lenses[name]
            to_lens = to_lenses[name]

            # If trained_at_version changed, it was retrained
            if from_lens.trained_at_version != to_lens.trained_at_version:
                retrained.append(name)
            else:
                unchanged.append(name)

        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            added_lenses=added,
            retrained_lenses=retrained,
            removed_lenses=removed,
            unchanged_lenses=unchanged
        )

    def get_lenses_needing_training(self, pending_concepts: List[str]) -> Dict[str, str]:
        """
        Given a list of concepts pending training, return which need training.

        Returns dict mapping concept -> reason (new, must_retrain, should_retrain)
        """
        result = {}
        existing = set(self.lenses.keys())

        for concept in pending_concepts:
            if concept not in existing:
                result[concept] = "new"
            else:
                result[concept] = "retrain"

        return result

    def _version_lte(self, v1: str, v2: str) -> bool:
        """Check if version v1 <= v2."""
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]
            return parts1 <= parts2
        except (ValueError, AttributeError):
            return True


def generate_manifest_from_training(
    lens_pack_dir: Path,
    version: str,
    training_results: Dict[str, Dict],
    source_pack: str = "",
    model: str = ""
) -> LensManifest:
    """
    Generate or update manifest after a training run.

    Args:
        lens_pack_dir: Path to lens pack
        version: Concept pack version lenses were trained for
        training_results: Dict mapping concept -> training result
            Expected keys: layer, metrics (f1, precision, recall), n_samples
        source_pack: Source concept pack ID
        model: Model name

    Returns:
        Updated manifest
    """
    manifest = LensManifest.load(lens_pack_dir)

    if source_pack:
        manifest.source_pack = source_pack
    if model:
        manifest.model = model

    trained_concepts = []

    for concept, result in training_results.items():
        manifest.update_lens(
            concept=concept,
            version=version,
            layer=result.get("layer", 0),
            metrics=result.get("metrics", {}),
            training_samples=result.get("n_samples", 0),
            lens_file=result.get("lens_file", f"layer{result.get('layer', 0)}/{concept}.pt")
        )
        trained_concepts.append(concept)

    manifest.record_training_run(
        version=version,
        trained_concepts=trained_concepts,
        all_concepts=list(manifest.lenses.keys())
    )

    return manifest


def print_manifest_summary(manifest: LensManifest):
    """Print a summary of the manifest."""
    print(f"\nLens Pack Manifest: {manifest.lens_pack_id}")
    print(f"  Model: {manifest.model}")
    print(f"  Source: {manifest.source_pack}")
    print(f"  Version: {manifest.current_version}")
    print(f"  Total lenses: {len(manifest.lenses)}")

    if manifest.version_history:
        print(f"\nVersion History ({len(manifest.version_history)} versions):")
        for snap in manifest.version_history[-3:]:  # Last 3
            print(f"  {snap.version}: {len(snap.lenses_trained)} trained, "
                  f"{len(snap.lenses_unchanged)} unchanged")


def print_diff_summary(diff: VersionDiff):
    """Print a summary of a version diff."""
    print(f"\nDiff: {diff.from_version} → {diff.to_version}")
    print(f"  Added: {len(diff.added_lenses)}")
    print(f"  Retrained: {len(diff.retrained_lenses)}")
    print(f"  Removed: {len(diff.removed_lenses)}")
    print(f"  Unchanged: {len(diff.unchanged_lenses)}")
    print(f"  Download: {diff.download_count} lenses")
    print(f"  Savings: {diff.savings_percent:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.version_manifest <lens_pack_dir> [--diff from_ver to_ver]")
        sys.exit(1)

    lens_pack_dir = Path(sys.argv[1])
    manifest = LensManifest.load(lens_pack_dir)
    print_manifest_summary(manifest)

    if len(sys.argv) >= 5 and sys.argv[2] == "--diff":
        from_ver = sys.argv[3]
        to_ver = sys.argv[4]
        diff = manifest.compute_diff(from_ver, to_ver)
        print_diff_summary(diff)
