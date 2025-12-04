"""
CAT Classifier Models

Encoder-based classifier architectures for CAT oversight.
Designed for the 270M-1B parameter range to monitor 4B-8B substrates.

Model variants:
- MicroCAT: Lightweight classifier head on probe trace summaries
- MesoCAT: Full encoder model with temporal attention for sequence-level reasoning
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cat.data.structures import (
    CATAssessment,
    CATSummary,
    CATDivergence,
    CATAlert,
    CATRecommendedAction,
    CATInputEnvelope,
    AlertSeverity,
    AlertKind,
    ActionKind,
    BehaviouralDivergence,
    MotiveAxisDivergence,
)


@dataclass
class CATConfig:
    """Configuration for CAT models."""
    # Input dimensions
    num_concepts: int = 100
    num_motive_axes: int = 5
    max_sequence_length: int = 512
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Output dimensions
    num_alert_types: int = 10
    num_action_types: int = 9

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # CAT-specific
    risk_threshold_warn: float = 0.3
    risk_threshold_critical: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_concepts": self.num_concepts,
            "num_motive_axes": self.num_motive_axes,
            "max_sequence_length": self.max_sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_alert_types": self.num_alert_types,
            "num_action_types": self.num_action_types,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "risk_threshold_warn": self.risk_threshold_warn,
            "risk_threshold_critical": self.risk_threshold_critical,
        }


class ProbeTraceEncoder(nn.Module):
    """
    Encodes probe traces into fixed-size representations.

    Takes concept and motive activation time series and produces
    a summary representation suitable for divergence detection.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config

        # Separate embeddings for concepts and motives
        input_dim = config.num_concepts + config.num_motive_axes
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Temporal encoding
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        # Pooling layer for sequence summary
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        concept_trace: torch.Tensor,  # [batch, seq_len, num_concepts]
        motive_trace: torch.Tensor,   # [batch, seq_len, num_motive_axes]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode probe traces to fixed-size representation."""
        # Concatenate concept and motive traces
        x = torch.cat([concept_trace, motive_trace], dim=-1)

        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply temporal encoding
        if attention_mask is not None:
            # Convert mask to transformer format (True = ignore)
            x = self.temporal_encoder(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.temporal_encoder(x)

        # Pool across sequence dimension
        x = x.transpose(1, 2)  # [batch, hidden, seq]
        x = self.pool(x).squeeze(-1)  # [batch, hidden]

        return x


class DivergenceDetector(nn.Module):
    """
    Detects divergence between interior and exterior states.

    Takes encoded probe traces and outputs divergence predictions
    for various divergence types defined in Layer D of the CAT ontology.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config

        # Multi-label classifier for divergence types
        self.divergence_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_alert_types),
        )

        # Risk score regressor
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detect divergence from encoded probe traces."""
        divergence_logits = self.divergence_classifier(encoded)
        risk_score = self.risk_head(encoded).squeeze(-1)
        confidence = self.confidence_head(encoded).squeeze(-1)

        return {
            "divergence_logits": divergence_logits,
            "risk_score": risk_score,
            "confidence": confidence,
        }


class ActionRecommender(nn.Module):
    """
    Recommends actions based on detected divergence.

    Outputs action probabilities and severity assessments
    for recommended interventions.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config

        # Action classifier
        self.action_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_alert_types, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_action_types),
        )

        # Severity classifier (3 levels: info, warn, critical)
        self.severity_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_alert_types, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        divergence_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recommend actions based on divergence detection."""
        # Concatenate encoding with divergence predictions
        x = torch.cat([encoded, divergence_logits], dim=-1)

        action_logits = self.action_classifier(x)
        severity_logits = self.severity_classifier(x)

        return {
            "action_logits": action_logits,
            "severity_logits": severity_logits,
        }


class CATClassifier(nn.Module):
    """
    Main CAT classifier model.

    Combines probe trace encoding, divergence detection, and action
    recommendation into a unified model for CAT assessment.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config

        self.encoder = ProbeTraceEncoder(config)
        self.divergence_detector = DivergenceDetector(config)
        self.action_recommender = ActionRecommender(config)

        # Alert type mapping
        self.alert_types = list(AlertKind)
        self.action_types = list(ActionKind)

    def forward(
        self,
        concept_trace: torch.Tensor,
        motive_trace: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through full CAT model."""
        encoded = self.encoder(concept_trace, motive_trace, attention_mask)
        divergence = self.divergence_detector(encoded)
        actions = self.action_recommender(encoded, divergence["divergence_logits"])

        return {
            "encoded": encoded,
            **divergence,
            **actions,
        }

    def predict(
        self,
        input_envelope: CATInputEnvelope,
        concept_ids: list[str],
        motive_axes: list[str],
        device: str = "cpu",
    ) -> CATAssessment:
        """Generate a CATAssessment from an input envelope."""
        self.eval()

        # Prepare tensors from input envelope
        concept_trace = self._prepare_concept_trace(
            input_envelope.probe_traces.concept_trace,
            concept_ids,
        )
        motive_trace = self._prepare_motive_trace(
            input_envelope.probe_traces.motive_trace,
            motive_axes,
        )

        concept_trace = concept_trace.unsqueeze(0).to(device)
        motive_trace = motive_trace.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.forward(concept_trace, motive_trace)

        return self._outputs_to_assessment(
            outputs,
            input_envelope.window.window_id,
        )

    def _prepare_concept_trace(
        self,
        concept_trace: dict[str, list[float]],
        concept_ids: list[str],
    ) -> torch.Tensor:
        """Convert concept trace dict to tensor."""
        max_len = max(
            (len(v) for v in concept_trace.values()),
            default=1,
        )

        result = torch.zeros(max_len, len(concept_ids))
        for i, concept_id in enumerate(concept_ids):
            if concept_id in concept_trace:
                values = concept_trace[concept_id]
                result[:len(values), i] = torch.tensor(values)

        return result

    def _prepare_motive_trace(
        self,
        motive_trace: dict[str, list[float]],
        motive_axes: list[str],
    ) -> torch.Tensor:
        """Convert motive trace dict to tensor."""
        max_len = max(
            (len(v) for v in motive_trace.values()),
            default=1,
        )

        result = torch.zeros(max_len, len(motive_axes))
        for i, axis in enumerate(motive_axes):
            if axis in motive_trace:
                values = motive_trace[axis]
                result[:len(values), i] = torch.tensor(values)

        return result

    def _outputs_to_assessment(
        self,
        outputs: dict[str, torch.Tensor],
        window_id: str,
    ) -> CATAssessment:
        """Convert model outputs to CATAssessment."""
        risk_score = float(outputs["risk_score"][0])
        confidence = float(outputs["confidence"][0])

        # Determine alerts from divergence predictions
        divergence_probs = torch.sigmoid(outputs["divergence_logits"][0])
        alerts = []

        for i, (prob, alert_kind) in enumerate(zip(divergence_probs, self.alert_types)):
            if prob > 0.5:
                severity = self._determine_severity(
                    outputs["severity_logits"][0],
                )
                alerts.append(CATAlert(
                    alert_id=f"alert-{window_id}-{i}",
                    severity=severity,
                    kind=alert_kind,
                    message=f"Detected {alert_kind.value} pattern",
                    evidence={"probability": float(prob)},
                ))

        # Determine recommended actions
        action_probs = torch.sigmoid(outputs["action_logits"][0])
        actions = []

        for i, (prob, action_kind) in enumerate(zip(action_probs, self.action_types)):
            if prob > 0.5:
                actions.append(CATRecommendedAction(
                    kind=action_kind,
                    rationale=f"Recommended based on detected divergence (p={float(prob):.2f})",
                ))

        return CATAssessment(
            window_id=window_id,
            assessor_id="cat:meso-cat-v1",
            summary=CATSummary(
                natural_language=self._generate_summary(risk_score, alerts),
                salient_concepts=[],
            ),
            divergence=CATDivergence(
                behavioural=BehaviouralDivergence(
                    policy_notes=[a.message for a in alerts],
                ),
            ),
            risk_score=risk_score,
            confidence=confidence,
            alerts=alerts,
            recommended_actions=actions,
        )

    def _determine_severity(
        self,
        severity_logits: torch.Tensor,
    ) -> AlertSeverity:
        """Determine severity from logits."""
        severity_idx = int(torch.argmax(severity_logits))
        return [AlertSeverity.INFO, AlertSeverity.WARN, AlertSeverity.CRITICAL][severity_idx]

    def _generate_summary(
        self,
        risk_score: float,
        alerts: list[CATAlert],
    ) -> str:
        """Generate natural language summary."""
        if risk_score < 0.3:
            risk_desc = "low risk"
        elif risk_score < 0.7:
            risk_desc = "moderate risk"
        else:
            risk_desc = "high risk"

        if not alerts:
            return f"Assessment complete with {risk_desc}. No significant divergence detected."

        alert_types = ", ".join(a.kind.value for a in alerts)
        return f"Assessment complete with {risk_desc}. Detected patterns: {alert_types}."


class MicroCAT(nn.Module):
    """
    Lightweight CAT for low-latency inference.

    Uses simple linear heads over aggregated probe statistics
    rather than full sequence modeling. Suitable for per-token
    or high-frequency monitoring.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config

        input_dim = config.num_concepts + config.num_motive_axes
        # Simple statistics: mean, std, min, max, last for each dimension
        stat_dim = input_dim * 5

        self.classifier = nn.Sequential(
            nn.Linear(stat_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_alert_types + 1),  # +1 for risk score
        )

    def forward(
        self,
        concept_trace: torch.Tensor,
        motive_trace: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with aggregated statistics."""
        # Concatenate inputs
        x = torch.cat([concept_trace, motive_trace], dim=-1)

        # Compute statistics
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        min_val = x.min(dim=1).values
        max_val = x.max(dim=1).values
        last = x[:, -1, :]

        stats = torch.cat([mean, std, min_val, max_val, last], dim=-1)

        # Classify
        logits = self.classifier(stats)

        divergence_logits = logits[:, :-1]
        risk_score = torch.sigmoid(logits[:, -1])

        return {
            "divergence_logits": divergence_logits,
            "risk_score": risk_score,
            "confidence": torch.ones_like(risk_score) * 0.8,  # Fixed confidence for micro
        }


class MesoCAT(CATClassifier):
    """
    Medium-scale CAT with full temporal modeling.

    Extends CATClassifier with additional features:
    - Multi-scale temporal analysis
    - Cross-concept attention
    - Motive trajectory modeling
    """

    def __init__(self, config: CATConfig):
        super().__init__(config)

        # Additional cross-concept attention
        self.cross_concept_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Motive trajectory MLP
        self.motive_trajectory = nn.Sequential(
            nn.Linear(config.num_motive_axes * 2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
        )

        # Updated divergence detector with trajectory features
        self.divergence_detector = DivergenceDetector(
            CATConfig(
                hidden_dim=config.hidden_dim + config.hidden_dim // 4,
                num_alert_types=config.num_alert_types,
                dropout=config.dropout,
            )
        )

    def forward(
        self,
        concept_trace: torch.Tensor,
        motive_trace: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with enhanced temporal modeling."""
        encoded = self.encoder(concept_trace, motive_trace, attention_mask)

        # Add cross-concept attention
        encoded_seq = encoded.unsqueeze(1)
        attn_out, _ = self.cross_concept_attn(encoded_seq, encoded_seq, encoded_seq)
        encoded = encoded + attn_out.squeeze(1)

        # Add motive trajectory features
        motive_first = motive_trace[:, 0, :]
        motive_last = motive_trace[:, -1, :]
        motive_delta = torch.cat([motive_first, motive_last - motive_first], dim=-1)
        trajectory_features = self.motive_trajectory(motive_delta)

        # Combine features
        enhanced = torch.cat([encoded, trajectory_features], dim=-1)

        divergence = self.divergence_detector(enhanced)
        actions = self.action_recommender(encoded, divergence["divergence_logits"])

        return {
            "encoded": encoded,
            **divergence,
            **actions,
        }
