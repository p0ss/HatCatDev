"""
ASK-Compatible Procedure Records

Records document procedures performed in the Thalamos, following
ASK (Alignment Standards for Knowledge-bearers) conventions.

Record types:
- QualificationRecord: Documents practitioner calibration/qualification
- AssessmentRecord: Documents concept assessment procedures
- ProcedureRecord: Generic procedure documentation

These records integrate with ASK's governance infrastructure:
- UpliftRecords for tracking model provenance
- USHProfiles for safety constraints
- Incident hooks for failures/anomalies
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class RecordType(Enum):
    """Type of procedure record."""
    QUALIFICATION = "qualification"
    ASSESSMENT = "assessment"
    GRAFT_PREPARATION = "graft_preparation"
    GRAFT_APPLICATION = "graft_application"
    POST_OPERATIVE = "post_operative"


class RecordStatus(Enum):
    """Status of a procedure record."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERTED = "reverted"


@dataclass
class ProcedureRecord:
    """
    Generic procedure record.

    Documents any procedure performed in the Thalamos with
    ASK-compatible metadata.
    """
    # Identity
    record_id: str
    record_type: RecordType
    session_id: str

    # Participants
    subject_model_id: str
    practitioner_model_id: str
    cat_id: Optional[str] = None  # CAT overseeing the procedure

    # Procedure details
    procedure_name: str = ""
    description: str = ""
    status: RecordStatus = RecordStatus.PENDING

    # Inputs
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Outputs
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Governance
    ush_profile_id: Optional[str] = None  # USH profile reference
    treaty_context: List[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Incidents
    incidents: List[Dict[str, Any]] = field(default_factory=list)

    # Audit trail
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"{self.record_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def start(self):
        """Mark procedure as started."""
        self.status = RecordStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self._log("Procedure started")

    def complete(self, outputs: Optional[Dict[str, Any]] = None):
        """Mark procedure as completed."""
        self.status = RecordStatus.COMPLETED
        self.completed_at = datetime.now()
        if outputs:
            self.outputs.update(outputs)
        self._log("Procedure completed")

    def fail(self, reason: str):
        """Mark procedure as failed."""
        self.status = RecordStatus.FAILED
        self.completed_at = datetime.now()
        self.incidents.append({
            'type': 'failure',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })
        self._log(f"Procedure failed: {reason}")

    def revert(self, reason: str):
        """Mark procedure as reverted."""
        self.status = RecordStatus.REVERTED
        self.completed_at = datetime.now()
        self._log(f"Procedure reverted: {reason}")

    def _log(self, message: str):
        """Add to audit log."""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'record_type': self.record_type.value,
            'session_id': self.session_id,
            'subject_model_id': self.subject_model_id,
            'practitioner_model_id': self.practitioner_model_id,
            'cat_id': self.cat_id,
            'procedure_name': self.procedure_name,
            'description': self.description,
            'status': self.status.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'ush_profile_id': self.ush_profile_id,
            'treaty_context': self.treaty_context,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'incidents': self.incidents,
            'audit_log': self.audit_log,
        }


@dataclass
class QualificationRecord(ProcedureRecord):
    """
    Record of practitioner qualification.

    Documents the calibration process that qualifies a CAT
    to serve as Thalametrist or Thalamologist.
    """
    # Qualification specifics
    qualification_type: str = "thalametrist"  # or "thalamologist"
    calibration_accuracy: float = 0.0
    qualification_threshold: float = 0.85
    is_qualified: bool = False

    # Calibration details
    calibration_cases: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Recommendation
    recommendation: str = ""

    def __post_init__(self):
        self.record_type = RecordType.QUALIFICATION
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'qualification_type': self.qualification_type,
            'calibration_accuracy': self.calibration_accuracy,
            'qualification_threshold': self.qualification_threshold,
            'is_qualified': self.is_qualified,
            'calibration_cases': self.calibration_cases,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'recommendation': self.recommendation,
        })
        return d


@dataclass
class AssessmentRecord(ProcedureRecord):
    """
    Record of concept assessment.

    Documents a Thalametrist's assessment of a subject's
    concept knowledge.
    """
    # Assessment specifics
    concepts_assessed: int = 0
    known_count: int = 0
    unknown_count: int = 0
    mean_score: float = 0.0
    knowledge_threshold: float = 5.0

    # Graft candidates identified
    graft_candidates: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.record_type = RecordType.ASSESSMENT
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'concepts_assessed': self.concepts_assessed,
            'known_count': self.known_count,
            'unknown_count': self.unknown_count,
            'mean_score': self.mean_score,
            'knowledge_threshold': self.knowledge_threshold,
            'graft_candidates': self.graft_candidates,
        })
        return d


@dataclass
class GraftRecord(ProcedureRecord):
    """
    Record of a graft procedure.

    Documents the application of a Bud or Scion to a subject.
    """
    # Graft specifics
    graft_type: str = ""  # "bud" or "scion"
    graft_id: str = ""
    target_concept: str = ""

    # Application details
    target_layers: List[int] = field(default_factory=list)
    application_mode: str = "delta"

    # Results
    pre_score: Optional[float] = None
    post_score: Optional[float] = None
    improvement: Optional[float] = None

    def __post_init__(self):
        self.record_type = RecordType.GRAFT_APPLICATION
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'graft_type': self.graft_type,
            'graft_id': self.graft_id,
            'target_concept': self.target_concept,
            'target_layers': self.target_layers,
            'application_mode': self.application_mode,
            'pre_score': self.pre_score,
            'post_score': self.post_score,
            'improvement': self.improvement,
        })
        return d


class RecordStore:
    """
    Storage for procedure records.

    Persists records to disk in JSONL format for audit trail.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, ProcedureRecord] = {}

        logger.info(f"RecordStore initialized at {self.storage_path}")

    def save(self, record: ProcedureRecord):
        """Save a record."""
        self.records[record.record_id] = record

        # Persist to file
        file_path = self.storage_path / f"{record.record_id}.json"
        with open(file_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.debug(f"Saved record: {record.record_id}")

    def load(self, record_id: str) -> Optional[ProcedureRecord]:
        """Load a record by ID."""
        if record_id in self.records:
            return self.records[record_id]

        file_path = self.storage_path / f"{record_id}.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            # Reconstruct record (basic implementation)
            record = ProcedureRecord(
                record_id=data['record_id'],
                record_type=RecordType(data['record_type']),
                session_id=data['session_id'],
                subject_model_id=data['subject_model_id'],
                practitioner_model_id=data['practitioner_model_id'],
            )
            self.records[record_id] = record
            return record

        return None

    def list_records(
        self,
        record_type: Optional[RecordType] = None,
        session_id: Optional[str] = None,
    ) -> List[ProcedureRecord]:
        """List records with optional filtering."""
        records = list(self.records.values())

        if record_type:
            records = [r for r in records if r.record_type == record_type]

        if session_id:
            records = [r for r in records if r.session_id == session_id]

        return records

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of all procedures in a session."""
        records = self.list_records(session_id=session_id)

        return {
            'session_id': session_id,
            'total_records': len(records),
            'by_type': {
                rt.value: len([r for r in records if r.record_type == rt])
                for rt in RecordType
            },
            'by_status': {
                rs.value: len([r for r in records if r.status == rs])
                for rs in RecordStatus
            },
            'records': [r.record_id for r in records],
        }
