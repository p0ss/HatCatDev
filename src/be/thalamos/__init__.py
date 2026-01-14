"""
Thalamos: The Examination Room

Cognitive assessment and surgery infrastructure for HAT/CAT systems.

The Thalamos module provides the "examination room" where subjects undergo:
- Assessment (Thalametry) - evaluating concept knowledge
- Surgery (Thalamology) - grafting new cognitive capabilities

Key components:
- ExaminationRoom: Orchestrates BEDFrame + CAT for procedures
- Thalametrist: CAT performing cognitive assessment
- Thalamologist: CAT conducting graft procedures (future)
- Calibration: Practitioner qualification suite

Terminology:
- Thalametrist: A CAT performing cognitive assessment (like an optometrist)
- Thalamologist: A CAT performing cognitive surgery (like an ophthalmologist)
- Thalamos: The examination/operating room itself

The naming connects to the thalamus (brain region analyzed with lenses)
while using medical metaphors that are supportive rather than constraining.
"""

from .room import ExaminationRoom, ExaminationConfig
from .thalametrist import Thalametrist, ConceptAssessment, AssessmentResult
from .calibration import (
    CalibrationSuite,
    CalibrationCase,
    CalibrationResult,
    CalibrationReport,
)
from .protocols import (
    AssessmentProtocol,
    ExplanationProtocol,
    ClassificationProtocol,
)
from .records import (
    ProcedureRecord,
    AssessmentRecord,
    QualificationRecord,
)
from .model_candidates import (
    ModelCandidate,
    MODEL_CANDIDATES,
    CandidateLoader,
    CandidateEvaluator,
    CandidateEvalResult,
    CandidateComparisonReport,
)
from .judge_evaluation import (
    JudgeEvaluator,
    JudgeEvalReport,
    ResponseQuality,
    ConceptJudgingCase,
    ReasoningCase,
    CONCEPT_JUDGING_CASES,
    REASONING_CASES,
)
from .meld_evaluation import (
    MeldExampleLoader,
    MeldJudgeEvaluator,
    MeldEvalReport,
    MeldExample,
    MeldTestResult,
)

__all__ = [
    # Room
    'ExaminationRoom',
    'ExaminationConfig',
    # Thalametrist
    'Thalametrist',
    'ConceptAssessment',
    'AssessmentResult',
    # Calibration
    'CalibrationSuite',
    'CalibrationCase',
    'CalibrationResult',
    'CalibrationReport',
    # Protocols
    'AssessmentProtocol',
    'ExplanationProtocol',
    'ClassificationProtocol',
    # Records
    'ProcedureRecord',
    'AssessmentRecord',
    'QualificationRecord',
    # Model Candidates
    'ModelCandidate',
    'MODEL_CANDIDATES',
    'CandidateLoader',
    'CandidateEvaluator',
    'CandidateEvalResult',
    'CandidateComparisonReport',
    # Judge Evaluation
    'JudgeEvaluator',
    'JudgeEvalReport',
    'ResponseQuality',
    'ConceptJudgingCase',
    'ReasoningCase',
    'CONCEPT_JUDGING_CASES',
    'REASONING_CASES',
    # Meld Evaluation
    'MeldExampleLoader',
    'MeldJudgeEvaluator',
    'MeldEvalReport',
    'MeldExample',
    'MeldTestResult',
]
