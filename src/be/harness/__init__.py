"""
Graft Testing Harness

Test harness for validating that grafting enables models to learn new concepts.
Uses a judge model to evaluate a target model's concept knowledge before and after grafting.

Graft infrastructure (Scion, Bud) lives in MAP since it operates on clefts/lenses.
This module provides the evaluation harness for testing graft effectiveness.

Usage:
    from src.be.harness import GraftTester, HarnessConfig

    config = HarnessConfig(
        target_model_id="allenai/OLMo-1B",
        judge_model_id="google/gemma-3-4b-it",
    )

    tester = GraftTester(config)
    report = tester.run_full_test(max_concepts=50)
"""

from .config import HarnessConfig
from .models import TargetModel, JudgeModel
from .evaluator import ConceptEvaluator, EvaluationResult
from .reporter import HarnessReporter, HarnessReport
from .calibration import JudgeCalibrator, CalibrationReport
from .meld_designer import MeldDesigner
from .graft_tester import GraftTester, GraftTestReport

__all__ = [
    # Config
    'HarnessConfig',
    # Models
    'TargetModel',
    'JudgeModel',
    # Evaluation
    'ConceptEvaluator',
    'EvaluationResult',
    # Reporting
    'HarnessReporter',
    'HarnessReport',
    # Calibration
    'JudgeCalibrator',
    'CalibrationReport',
    # Meld
    'MeldDesigner',
    # Graft testing
    'GraftTester',
    'GraftTestReport',
]
