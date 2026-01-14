#!/usr/bin/env python3
"""
CLI for the graft testing harness.

Usage:
    # Run full graft test
    python scripts/run_graft_harness.py --mode full

    # Just evaluate baseline (no grafting)
    python scripts/run_graft_harness.py --mode baseline

    # Calibrate judge only
    python scripts/run_graft_harness.py --mode calibrate

    # Test specific concept graft
    python scripts/run_graft_harness.py --mode single --concept Democracy
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.be.harness import (
    GraftTester,
    HarnessConfig,
    JudgeCalibrator,
    JudgeModel,
    TargetModel,
    ConceptEvaluator,
    HarnessReporter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graft Testing Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "baseline", "calibrate", "single"],
        default="full",
        help="Operation mode",
    )

    parser.add_argument(
        "--target",
        default="allenai/OLMo-1B",
        help="Target model ID (default: allenai/OLMo-1B)",
    )

    parser.add_argument(
        "--judge",
        default="google/gemma-3-4b-it",
        help="Judge model ID (default: google/gemma-3-4b-it)",
    )

    parser.add_argument(
        "--pack",
        default="concept_packs/first-light",
        help="Concept pack path (default: concept_packs/first-light)",
    )

    parser.add_argument(
        "--melds",
        default="melds/applied",
        help="Melds directory path (default: melds/applied)",
    )

    parser.add_argument(
        "--output",
        default="results/harness",
        help="Output directory (default: results/harness)",
    )

    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="Maximum concepts to evaluate in baseline",
    )

    parser.add_argument(
        "--max-grafts",
        type=int,
        default=5,
        help="Maximum concepts to graft (default: 5)",
    )

    parser.add_argument(
        "--concept",
        help="Specific concept ID for single mode",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--target-dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Target model dtype (default: float16)",
    )

    parser.add_argument(
        "--judge-dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Judge model dtype (default: float16)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Knowledge threshold (0-10, default: 6.0)",
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[8, 10, 12],
        help="Layers to graft (default: 8 10 12)",
    )

    parser.add_argument(
        "--graft-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Graft mode: 'soft' (bud, reversible) or 'hard' (scion, permanent). Default: soft",
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip judge calibration",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def run_full_test(config: HarnessConfig, args):
    """Run the complete graft testing workflow."""
    tester = GraftTester(config)

    report = tester.run_full_test(
        max_baseline_concepts=args.max_concepts,
        max_graft_concepts=args.max_grafts,
        run_calibration=not args.no_calibration,
        save_results=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GRAFT TEST SUMMARY")
    print("=" * 60)
    print(f"Graft Mode: {config.graft_mode.upper()} ({'bud/reversible' if config.graft_mode == 'soft' else 'scion/permanent'})")
    print(f"Concepts Evaluated: {report.baseline_report['summary']['concepts_evaluated']}")
    print(f"Known: {report.baseline_report['summary']['known']}")
    print(f"Unknown: {report.baseline_report['summary']['unknown']}")
    print("-" * 60)
    print(f"Concepts Grafted: {report.concepts_grafted}")
    print(f"Concepts Learned: {report.concepts_learned}")
    print(f"Learning Rate: {report.learning_rate:.1%}")
    print(f"Mean Improvement: {report.mean_improvement:+.2f}")
    print("=" * 60)

    if report.graft_results:
        print("\nDetailed Results:")
        for r in report.graft_results:
            status = "[LEARNED]" if r["learned"] else "[---]"
            print(
                f"  {status} {r['concept_term']}: "
                f"{r['pre_score']:.1f} -> {r['post_score']:.1f} "
                f"({r['score_improvement']:+.1f})"
            )


def run_baseline(config: HarnessConfig, args):
    """Run baseline evaluation only."""
    logger.info("Running baseline evaluation...")

    # Load models
    target = TargetModel(
        model_id=config.target_model_id,
        device=config.device,
        dtype=config.target_dtype,
    )

    judge = JudgeModel(
        model_id=config.judge_model_id,
        device=config.device,
        dtype=config.judge_dtype,
    )

    # Create evaluator
    evaluator = ConceptEvaluator(
        target=target,
        judge=judge,
        config=config,
    )

    # Evaluate
    results = evaluator.evaluate_all(
        max_concepts=args.max_concepts,
        with_training_data_only=True,
    )

    # Generate report
    reporter = HarnessReporter(config)
    report = reporter.create_report(results)

    # Save
    paths = reporter.save_all(report, prefix="baseline_report")

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Concepts Evaluated: {report.concepts_evaluated}")
    print(f"Known: {report.known_count} ({100*report.known_count/max(report.concepts_evaluated, 1):.1f}%)")
    print(f"Unknown: {report.unknown_count} ({100*report.unknown_count/max(report.concepts_evaluated, 1):.1f}%)")
    print(f"Trainable Unknown: {report.trainable_unknown_count}")
    print("-" * 60)
    print(f"Mean Score: {report.mean_score:.2f}")
    print(f"Median Score: {report.median_score:.2f}")
    print(f"Score Range: {report.min_score:.1f} - {report.max_score:.1f}")
    print("=" * 60)
    print(f"\nReports saved to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")


def run_calibration(config: HarnessConfig, args):
    """Run judge calibration only."""
    logger.info("Running judge calibration...")

    judge = JudgeModel(
        model_id=config.judge_model_id,
        device=config.device,
        dtype=config.judge_dtype,
    )

    calibrator = JudgeCalibrator(judge, accuracy_threshold=0.75)
    report = calibrator.run_calibration()

    # Print results
    print("\n" + "=" * 60)
    print("JUDGE CALIBRATION REPORT")
    print("=" * 60)
    print(f"Judge Model: {config.judge_model_id}")
    print(f"Total Cases: {report.total_cases}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Accuracy: {report.accuracy:.1%}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  True Positives:  {report.true_positives} (correctly accepted correct answers)")
    print(f"  True Negatives:  {report.true_negatives} (correctly rejected wrong answers)")
    print(f"  False Positives: {report.false_positives} (wrongly accepted wrong answers)")
    print(f"  False Negatives: {report.false_negatives} (wrongly rejected correct answers)")
    print("-" * 60)
    print(f"Calibrated: {'YES' if report.is_calibrated else 'NO'}")
    print(f"Recommendation: {report.recommendation}")
    print("=" * 60)

    # Show failed cases
    failed = [r for r in report.results if not r.passed]
    if failed:
        print("\nFailed Cases:")
        for r in failed:
            expected = "CORRECT" if r.case.should_pass else "INCORRECT"
            got = "CORRECT" if r.judge_verdict else "INCORRECT"
            print(f"  - Q: {r.case.question}")
            print(f"    Correct answer: {r.case.correct_answer}")
            print(f"    Test response: {r.case.test_response}")
            print(f"    Expected: {expected}, Judge said: {got}")
            print(f"    Judge's reasoning: {r.reasoning}")
            print()

    # Save report
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_report.json"
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to: {output_path}")


def run_single_graft(config: HarnessConfig, args):
    """Graft a single concept."""
    if not args.concept:
        logger.error("--concept is required for single mode")
        sys.exit(1)

    logger.info(f"Grafting single concept: {args.concept}")

    tester = GraftTester(config)
    result = tester.run_single_graft(
        concept_id=args.concept,
        save_results=True,
    )

    # Print result
    print("\n" + "=" * 60)
    print("SINGLE GRAFT RESULT")
    print("=" * 60)
    print(f"Concept: {result.concept_term}")
    print(f"Pre-graft Score: {result.pre_score:.1f}")
    print(f"Post-graft Score: {result.post_score:.1f}")
    print(f"Improvement: {result.score_improvement:+.1f}")
    print(f"Learned: {'YES' if result.learned else 'NO'}")
    print("-" * 60)
    print(f"Training Data Source: {result.training_data_source}")
    print(f"Examples: {result.n_positive_examples} positive, {result.n_negative_examples} negative")
    if result.scion_id:
        print(f"Scion ID: {result.scion_id}")
    if result.training_loss:
        print(f"Final Training Loss: {result.training_loss:.4f}")
    print("=" * 60)


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = HarnessConfig(
        target_model_id=args.target,
        judge_model_id=args.judge,
        concept_pack_path=Path(args.pack),
        melds_path=Path(args.melds),
        output_dir=Path(args.output),
        device=args.device,
        target_dtype=args.target_dtype,
        judge_dtype=args.judge_dtype,
        knowledge_threshold=args.threshold,
        layers_to_graft=args.layers,
        graft_mode=args.graft_mode,
    )

    # Run requested mode
    if args.mode == "full":
        run_full_test(config, args)
    elif args.mode == "baseline":
        run_baseline(config, args)
    elif args.mode == "calibrate":
        run_calibration(config, args)
    elif args.mode == "single":
        run_single_graft(config, args)


if __name__ == "__main__":
    main()
