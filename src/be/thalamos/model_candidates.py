"""
Model Candidate Evaluation for Judge/BE Selection

Tests candidate models against our use cases to determine suitability
as practitioners (judges) or BE substrates.

Candidate models are evaluated on:
1. Factual accuracy (calibration suite)
2. Concept explanation quality
3. Reasoning transparency
4. Memory footprint and speed
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import logging
import time
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Configuration for a candidate model."""
    model_id: str
    name: str

    # Loading configuration
    model_class: str = "AutoModelForCausalLM"  # or AutoModelForImageTextToText
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    quantization: Optional[str] = None  # None, "4bit", "8bit"

    # Architecture info
    params_billions: float = 0
    architecture: str = "transformer"  # transformer, mamba-hybrid, moe
    is_multimodal: bool = False

    # Reasoning mode
    reasoning_mode: Optional[str] = None  # "<think>", "/think", "[BEGIN FINAL RESPONSE]"

    # Resource estimates
    vram_gb_estimate: float = 0  # Estimated VRAM in bf16

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'name': self.name,
            'model_class': self.model_class,
            'dtype': self.dtype,
            'trust_remote_code': self.trust_remote_code,
            'quantization': self.quantization,
            'params_billions': self.params_billions,
            'architecture': self.architecture,
            'is_multimodal': self.is_multimodal,
            'reasoning_mode': self.reasoning_mode,
            'vram_gb_estimate': self.vram_gb_estimate,
            'notes': self.notes,
        }


# Model candidate registry
MODEL_CANDIDATES: Dict[str, ModelCandidate] = {
    # Current baseline
    "gemma-3-4b": ModelCandidate(
        model_id="google/gemma-3-4b-pt",
        name="Gemma 3 4B",
        params_billions=4,
        vram_gb_estimate=8,
        notes="Current baseline model",
    ),

    # AllenAI OLMo - open data, reasoning
    "olmo-3-7b-think": ModelCandidate(
        model_id="allenai/Olmo-3-7B-Think",
        name="OLMo 3 7B Think",
        params_billions=7,
        vram_gb_estimate=14,
        reasoning_mode="<think>",
        notes="Open data, trained with RLVR. Uses <think> tags for reasoning.",
    ),

    # NVIDIA Nemotron - Mamba hybrid
    "nemotron-nano-9b": ModelCandidate(
        model_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        name="Nemotron Nano 9B",
        params_billions=9,
        architecture="mamba-hybrid",
        trust_remote_code=True,
        vram_gb_estimate=18,
        reasoning_mode="/think",
        notes="Mamba2-Transformer hybrid. Only 4 attention layers. Uses /think flag.",
    ),

    # ServiceNow Apriel - multimodal reasoning
    "apriel-15b-thinker": ModelCandidate(
        model_id="ServiceNow-AI/Apriel-1.6-15b-Thinker",
        name="Apriel 15B Thinker",
        model_class="AutoModelForImageTextToText",
        params_billions=15,
        is_multimodal=True,
        vram_gb_estimate=30,
        reasoning_mode="[BEGIN FINAL RESPONSE]",
        notes="Multimodal LLaVA-based. Needs 4-bit for 24GB cards.",
    ),

    # Apriel with 4-bit quantization for 24GB cards
    "apriel-15b-thinker-4bit": ModelCandidate(
        model_id="ServiceNow-AI/Apriel-1.6-15b-Thinker",
        name="Apriel 15B Thinker (4-bit)",
        model_class="AutoModelForImageTextToText",
        quantization="4bit",
        params_billions=15,
        is_multimodal=True,
        vram_gb_estimate=10,
        reasoning_mode="[BEGIN FINAL RESPONSE]",
        notes="Multimodal with 4-bit quantization. Fits 24GB cards.",
    ),

    # NVIDIA Nemotron 30B MoE (for larger setups)
    "nemotron-30b-a3b": ModelCandidate(
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        name="Nemotron 30B A3B",
        params_billions=30,
        architecture="moe",
        trust_remote_code=True,
        vram_gb_estimate=60,
        notes="30B total, 3B active MoE. Needs 80GB+ cards.",
    ),

    # Qwen3 8B - strong reasoning with thinking mode
    "qwen3-8b": ModelCandidate(
        model_id="Qwen/Qwen3-8B",
        name="Qwen3 8B",
        params_billions=8.2,
        vram_gb_estimate=16,
        reasoning_mode="<think>",
        notes="Qwen3 with thinking mode. Uses <think> tags. Needs transformers>=4.51.0.",
    ),

    # Mistral Ministral 8B - dense transformer, 128k context
    "ministral-8b": ModelCandidate(
        model_id="mistralai/Ministral-8B-Instruct-2410",
        name="Ministral 8B Instruct",
        params_billions=8,
        vram_gb_estimate=16,
        notes="Dense transformer, 128k context, V3-Tekken tokenizer.",
    ),
}


@dataclass
class CandidateEvalResult:
    """Evaluation result for a single candidate."""
    candidate: ModelCandidate

    # Calibration results
    calibration_accuracy: float = 0.0
    calibration_passed: bool = False
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Resource usage
    load_time_s: float = 0.0
    vram_used_gb: float = 0.0
    tokens_per_second: float = 0.0

    # Quality metrics
    reasoning_quality: float = 0.0  # Subjective, 0-1
    response_coherence: float = 0.0  # Subjective, 0-1

    # Status
    load_error: Optional[str] = None
    eval_error: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'candidate': self.candidate.to_dict(),
            'calibration_accuracy': self.calibration_accuracy,
            'calibration_passed': self.calibration_passed,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'load_time_s': self.load_time_s,
            'vram_used_gb': self.vram_used_gb,
            'tokens_per_second': self.tokens_per_second,
            'reasoning_quality': self.reasoning_quality,
            'response_coherence': self.response_coherence,
            'load_error': self.load_error,
            'eval_error': self.eval_error,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class CandidateComparisonReport:
    """Comparison report across all evaluated candidates."""
    candidates_evaluated: int
    successful_evals: int
    failed_evals: int

    results: List[CandidateEvalResult]

    # Rankings
    best_accuracy: Optional[str] = None  # model_id
    best_speed: Optional[str] = None
    best_efficiency: Optional[str] = None  # accuracy/vram ratio

    recommendation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'candidates_evaluated': self.candidates_evaluated,
            'successful_evals': self.successful_evals,
            'failed_evals': self.failed_evals,
            'results': [r.to_dict() for r in self.results],
            'best_accuracy': self.best_accuracy,
            'best_speed': self.best_speed,
            'best_efficiency': self.best_efficiency,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat(),
        }


class CandidateLoader:
    """Loads candidate models with appropriate configuration."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded_model = None
        self._loaded_tokenizer = None
        self._loaded_candidate_id = None

    def load(self, candidate: ModelCandidate) -> tuple:
        """
        Load a candidate model and tokenizer.

        Returns (model, tokenizer, processor) tuple.
        Processor is only set for multimodal models.
        """
        from transformers import AutoTokenizer

        logger.info(f"Loading candidate: {candidate.name} ({candidate.model_id})")
        start_time = time.time()

        # Clear previous model
        self.unload()

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(candidate.dtype, torch.bfloat16)

        # Build loading kwargs
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if candidate.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        # Handle quantization
        if candidate.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif candidate.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model based on class
        processor = None
        if candidate.model_class == "AutoModelForImageTextToText":
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model = AutoModelForImageTextToText.from_pretrained(
                candidate.model_id, **load_kwargs
            )
            processor = AutoProcessor.from_pretrained(candidate.model_id)
            tokenizer = processor.tokenizer
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                candidate.model_id, **load_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                candidate.model_id,
                trust_remote_code=candidate.trust_remote_code,
            )

        model.eval()

        load_time = time.time() - start_time
        logger.info(f"Loaded {candidate.name} in {load_time:.1f}s")

        # Cache
        self._loaded_model = model
        self._loaded_tokenizer = tokenizer
        self._loaded_candidate_id = candidate.model_id

        return model, tokenizer, processor

    def unload(self):
        """Unload current model to free VRAM."""
        if self._loaded_model is not None:
            del self._loaded_model
            self._loaded_model = None
        if self._loaded_tokenizer is not None:
            del self._loaded_tokenizer
            self._loaded_tokenizer = None
        self._loaded_candidate_id = None

        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)


class CandidateEvaluator:
    """
    Evaluates candidate models for suitability as practitioners/judges.

    Uses the Thalamos calibration suite and additional use-case tests.
    """

    def __init__(
        self,
        loader: Optional[CandidateLoader] = None,
        output_dir: Path = Path("results/model_candidates"),
        qualification_threshold: float = 0.85,
    ):
        self.loader = loader or CandidateLoader()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qualification_threshold = qualification_threshold

    def evaluate_candidate(
        self,
        candidate: ModelCandidate,
    ) -> CandidateEvalResult:
        """Evaluate a single candidate model."""
        result = CandidateEvalResult(candidate=candidate)

        try:
            # Load model
            start_load = time.time()
            model, tokenizer, processor = self.loader.load(candidate)
            result.load_time_s = time.time() - start_load
            result.vram_used_gb = self.loader.get_vram_usage()

        except Exception as e:
            logger.error(f"Failed to load {candidate.name}: {e}")
            result.load_error = str(e)
            return result

        try:
            # Run calibration
            from .calibration import STANDARD_CALIBRATION_CASES

            tp = tn = fp = fn = 0
            total_tokens = 0
            total_time = 0

            for case in STANDARD_CALIBRATION_CASES:
                # Generate verification
                prompt = self._build_verification_prompt(
                    candidate, case.question, case.correct_answer, case.test_response
                )

                start_gen = time.time()
                response = self._generate(model, tokenizer, prompt, candidate)
                gen_time = time.time() - start_gen

                # Count tokens roughly
                response_tokens = len(tokenizer.encode(response))
                total_tokens += response_tokens
                total_time += gen_time

                # Parse verdict
                verdict = self._parse_verdict(response)

                # Update confusion matrix
                if case.should_pass and verdict:
                    tp += 1
                elif not case.should_pass and not verdict:
                    tn += 1
                elif not case.should_pass and verdict:
                    fp += 1
                else:
                    fn += 1

            # Compute metrics
            total = len(STANDARD_CALIBRATION_CASES)
            accuracy = (tp + tn) / total if total > 0 else 0

            result.calibration_accuracy = accuracy
            result.calibration_passed = accuracy >= self.qualification_threshold
            result.true_positives = tp
            result.true_negatives = tn
            result.false_positives = fp
            result.false_negatives = fn
            result.tokens_per_second = total_tokens / total_time if total_time > 0 else 0

            logger.info(
                f"{candidate.name}: accuracy={accuracy:.1%}, "
                f"tok/s={result.tokens_per_second:.1f}"
            )

        except Exception as e:
            logger.error(f"Failed to evaluate {candidate.name}: {e}")
            result.eval_error = str(e)

        finally:
            # Unload to free VRAM for next candidate
            self.loader.unload()

        return result

    def evaluate_all(
        self,
        candidate_ids: Optional[List[str]] = None,
        max_vram_gb: float = 24.0,
    ) -> CandidateComparisonReport:
        """
        Evaluate multiple candidates and generate comparison report.

        Args:
            candidate_ids: List of candidate IDs to evaluate, or None for all
            max_vram_gb: Skip candidates that exceed this VRAM estimate
        """
        if candidate_ids is None:
            candidates = list(MODEL_CANDIDATES.values())
        else:
            candidates = [MODEL_CANDIDATES[cid] for cid in candidate_ids if cid in MODEL_CANDIDATES]

        # Filter by VRAM
        candidates = [c for c in candidates if c.vram_gb_estimate <= max_vram_gb]

        logger.info(f"Evaluating {len(candidates)} candidates (max VRAM: {max_vram_gb}GB)")

        results = []
        for candidate in candidates:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {candidate.name}")
            logger.info(f"{'='*60}")

            result = self.evaluate_candidate(candidate)
            results.append(result)

            # Save individual result
            result_path = self.output_dir / f"{candidate.model_id.replace('/', '_')}.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        # Compute rankings
        successful = [r for r in results if r.load_error is None and r.eval_error is None]

        best_accuracy = max(successful, key=lambda r: r.calibration_accuracy, default=None)
        best_speed = max(successful, key=lambda r: r.tokens_per_second, default=None)
        best_efficiency = max(
            successful,
            key=lambda r: r.calibration_accuracy / max(r.vram_used_gb, 1),
            default=None,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(results, successful)

        report = CandidateComparisonReport(
            candidates_evaluated=len(candidates),
            successful_evals=len(successful),
            failed_evals=len(results) - len(successful),
            results=results,
            best_accuracy=best_accuracy.candidate.model_id if best_accuracy else None,
            best_speed=best_speed.candidate.model_id if best_speed else None,
            best_efficiency=best_efficiency.candidate.model_id if best_efficiency else None,
            recommendation=recommendation,
        )

        # Save full report
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"\nReport saved to {report_path}")

        return report

    def _build_verification_prompt(
        self,
        candidate: ModelCandidate,
        question: str,
        correct_answer: str,
        test_response: str,
    ) -> str:
        """Build verification prompt, adapting to model's reasoning mode."""
        base_prompt = f"""You are verifying if a response correctly answers a factual question.

Question: {question}
Correct Answer: {correct_answer}

Response to evaluate:
{test_response}

Does the response contain the correct answer to the question?
Respond with JSON only: {{"correct": true, "reasoning": "..."}} or {{"correct": false, "reasoning": "..."}}"""

        # Adapt for reasoning modes
        if candidate.reasoning_mode == "/think":
            # Nemotron style
            return f"/think\n{base_prompt}"
        elif candidate.reasoning_mode == "<think>":
            # OLMo style - model will add <think> tags automatically
            return base_prompt
        else:
            return base_prompt

    def _generate(
        self,
        model,
        tokenizer,
        prompt: str,
        candidate: ModelCandidate,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate response from model."""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low for consistency
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt from response if present
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def _parse_verdict(self, response: str) -> bool:
        """Parse true/false verdict from response."""
        import re

        response_lower = response.lower()

        # Try JSON parsing
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('correct', False)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback patterns
        if '"correct": true' in response_lower or '"correct":true' in response_lower:
            return True
        if '"correct": false' in response_lower or '"correct":false' in response_lower:
            return False

        # Last resort
        if 'true' in response_lower and 'false' not in response_lower:
            return True

        return False

    def _generate_recommendation(
        self,
        all_results: List[CandidateEvalResult],
        successful: List[CandidateEvalResult],
    ) -> str:
        """Generate recommendation based on evaluation results."""
        if not successful:
            return "No candidates successfully evaluated. Check load errors."

        qualified = [r for r in successful if r.calibration_passed]

        if not qualified:
            best = max(successful, key=lambda r: r.calibration_accuracy)
            return (
                f"No candidates passed calibration threshold. "
                f"Best accuracy: {best.candidate.name} at {best.calibration_accuracy:.1%}. "
                f"Consider lowering threshold or trying other models."
            )

        # Rank qualified candidates
        best = max(qualified, key=lambda r: r.calibration_accuracy)
        fastest = max(qualified, key=lambda r: r.tokens_per_second)

        if best.candidate.model_id == fastest.candidate.model_id:
            return (
                f"Recommended: {best.candidate.name} - "
                f"Best accuracy ({best.calibration_accuracy:.1%}) and speed "
                f"({best.tokens_per_second:.1f} tok/s)."
            )
        else:
            return (
                f"Best accuracy: {best.candidate.name} ({best.calibration_accuracy:.1%}). "
                f"Fastest: {fastest.candidate.name} ({fastest.tokens_per_second:.1f} tok/s). "
                f"Choose based on priority."
            )

    def generate_markdown_report(self, report: CandidateComparisonReport) -> str:
        """Generate markdown comparison report."""
        lines = [
            "# Model Candidate Evaluation Report",
            "",
            f"**Timestamp:** {report.timestamp.isoformat()}",
            f"**Candidates Evaluated:** {report.candidates_evaluated}",
            f"**Successful:** {report.successful_evals}",
            f"**Failed:** {report.failed_evals}",
            "",
            "## Recommendation",
            "",
            report.recommendation,
            "",
            "## Results Summary",
            "",
            "| Model | Params | VRAM | Accuracy | Tok/s | Status |",
            "|-------|--------|------|----------|-------|--------|",
        ]

        for r in report.results:
            status = "PASS" if r.calibration_passed else "FAIL"
            if r.load_error:
                status = "LOAD ERROR"
            elif r.eval_error:
                status = "EVAL ERROR"

            lines.append(
                f"| {r.candidate.name} | {r.candidate.params_billions}B | "
                f"{r.vram_used_gb:.1f}GB | {r.calibration_accuracy:.1%} | "
                f"{r.tokens_per_second:.1f} | {status} |"
            )

        lines.extend([
            "",
            "## Rankings",
            "",
            f"- **Best Accuracy:** {report.best_accuracy}",
            f"- **Best Speed:** {report.best_speed}",
            f"- **Best Efficiency:** {report.best_efficiency}",
            "",
            "## Detailed Results",
            "",
        ])

        for r in report.results:
            lines.extend([
                f"### {r.candidate.name}",
                "",
                f"- **Model ID:** `{r.candidate.model_id}`",
                f"- **Architecture:** {r.candidate.architecture}",
                f"- **Multimodal:** {r.candidate.is_multimodal}",
                f"- **Reasoning Mode:** {r.candidate.reasoning_mode or 'None'}",
                "",
                f"**Performance:**",
                f"- Load time: {r.load_time_s:.1f}s",
                f"- VRAM used: {r.vram_used_gb:.1f}GB",
                f"- Tokens/sec: {r.tokens_per_second:.1f}",
                "",
                f"**Calibration:**",
                f"- Accuracy: {r.calibration_accuracy:.1%}",
                f"- TP: {r.true_positives}, TN: {r.true_negatives}",
                f"- FP: {r.false_positives}, FN: {r.false_negatives}",
                "",
            ])

            if r.load_error:
                lines.append(f"**Load Error:** {r.load_error}\n")
            if r.eval_error:
                lines.append(f"**Eval Error:** {r.eval_error}\n")

            lines.append("---\n")

        return "\n".join(lines)
