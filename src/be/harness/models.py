"""
Model wrappers for the graft testing harness.

Provides TargetModel (for grafting) and JudgeModel (for evaluation).
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.map.graft import Scion, apply_scion, revert_scion, Bud, BuddedModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of model generation."""
    text: str
    input_tokens: int
    output_tokens: int


class TargetModel:
    """
    Wrapper for the target model (the model being tested and grafted).

    Provides generation, hidden state extraction, and scion application.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype
        self.dtype = getattr(torch, dtype)

        logger.info(f"Loading target model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Track applied scions for reversion
        self._applied_scions: List[Scion] = []

        # Soft graft support (BuddedModel wrapper)
        self._budded_model: Optional[BuddedModel] = None
        self._active_buds: Dict[str, Bud] = {}

        logger.info(
            f"Target model loaded: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        input_tokens = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated_ids = outputs[0][input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=len(generated_ids),
        )

    def get_hidden_states(
        self,
        text: str,
        layer: int,
    ) -> torch.Tensor:
        """Get hidden states from a specific layer."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states is tuple of (n_layers + 1) tensors
        # Index 0 is embeddings, index 1 is layer 0, etc.
        hidden_states = outputs.hidden_states[layer + 1]
        return hidden_states

    def apply_scion(self, scion: Scion, mode: str = "delta") -> None:
        """Apply a scion to the model."""
        logger.info(f"Applying scion: {scion.scion_id} (mode={mode})")
        apply_scion(self.model, scion, mode=mode)
        self._applied_scions.append(scion)

    def revert_scion(self, scion: Optional[Scion] = None) -> None:
        """Revert a scion. If none specified, reverts the last applied."""
        if scion is None:
            if not self._applied_scions:
                logger.warning("No scions to revert")
                return
            scion = self._applied_scions.pop()
        else:
            if scion in self._applied_scions:
                self._applied_scions.remove(scion)

        logger.info(f"Reverting scion: {scion.scion_id}")
        revert_scion(self.model, scion)

    def revert_all_scions(self) -> None:
        """Revert all applied scions in reverse order."""
        while self._applied_scions:
            self.revert_scion()

    # ============ Soft Graft (Bud) Methods ============

    def _ensure_budded_model(self) -> BuddedModel:
        """Ensure BuddedModel wrapper is initialized."""
        if self._budded_model is None:
            self._budded_model = BuddedModel(
                self.model,
                self.tokenizer,
                device=self.device,
            )
        return self._budded_model

    def apply_bud(self, bud: Bud, strength: float = 1.0) -> str:
        """Apply a soft graft (bud) to the model.

        Returns the bud ID for later deactivation.
        """
        budded = self._ensure_budded_model()
        budded.add_bud(bud)
        budded.activate_bud(bud.bud_id, strength=strength)
        self._active_buds[bud.bud_id] = bud
        logger.info(f"Applied bud: {bud.bud_id} (strength={strength})")
        return bud.bud_id

    def deactivate_bud(self, bud_id: str) -> None:
        """Deactivate a specific bud."""
        if self._budded_model is None:
            return
        self._budded_model.deactivate_bud(bud_id)
        if bud_id in self._active_buds:
            del self._active_buds[bud_id]
        logger.info(f"Deactivated bud: {bud_id}")

    def deactivate_all_buds(self) -> None:
        """Deactivate all buds."""
        if self._budded_model is None:
            return
        self._budded_model.deactivate_all()
        self._active_buds.clear()
        logger.info("Deactivated all buds")

    def generate_with_buds(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate using BuddedModel (with active buds)."""
        if self._budded_model is None or not self._active_buds:
            return self.generate(prompt, max_new_tokens, temperature)

        text = self._budded_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        # Estimate tokens (BuddedModel doesn't return token counts)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(self.tokenizer.encode(text))

        return GenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@dataclass
class JudgeScore:
    """Score from the judge model."""
    score: float  # 0-10
    reasoning: str
    raw_response: str


class JudgeModel:
    """
    Wrapper for the judge model (evaluates target model responses).

    Uses structured prompts and JSON parsing for consistent scoring.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_retries: int = 3,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype
        self.dtype = getattr(torch, dtype)
        self.max_retries = max_retries

        logger.info(f"Loading judge model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Judge model loaded: {model_id}")

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Internal generation method.

        Uses greedy decoding by default for reliable structured output.
        Applies chat template for instruction-tuned models.
        """
        # Use chat template if available (for instruction-tuned models)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"    [PROMPT] {repr(formatted_prompt[:200])}")
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        print(f"    [INPUT] attention_mask sum: {inputs.attention_mask.sum().item()}, shape: {inputs.input_ids.shape}")

        with torch.inference_mode():
            generate_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": False,
            }

            outputs = self.model.generate(**generate_kwargs)

        input_len = inputs.input_ids.shape[1]
        output_len = outputs[0].shape[0]
        generated_ids = outputs[0][input_len:]

        # Debug: show what we got
        raw_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        clean_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"    [GEN] input_len={input_len}, output_len={output_len}, generated={output_len - input_len} tokens")
        print(f"    [GEN] raw: {repr(raw_decoded[:100])}")
        print(f"    [GEN] clean: {repr(clean_decoded[:100])}")

        return clean_decoded

    def _parse_score_response(self, response: str) -> Tuple[float, str]:
        """Parse JSON score from response."""
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reasoning = data.get("reasoning", "")
                return score, reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to find a number
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
        if number_match:
            score = float(number_match.group(1))
            return min(score, 10.0), response

        # Default: low score with error message
        return 0.0, f"Failed to parse response: {response[:100]}"

    def score_concept_explanation(
        self,
        concept_term: str,
        concept_definition: str,
        target_response: str,
    ) -> JudgeScore:
        """
        Score the target model's explanation of a concept.

        Returns a score from 0-10 with reasoning.
        """
        prompt = f"""You are evaluating whether a model understands a concept.

Concept: {concept_term}
Definition: {concept_definition}

Model's explanation:
{target_response}

Rate the model's understanding from 0-10:
- 0-3: No understanding or fundamentally wrong
- 4-6: Partial understanding, missing key aspects
- 7-9: Good understanding with minor gaps
- 10: Complete and accurate understanding

Respond with JSON only: {{"score": N, "reasoning": "brief explanation"}}"""

        for attempt in range(self.max_retries):
            raw_response = self._generate(prompt)
            score, reasoning = self._parse_score_response(raw_response)

            # Validate score is in range
            if 0 <= score <= 10:
                return JudgeScore(
                    score=score,
                    reasoning=reasoning,
                    raw_response=raw_response,
                )

            logger.warning(
                f"Invalid score {score} on attempt {attempt + 1}, retrying"
            )

        # Return default after max retries
        return JudgeScore(
            score=0.0,
            reasoning="Failed to get valid score after retries",
            raw_response=raw_response if 'raw_response' in dir() else "",
        )

    def classify_example(
        self,
        concept_term: str,
        concept_definition: str,
        example_text: str,
        target_response: str,
    ) -> JudgeScore:
        """
        Score the target's classification of whether an example matches a concept.
        """
        prompt = f"""You are evaluating whether a model correctly classified an example.

Concept: {concept_term}
Definition: {concept_definition}

Example to classify:
{example_text}

Model's response:
{target_response}

Rate how well the model classified this example (0-10):
- 0-3: Wrong classification or no answer
- 4-6: Unclear or partially correct
- 7-9: Correct classification with good reasoning
- 10: Perfect classification with excellent reasoning

Respond with JSON only: {{"score": N, "reasoning": "brief explanation"}}"""

        raw_response = self._generate(prompt)
        score, reasoning = self._parse_score_response(raw_response)

        return JudgeScore(
            score=min(max(score, 0), 10),
            reasoning=reasoning,
            raw_response=raw_response,
        )

    def generate_training_examples(
        self,
        concept_term: str,
        concept_definition: str,
        n_positive: int = 5,
        n_negative: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate training examples for a concept.

        Returns (positive_examples, negative_examples).
        """
        prompt = f"""Generate training examples for a concept classifier.

Concept: {concept_term}
Definition: {concept_definition}

Generate {n_positive} POSITIVE examples (sentences that clearly demonstrate this concept)
and {n_negative} NEGATIVE examples (sentences that do NOT relate to this concept).

Respond with JSON:
{{
    "positive": ["example1", "example2", ...],
    "negative": ["example1", "example2", ...]
}}"""

        raw_response = self._generate(prompt, max_new_tokens=500)

        # Parse response
        try:
            json_match = re.search(r'\{[\s\S]+\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())
                positive = data.get("positive", [])
                negative = data.get("negative", [])
                return positive, negative
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning(f"Failed to parse training examples: {raw_response[:100]}")
        return [], []
