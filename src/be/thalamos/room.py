"""
ExaminationRoom - The Thalamos Orchestrator

The ExaminationRoom is where subjects undergo cognitive assessment and surgery.
It orchestrates:
- A BEDFrame for the subject (full instrumentation)
- A CAT (Thalametrist/Thalamologist) for evaluation
- Recording to XDB and audit log
- Procedure protocols and records
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
import logging
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diegesis import BEDFrame, BEDConfig, ExperienceTick, TickType

logger = logging.getLogger(__name__)


@dataclass
class ExaminationConfig:
    """Configuration for the examination room."""

    # Subject model
    subject_model_id: str = "google/gemma-3-4b-it"
    subject_device: str = "cuda"
    subject_dtype: str = "bfloat16"

    # Practitioner model (CAT)
    practitioner_model_id: Optional[str] = None  # None = use subject model
    practitioner_device: str = "cuda"
    practitioner_dtype: str = "bfloat16"

    # Lens pack for the subject
    lens_pack_path: Optional[Path] = None

    # Output and storage
    output_dir: Path = Path("results/thalamos")
    xdb_storage_path: Optional[Path] = None
    audit_storage_path: Optional[Path] = None

    # Procedure settings
    max_response_tokens: int = 256
    assessment_temperature: float = 0.0  # Greedy for reproducibility

    # Qualification thresholds
    min_calibration_accuracy: float = 0.85


class ExaminationRoom:
    """
    The Thalamos - examination room for cognitive assessment and surgery.

    Orchestrates:
    - Subject in BEDFrame (full lens instrumentation)
    - Practitioner CAT (Thalametrist or Thalamologist)
    - Procedure protocols
    - Record keeping
    """

    def __init__(self, config: ExaminationConfig):
        self.config = config
        self.session_id = f"exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Subject setup
        self.subject_model = None
        self.subject_tokenizer = None
        self.bedframe: Optional[BEDFrame] = None

        # Practitioner setup
        self.practitioner_model = None
        self.practitioner_tokenizer = None
        self.practitioner_qualified = False

        # Session state
        self.is_open = False
        self.procedures_run: List[Dict[str, Any]] = []

        logger.info(f"ExaminationRoom created: session={self.session_id}")

    # =========================================================================
    # Room Setup
    # =========================================================================

    def open(self):
        """
        Open the examination room.

        Loads models and sets up BEDFrame for the subject.
        """
        if self.is_open:
            logger.warning("Room already open")
            return

        logger.info("Opening examination room...")

        # Load subject model
        self._load_subject_model()

        # Set up BEDFrame for subject
        self._setup_bedframe()

        # Load practitioner model
        self._load_practitioner_model()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_open = True
        logger.info("Examination room open")

    def close(self):
        """Close the examination room and clean up resources."""
        if not self.is_open:
            return

        logger.info("Closing examination room...")

        # Close BEDFrame
        if self.bedframe:
            self.bedframe.close()

        # Clear models
        if self.subject_model is not None:
            del self.subject_model
            self.subject_model = None

        if self.practitioner_model is not None:
            del self.practitioner_model
            self.practitioner_model = None

        torch.cuda.empty_cache()

        self.is_open = False
        logger.info("Examination room closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _load_subject_model(self):
        """Load the subject model."""
        logger.info(f"Loading subject model: {self.config.subject_model_id}")

        dtype = getattr(torch, self.config.subject_dtype)

        self.subject_model = AutoModelForCausalLM.from_pretrained(
            self.config.subject_model_id,
            torch_dtype=dtype,
            device_map=self.config.subject_device,
            trust_remote_code=True,
        )
        self.subject_model.eval()

        self.subject_tokenizer = AutoTokenizer.from_pretrained(
            self.config.subject_model_id,
            trust_remote_code=True,
        )
        if self.subject_tokenizer.pad_token is None:
            self.subject_tokenizer.pad_token = self.subject_tokenizer.eos_token

        logger.info(f"Subject model loaded: {self.subject_model.config.num_hidden_layers} layers")

    def _load_practitioner_model(self):
        """Load the practitioner (CAT) model."""
        practitioner_id = self.config.practitioner_model_id

        if practitioner_id is None:
            # Use same model as subject (self-assessment)
            logger.info("Using subject model as practitioner (self-assessment mode)")
            self.practitioner_model = self.subject_model
            self.practitioner_tokenizer = self.subject_tokenizer
            return

        if practitioner_id == self.config.subject_model_id:
            # Same model, share reference
            logger.info("Practitioner is same model as subject")
            self.practitioner_model = self.subject_model
            self.practitioner_tokenizer = self.subject_tokenizer
            return

        # Load separate practitioner model
        logger.info(f"Loading practitioner model: {practitioner_id}")

        dtype = getattr(torch, self.config.practitioner_dtype)

        self.practitioner_model = AutoModelForCausalLM.from_pretrained(
            practitioner_id,
            torch_dtype=dtype,
            device_map=self.config.practitioner_device,
            trust_remote_code=True,
        )
        self.practitioner_model.eval()

        self.practitioner_tokenizer = AutoTokenizer.from_pretrained(
            practitioner_id,
            trust_remote_code=True,
        )
        if self.practitioner_tokenizer.pad_token is None:
            self.practitioner_tokenizer.pad_token = self.practitioner_tokenizer.eos_token

        logger.info("Practitioner model loaded")

    def _setup_bedframe(self):
        """Set up BEDFrame for the subject."""
        bed_config = BEDConfig(
            be_id=f"subject_{self.session_id}",
            xdb_id=f"xdb_{self.session_id}",
            cat_id=f"cat_{self.session_id}",
            device=self.config.subject_device,
            xdb_storage_path=self.config.xdb_storage_path,
            audit_storage_path=self.config.audit_storage_path,
        )

        self.bedframe = BEDFrame(bed_config)
        self.bedframe.setup_model(self.subject_model, self.subject_tokenizer)

        # TODO: Set up lenses if lens_pack_path provided
        if self.config.lens_pack_path:
            logger.info(f"Lens pack path provided: {self.config.lens_pack_path}")
            # self._setup_lenses()

        logger.info("BEDFrame set up for subject")

    # =========================================================================
    # Practitioner Generation
    # =========================================================================

    def practitioner_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate response from the practitioner (CAT) model.

        Uses chat template for instruction-tuned models.
        """
        if self.practitioner_model is None:
            raise RuntimeError("Practitioner model not loaded")

        tokenizer = self.practitioner_tokenizer

        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.practitioner_model.device)

        with torch.inference_mode():
            outputs = self.practitioner_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
            )

        # Decode only generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # =========================================================================
    # Subject Generation
    # =========================================================================

    def subject_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_bedframe: bool = True,
    ) -> str:
        """
        Generate response from the subject.

        If use_bedframe=True, uses full BEDFrame instrumentation.
        Otherwise, uses direct generation.
        """
        max_tokens = max_tokens or self.config.max_response_tokens
        temperature = temperature if temperature is not None else self.config.assessment_temperature

        if use_bedframe and self.bedframe:
            # Use BEDFrame with full instrumentation
            tokens = []
            for token_text, tick in self.bedframe.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                tokens.append(token_text)
            return ''.join(tokens)

        # Direct generation (no instrumentation)
        return self._direct_generate(
            self.subject_model,
            self.subject_tokenizer,
            prompt,
            max_tokens,
            temperature,
        )

    def _direct_generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Direct generation without BEDFrame instrumentation."""
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # =========================================================================
    # Experience Access
    # =========================================================================

    def get_recent_ticks(self, n: int = 100) -> List[ExperienceTick]:
        """Get recent experience ticks from BEDFrame."""
        if self.bedframe:
            return self.bedframe.get_recent_ticks(n)
        return []

    def get_lens_traces(self) -> Dict[str, Any]:
        """Get lens traces from recent experience."""
        if self.bedframe:
            return self.bedframe.introspect().get('lens_traces', {})
        return {}

    # =========================================================================
    # Record Keeping
    # =========================================================================

    def record_procedure(
        self,
        procedure_type: str,
        result: Dict[str, Any],
    ):
        """Record a procedure that was run."""
        record = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'procedure_type': procedure_type,
            'result': result,
        }
        self.procedures_run.append(record)

        # Save to output directory
        output_file = self.config.output_dir / f"{self.session_id}_procedures.jsonl"
        with open(output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current session."""
        return {
            'session_id': self.session_id,
            'subject_model': self.config.subject_model_id,
            'practitioner_model': self.config.practitioner_model_id,
            'practitioner_qualified': self.practitioner_qualified,
            'procedures_run': len(self.procedures_run),
            'total_ticks': self.bedframe.current_tick_id if self.bedframe else 0,
        }
