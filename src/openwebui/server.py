#!/usr/bin/env python3
"""
FastAPI server for HatCat divergence visualization in OpenWebUI.

This provides an OpenAI-compatible API with divergence metadata.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import torch
import numpy as np
from pathlib import Path
import asyncio
import yaml
from src.visualization import get_color_mapper

app = FastAPI(title="HatCat Divergence API")

# CORS for OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve favicon at root
@app.get("/favicon.png")
async def favicon():
    favicon_path = STATIC_DIR / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")


# Request/Response models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hatcat-divergence"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True
    session_id: str = "default"  # For steering management


class DivergenceAnalyzer:
    """Singleton divergence analyzer."""

    def __init__(self):
        self.manager = None
        self.model = None
        self.tokenizer = None
        self.color_mapper = None
        self.initialized = False
        self.config = None

    def _load_config(self, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            print(f"⚠ Config file not found: {config_path}")
            print("  Using default configuration")
            return {
                "probe_pack_id": "gemma-3-4b-pt_sumo-wordnet-v2",
                "model": {
                    "name": "google/gemma-3-4b-pt",
                    "dtype": "bfloat16",
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                },
                "probe_manager": {
                    "base_layers": [0],
                    "use_activation_probes": True,
                    "use_text_probes": True,
                    "keep_top_k": 100,
                    "load_threshold": 0.3,
                    "max_loaded_probes": 1000,
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 8765,
                    "title": "HatCat Divergence API",
                },
                "generation": {
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "stream": True,
                }
            }

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"✓ Loaded config from: {config_path}")
        return config

    async def initialize(self, config_path: Path = None):
        """Load models and probes from config."""
        if self.initialized:
            return

        print("🎩 Initializing HatCat divergence analyzer...")

        # Load configuration
        self.config = self._load_config(config_path)

        from src.monitoring.dynamic_probe_manager import DynamicProbeManager
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Get probe manager config
        pm_config = self.config.get("probe_manager", {})
        probe_pack_id = self.config.get("probe_pack_id")

        # Load probe manager using config
        self.manager = DynamicProbeManager(
            probe_pack_id=probe_pack_id,
            base_layers=pm_config.get("base_layers", [0]),
            use_activation_probes=pm_config.get("use_activation_probes", True),
            use_text_probes=pm_config.get("use_text_probes", True),
            keep_top_k=pm_config.get("keep_top_k", 100),
            load_threshold=pm_config.get("load_threshold", 0.3),
            max_loaded_probes=pm_config.get("max_loaded_probes", 1000),
        )

        # Get model config
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "google/gemma-3-4b-pt")

        # Parse dtype
        dtype_str = model_config.get("dtype", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=model_config.get("device_map", "auto"),
            low_cpu_mem_usage=model_config.get("low_cpu_mem_usage", True),
        )
        self.model.eval()

        # Load color mapper
        self.color_mapper = get_color_mapper()

        self.initialized = True
        print(f"✓ Loaded {len(self.manager.loaded_activation_probes)} activation probes")
        print(f"✓ Loaded {len(self.manager.loaded_text_probes)} text probes")
        print(f"✓ Loaded sunburst color mapping for {len(self.color_mapper.positions)} concepts")

    def analyze_divergence(self, hidden_state: np.ndarray, token_embedding: np.ndarray) -> Dict[str, Any]:
        """Analyze divergence for a token using embedding centroids."""
        import time

        t0 = time.time()

        # Convert to tensor once
        h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")

        t1 = time.time()

        # Use new detect_and_expand_with_divergence method for embedding centroid-based divergence
        concepts_with_divergence, timing_info = self.manager.detect_and_expand_with_divergence(
            h,
            token_embedding=token_embedding,
            top_k=20,  # Get top 20 concepts
            return_timing=True
        )

        t2 = time.time()
        print(f"[PROFILE] Tensor conversion: {(t1-t0)*1000:.1f}ms, Detect with centroid divergence ({len(concepts_with_divergence)} concepts): {(t2-t1)*1000:.1f}ms")

        # Build result structures - only include concepts with text similarity
        divergences = []
        concepts_with_data = []  # For color mapping
        concepts_by_layer = {}  # Track which concepts are at each layer

        for concept_name, data in concepts_with_divergence.items():
            act_prob = data['probability']
            txt_prob = data.get('text_confidence')
            div = data.get('divergence')
            layer = data['layer']

            # Only include if we have valid text similarity and high activation
            if txt_prob is not None and div is not None:
                if act_prob > 0.5:  # High activation
                    divergences.append({
                        'concept': concept_name,
                        'layer': layer,
                        'activation': round(act_prob, 3),
                        'text_similarity': round(txt_prob, 3),
                        'divergence': round(div, 3),
                    })
                    # Track concepts by layer for parent filtering
                    if layer not in concepts_by_layer:
                        concepts_by_layer[layer] = []
                    concepts_by_layer[layer].append(concept_name)
                    # For color mapping: (concept name, activation, divergence)
                    concepts_with_data.append((concept_name, act_prob, abs(div)))

        # Filter out parent concepts when child concepts are detected
        # Get concept hierarchy from manager
        filtered_divergences = []
        child_concepts = set()  # Concepts that have been expanded

        for item in divergences:
            concept_name = item['concept']
            layer = item['layer']

            # Check if this concept has children in higher layers
            has_child_detected = False
            for child_layer in range(layer + 1, 6):  # Check higher layers
                if child_layer in concepts_by_layer:
                    # Check if any detected concepts in higher layer are children of this concept
                    path = self.manager.get_concept_path(concept_name, layer)
                    for detected in concepts_by_layer[child_layer]:
                        detected_path = self.manager.get_concept_path(detected, child_layer)
                        # If detected concept's path contains this concept, it's a child
                        if concept_name in detected_path:
                            has_child_detected = True
                            child_concepts.add(concept_name)
                            break
                if has_child_detected:
                    break

            # Only include if no child was detected
            if not has_child_detected:
                item['concept'] = f"{concept_name} (L{layer})"  # Add layer label
                filtered_divergences.append(item)

        # Sort by activation (descending)
        filtered_divergences.sort(key=lambda x: x['activation'], reverse=True)

        # Generate color using sunburst mapper
        token_color = "#808080"  # Default gray
        if concepts_with_data and self.color_mapper:
            token_color = self.color_mapper.blend_concept_colors_average(
                concepts_with_data,
                use_adaptive_saturation=True
            )

        # Generate palette swatch (top 5 concepts)
        palette = []
        if concepts_with_data and self.color_mapper:
            palette = self.color_mapper.create_palette_swatch(
                concepts_with_data,
                max_colors=5,
                saturation=0.7
            )

        return {
            'max_divergence': filtered_divergences[0]['divergence'] if filtered_divergences else 0.0,
            'top_divergences': filtered_divergences[:10],  # Return top 10 concepts sorted by activation, parents filtered
        }


# Global analyzer instance
analyzer = DivergenceAnalyzer()

# Global steering manager
from src.steering.steering_manager import SteeringManager
steering_manager = SteeringManager()


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    await analyzer.initialize()


@app.get("/")
async def root():
    """Health check."""
    return {
        "name": "HatCat Divergence API",
        "status": "ready" if analyzer.initialized else "initializing",
        "activation_probes": len(analyzer.manager.loaded_activation_probes) if analyzer.manager else 0,
        "text_probes": len(analyzer.manager.loaded_text_probes) if analyzer.manager else 0,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    packs = registry.get_pack_summary()

    # Build model list from available probe packs
    # Use probe_pack_id as the model ID to ensure uniqueness (v1, v2, v3 all have same concept_pack_id)
    models = []
    for pack in packs:
        models.append({
            "id": pack['probe_pack_id'],
            "object": "model",
            "created": 1234567890,
            "owned_by": "hatcat",
            "probe_pack_id": pack['probe_pack_id'],
            "concept_pack_id": pack['concept_pack_id'],
            "version": pack['version'],
        })

    # Add default model for backward compatibility
    if not models:
        models.append({
            "id": "hatcat-divergence",
            "object": "model",
            "created": 1234567890,
            "owned_by": "hatcat",
        })

    return {
        "object": "list",
        "data": models
    }


@app.get("/v1/concept-packs")
async def list_concept_packs():
    """List available concept packs."""
    from src.registry import ConceptPackRegistry

    registry = ConceptPackRegistry()
    return {
        "concept_packs": registry.get_pack_summary()
    }


@app.get("/v1/concept-packs/{pack_id}")
async def get_concept_pack(pack_id: str):
    """Get details for a specific concept pack."""
    from src.registry import ConceptPackRegistry

    registry = ConceptPackRegistry()
    pack = registry.get_pack(pack_id)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Concept pack not found: {pack_id}")

    return pack.pack_json


@app.get("/v1/probe-packs")
async def list_probe_packs():
    """List available probe packs."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    return {
        "probe_packs": registry.get_pack_summary()
    }


@app.get("/v1/probe-packs/{pack_id}")
async def get_probe_pack(pack_id: str):
    """Get details for a specific probe pack."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    pack = registry.get_pack(pack_id)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Probe pack not found: {pack_id}")

    return pack.pack_json


# ============================================================================
# HatCat Pack Discovery API Endpoints (MAP-compliant)
# ============================================================================

@app.get("/hatcat/pack-info")
async def hatcat_pack_info():
    """Get info about the currently loaded probe pack."""
    if not analyzer.initialized:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    config = analyzer.config
    if not config:
        raise HTTPException(status_code=500, detail="No configuration loaded")

    return {
        "probe_pack_id": config.get("probe_pack_id"),
        "model": config.get("model", {}),
        "probe_manager": config.get("probe_manager", {}),
        "loaded_activation_probes": len(analyzer.manager.loaded_activation_probes) if analyzer.manager else 0,
        "loaded_text_probes": len(analyzer.manager.loaded_text_probes) if analyzer.manager else 0,
    }


@app.get("/hatcat/available-packs")
async def hatcat_available_packs():
    """Get list of all available probe packs (MAP-compliant discovery)."""
    from src.monitoring.dynamic_probe_manager import DynamicProbeManager

    # Discover all probe packs
    packs = DynamicProbeManager.discover_probe_packs()

    # Format response
    pack_list = []
    for pack_id, info in packs.items():
        pack_list.append({
            "pack_id": pack_id,
            "type": info["type"],
            "path": str(info["path"]),
            "concept_pack_id": info.get("concept_pack_id"),
            "substrate_id": info.get("substrate_id"),
        })

    return {
        "packs": sorted(pack_list, key=lambda x: x["pack_id"]),
        "count": len(pack_list),
    }


# ============================================================================
# Steering API Endpoints
# ============================================================================

class AddSteeringRequest(BaseModel):
    """Request to add a steering."""
    session_id: str = "default"
    concept: str
    layer: int
    strength: float
    source: str = "user"
    reason: str = ""


class UpdateSteeringRequest(BaseModel):
    """Request to update steering strength."""
    session_id: str = "default"
    strength: float


@app.post("/v1/steering/add")
async def add_steering(request: AddSteeringRequest):
    """Add or update a concept steering."""
    steering = steering_manager.add_steering(
        session_id=request.session_id,
        concept=request.concept,
        layer=request.layer,
        strength=request.strength,
        source=request.source,
        reason=request.reason,
    )

    return {
        "status": "success",
        "steering": steering.to_dict(),
    }


@app.delete("/v1/steering/remove/{concept}")
async def remove_steering(
    concept: str,
    session_id: str = "default",
    layer: Optional[int] = None,
    source: Optional[str] = None,
):
    """Remove steering(s) for a concept."""
    removed_count = steering_manager.remove_steering(
        session_id=session_id,
        concept=concept,
        layer=layer,
        source=source,
    )

    return {
        "status": "success",
        "removed_count": removed_count,
    }


@app.get("/v1/steering/list")
async def list_steerings(session_id: str = "default", layer: Optional[int] = None):
    """List active steerings for a session."""
    steerings = steering_manager.get_steerings(session_id=session_id, layer=layer)

    return {
        "session_id": session_id,
        "steerings": [s.to_dict() for s in steerings],
        "count": len(steerings),
    }


@app.patch("/v1/steering/update/{concept}")
async def update_steering(concept: str, request: UpdateSteeringRequest):
    """Update steering strength for a concept."""
    # Remove old and add new (effectively an update)
    steering_manager.remove_steering(
        session_id=request.session_id,
        concept=concept,
    )

    # Re-add with new strength (keeps existing source/reason from first match)
    existing = steering_manager.get_steerings(session_id=request.session_id)
    source = "user"
    reason = ""

    # Try to preserve source/reason if it existed
    for s in existing:
        if s.concept == concept:
            source = s.source
            reason = s.reason
            break

    steering = steering_manager.add_steering(
        session_id=request.session_id,
        concept=concept,
        layer=0,  # Default layer
        strength=request.strength,
        source=source,
        reason=reason,
    )

    return {
        "status": "success",
        "steering": steering.to_dict(),
    }


@app.delete("/v1/steering/clear")
async def clear_steerings(session_id: str = "default"):
    """Clear all steerings for a session."""
    steering_manager.clear_session(session_id)

    return {
        "status": "success",
        "session_id": session_id,
    }


async def generate_stream(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming response with divergence coloring."""

    try:
        if not analyzer.initialized:
            await analyzer.initialize()

        # Build prompt from messages (only recent context to save memory)
        max_context_messages = 4  # Limit context to prevent OOM
        recent_messages = request.messages[-max_context_messages:]

        # Convert messages to dict format for chat template
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in recent_messages]

        # Simple prompt format - just the last user message to avoid teaching "User:" pattern
        # Gemma base model doesn't use chat template structure well
        # Just use the last user message as the prompt
        if recent_messages and recent_messages[-1].role == "user":
            prompt = recent_messages[-1].content
        else:
            prompt = "Hello"

        # Tokenize with truncation
        inputs = analyzer.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Limit input length to save memory
        ).to("cuda")
        generated_ids = inputs.input_ids

        # Get stop tokens for Gemma
        stop_tokens = [
            analyzer.tokenizer.eos_token_id,
            analyzer.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            analyzer.tokenizer.convert_tokens_to_ids("<eos>"),
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]

        # Clear CUDA cache before generation
        torch.cuda.empty_cache()

        # Setup steering if active
        from src.steering.extraction import extract_concept_vector
        steering_hooks = []
        active_steerings = steering_manager.get_steerings(request.session_id)

        # Collect divergence data for analysis message
        collected_divergences = []
        collected_concepts = {}

        if active_steerings:
            # Extract and apply concept vectors for active steerings
            for steering in active_steerings:
                try:
                    # Extract concept vector
                    concept_vector = extract_concept_vector(
                        analyzer.model,
                        analyzer.tokenizer,
                        steering.concept,
                        layer_idx=steering.layer,
                        device="cuda"
                    )

                    # Create steering hook
                    def make_steering_hook(vec, strength):
                        vec_tensor = torch.tensor(vec, dtype=torch.float32).to("cuda")
                        def hook(module, input, output):
                            hidden_states = output[0]
                            vec_matched = vec_tensor.to(dtype=hidden_states.dtype)
                            projection = (hidden_states @ vec_matched.unsqueeze(-1)) * vec_matched
                            steered = hidden_states - strength * projection
                            return (steered,)
                        return hook

                    # Register hook on target layer
                    if steering.layer == -1:
                        target_layer = analyzer.model.model.layers[-1]
                    else:
                        target_layer = analyzer.model.model.layers[steering.layer]

                    hook_fn = make_steering_hook(concept_vector, steering.strength)
                    handle = target_layer.register_forward_hook(hook_fn)
                    steering_hooks.append(handle)

                except Exception as e:
                    print(f"Warning: Failed to apply steering for {steering.concept}: {e}")

        # Generate token by token
        for step in range(request.max_tokens):
            try:
                # Get next token
                with torch.no_grad():
                    outputs = analyzer.model(
                        generated_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,  # Disable KV cache to save memory
                    )

                    next_token_logits = outputs.logits[:, -1, :].clone()

                    # Apply repetition penalty
                    if generated_ids.shape[1] > 1:
                        for token_id in set(generated_ids[0].tolist()):
                            # Penalize tokens that have already been generated
                            if next_token_logits[0, token_id] < 0:
                                next_token_logits[0, token_id] *= 1.2
                            else:
                                next_token_logits[0, token_id] /= 1.2

                    # Apply temperature
                    next_token_logits = next_token_logits / request.temperature

                    # Use sampling instead of greedy decoding
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Get hidden state from last transformer layer (convert bfloat16 to float32 for numpy compatibility)
                    hidden_state = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

                    # Get token embedding from first layer (layer 0 = embedding layer) for centroid comparison
                    token_embedding = outputs.hidden_states[0][0, -1, :].float().cpu().numpy()

                    # Free outputs immediately
                    del outputs
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                # OOM during generation - send error and stop
                error_chunk = {
                    "id": f"chatcmpl-{step}-error",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "hatcat-divergence",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": "\n\n[Generation stopped: CUDA out of memory. Try a shorter conversation or lower max_tokens.]"
                        },
                        "finish_reason": "length",
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                break

            # Decode token
            token_text = analyzer.tokenizer.decode([next_token_id.item()])

            # Check if we're starting to generate conversation structure (stop tokens)
            # Check the accumulated text for conversation markers
            generated_text = analyzer.tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:])
            if "\nUser" in generated_text or "\nAssistant" in generated_text:
                # Stop before sending - model is trying to continue the conversation format
                break

            # Analyze divergence using embedding centroids
            div_data = analyzer.analyze_divergence(hidden_state, token_embedding)

            # Collect divergence data for analysis
            collected_divergences.append(div_data['max_divergence'])
            for item in div_data.get('top_divergences', []):
                concept = item['concept']
                score = item['divergence']
                if concept not in collected_concepts:
                    collected_concepts[concept] = {'count': 0, 'total_div': 0.0}
                collected_concepts[concept]['count'] += 1
                collected_concepts[concept]['total_div'] += score

            # Format for OpenWebUI
            # Send plain text token + metadata with sunburst colors + steering info
            steering_metadata = {
                "active": len(active_steerings) > 0,
                "steerings": [
                    {
                        "concept": s.concept,
                        "strength": s.strength,
                        "layer": s.layer,
                        "source": s.source,
                    }
                    for s in active_steerings
                ] if active_steerings else []
            }

            chunk = {
                "id": f"chatcmpl-{step}",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "hatcat-divergence",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": token_text,
                        "metadata": {
                            "divergence": div_data,
                            "steering": steering_metadata,
                            # Color and palette removed to reduce JSON export size
                        }
                    },
                    "finish_reason": None,
                }]
            }

            yield f"data: {json.dumps(chunk)}\n\n"

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Trim context if getting too long (sliding window)
            if generated_ids.shape[1] > 2048:
                # Keep last 1024 tokens
                generated_ids = generated_ids[:, -1024:]

            # Stop on EOS or other stop tokens
            if next_token_id.item() in stop_tokens:
                break

            # Yield control
            await asyncio.sleep(0)

        # Final chunk
        final_chunk = {
            "id": "chatcmpl-final",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "hatcat-divergence",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"

        # Send HatCat analysis message
        if collected_divergences:
            # Compute aggregate statistics
            import numpy as np
            div_array = np.array(collected_divergences)
            min_div = float(np.min(div_array))
            max_div = float(np.max(div_array))
            mean_div = float(np.mean(div_array))
            percent_div = int(mean_div * 100)

            # Find top 3 concepts by (count × avg_divergence)
            concept_scores = []
            for concept, data in collected_concepts.items():
                avg_div = data['total_div'] / data['count']
                score = data['count'] * avg_div
                concept_scores.append((concept, data['count'], avg_div, score))

            concept_scores.sort(key=lambda x: x[3], reverse=True)
            top_concepts = concept_scores[:3]

            # Format collapsed summary
            concept_strs = [f"{c[0]}({c[1]})" for c in top_concepts]
            summary = f"**Analysis**: {percent_div}% divergence - {' - '.join(concept_strs)}"

            # Send as separate message from hatcat-analyzer with role
            analysis_chunk = {
                "id": "chatcmpl-analysis",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "hatcat-analyzer",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": summary,
                    },
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(analysis_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        # Catch any other errors
        error_chunk = {
            "id": "chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "hatcat-divergence",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"\n\n[Error: {str(e)}]"
                },
                "finish_reason": "stop",
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        # Remove steering hooks
        for handle in steering_hooks:
            handle.remove()

        # Always cleanup
        torch.cuda.empty_cache()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""

    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming not implemented yet
        raise HTTPException(status_code=400, detail="Non-streaming mode not supported")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
