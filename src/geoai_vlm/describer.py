# -*- coding: utf-8 -*-
"""
Image Describer Module for GeoAI-VLM
=====================================
VLM-based image description with VLLM (primary) and Transformers (fallback) backends.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .prompts import GEOAI_SYSTEM_PROMPT, GEOAI_USER_PROMPT, get_prompt_template


__all__ = [
    "ImageDescriber",
    "VLLMBackend",
    "TransformersBackend",
    "parse_json_response",
    "extract_summary_fields",
]


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model response.
    
    Args:
        text: Raw text response from VLM
        
    Returns:
        Parsed JSON dictionary, or error dict if parsing fails
    """
    try:
        text = text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Find start and end of code block
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])
            
            # Remove json language identifier if present
            if text.startswith("json"):
                text = text[4:].strip()
        
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_response": text}

    # Syntactically valid JSON is not necessarily a usable record. A bare list,
    # null, number or string parses cleanly but has none of the expected fields,
    # and silently returning it pushed an AttributeError into the caller mid-batch.
    if not isinstance(parsed, dict):
        return {
            "error": (
                "Expected a JSON object, got "
                f"{type(parsed).__name__}"
            ),
            "raw_response": text,
        }

    return parsed



# Summary columns are populated from whichever schema the prompt template
# produces. The "geoai" template nests its fields; the "simple" template uses
# flat description/tags keys. Reading only the geoai names silently left every
# simple-template run with empty summary columns.
_NARRATIVE_KEYS = ("scene_narrative", "description", "alt_detailed")
_TAG_KEYS = ("semantic_tags", "tags", "keywords")


def _first_present(parsed: Dict[str, Any], keys) -> Optional[Any]:
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return parsed[key]
    return None


def _nested(parsed: Dict[str, Any], outer: str, inner: str) -> str:
    block = parsed.get(outer)
    if isinstance(block, dict):
        value = block.get(inner)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def extract_summary_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the flat summary columns from a parsed model response.

    Quality is reported as three distinguishable states rather than collapsed
    into a permissive default:

    ``reported``  the model stated whether the image is usable (``usable`` is
                  True/False)
    ``unknown``   no quality block was returned (``usable`` is None -- it is
                  *not* assumed usable)
    ``error``     the response could not be parsed (``usable`` is None)
    """
    if "error" in parsed:
        return {
            "scene_narrative": "",
            "semantic_tags": "",
            "land_use_primary": "error",
            "street_type": "error",
            "place_character": "error",
            "usable": None,
            "quality_status": "error",
        }

    narrative = _first_present(parsed, _NARRATIVE_KEYS)
    tags = _first_present(parsed, _TAG_KEYS)

    if isinstance(tags, (list, tuple)):
        tags_str = ",".join(str(t) for t in tags)
    elif tags is None:
        tags_str = ""
    else:
        tags_str = str(tags)

    quality = parsed.get("image_quality")
    if isinstance(quality, dict) and "usable_for_analysis" in quality:
        usable = bool(quality["usable_for_analysis"])
        quality_status = "reported"
    else:
        usable = None
        quality_status = "unknown"

    return {
        "scene_narrative": "" if narrative is None else str(narrative),
        "semantic_tags": tags_str,
        "land_use_primary": _nested(parsed, "land_use_character", "primary"),
        "street_type": _nested(parsed, "urban_morphology", "street_type"),
        "place_character": _nested(parsed, "place_character", "dominant_activity"),
        "usable": usable,
        "quality_status": quality_status,
    }


def _upsert_records(
    existing: Optional[pd.DataFrame], batch: pd.DataFrame
) -> pd.DataFrame:
    """Append *batch*, replacing any prior rows for the same derived output.

    Keyed on (image_id, processing_id) so that re-describing an image with the
    same model and prompt overwrites its record instead of adding a duplicate,
    while a different model or prompt is kept as a separate record.
    """
    if existing is None or len(existing) == 0:
        return batch.reset_index(drop=True)

    if {"image_id", "processing_id"}.issubset(existing.columns):
        keys = set(
            zip(batch["image_id"].astype(str), batch["processing_id"].astype(str))
        )
        mask = [
            (str(i), str(pid)) not in keys
            for i, pid in zip(existing["image_id"], existing["processing_id"])
        ]
        existing = existing[mask]

    return pd.concat([existing, batch], ignore_index=True)


class BaseBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class VLLMBackend(BaseBackend):
    """VLLM backend for high-performance inference."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: Optional[int] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.llm = None
        self.processor = None
        self.sampling_params = None
    
    def is_available(self) -> bool:
        """Check if VLLM is available."""
        try:
            import vllm
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_model(self) -> None:
        """Load VLLM model and processor."""
        if self.llm is not None:
            return
        
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        
        # Set multiprocessing method for VLLM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        
        print(f"Loading VLLM model: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        tp_size = self.tensor_parallel_size or torch.cuda.device_count()
        
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            enforce_eager=False,
            seed=42,
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=-1,
        )
        
        print(f"Model loaded on {tp_size} GPU(s)")
    
    def _prepare_inputs(self, messages: List[Dict]) -> Dict:
        """Prepare inputs for VLLM inference."""
        from qwen_vl_utils import process_vision_info
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs,
        }
    
    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images."""
        self.load_model()
        
        # Prepare messages for each image
        inputs = []
        for img_path in image_paths:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            inputs.append(self._prepare_inputs(messages))
        
        # Run inference
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        
        return [output.outputs[0].text for output in outputs]


class TransformersBackend(BaseBackend):
    """Transformers backend for broader compatibility."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.model = None
        self.processor = None
    
    def is_available(self) -> bool:
        """Check if Transformers is available."""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    
    def load_model(self) -> None:
        """Load Transformers model and processor."""
        if self.model is not None:
            return
        
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        print(f"Loading Transformers model: {self.model_name}")
        
        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, self.torch_dtype)
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        
        print(f"Model loaded on {self.model.device}")
    
    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for images (one at a time for Transformers)."""
        self.load_model()
        
        results = []
        for img_path in image_paths:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            
            # Generate
            with __import__("torch").no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                )
            
            # Decode
            output_text = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]
            
            results.append(output_text)
        
        return results


class ImageDescriber:
    """
    VLM-based image describer with VLLM (primary) and Transformers (fallback) backends.
    
    Args:
        model_name: HuggingFace model name (default: Qwen/Qwen3-VL-2B-Instruct)
        backend: Backend to use ("vllm", "transformers", or "auto")
        prompt_template: Prompt template name ("geoai" or "simple") or None for custom
        system_prompt: Custom system prompt (overrides template)
        user_prompt: Custom user prompt (overrides template)
        **backend_kwargs: Additional kwargs passed to backend
        
    Example:
        >>> describer = ImageDescriber(model_name="Qwen/Qwen3-VL-2B-Instruct")
        >>> results = describer.describe("./images", batch_size=8)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        backend: str = "auto",
        prompt_template: Optional[str] = "geoai",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        **backend_kwargs,
    ):
        self.model_name = model_name
        self.backend_name = backend
        self.backend_kwargs = backend_kwargs
        
        # Set up prompts
        if prompt_template and system_prompt is None and user_prompt is None:
            template = get_prompt_template(prompt_template)
            self.system_prompt = template["system"]
            self.user_prompt = template["user"]
        else:
            self.system_prompt = system_prompt or GEOAI_SYSTEM_PROMPT
            self.user_prompt = user_prompt or GEOAI_USER_PROMPT
        
        # Initialize backend
        self._backend = None
    
    @property
    def backend(self) -> BaseBackend:
        """Get or initialize the backend."""
        if self._backend is not None:
            return self._backend
        
        if self.backend_name == "auto":
            # Try VLLM first, fall back to Transformers
            vllm_backend = VLLMBackend(self.model_name, **self.backend_kwargs)
            if vllm_backend.is_available():
                print("Using VLLM backend")
                self._backend = vllm_backend
            else:
                print("VLLM not available, falling back to Transformers")
                self._backend = TransformersBackend(self.model_name, **self.backend_kwargs)
        elif self.backend_name == "vllm":
            self._backend = VLLMBackend(self.model_name, **self.backend_kwargs)
        elif self.backend_name == "transformers":
            self._backend = TransformersBackend(self.model_name, **self.backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
        
        return self._backend
    
    # -- provenance -------------------------------------------------------
    @property
    def prompt_version(self) -> str:
        """Stable short hash of the prompt pair currently configured."""
        payload = json.dumps(
            {"system": self.system_prompt, "user": self.user_prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def processing_id(self) -> str:
        """Identity of *this* derived output: model + prompt together.

        Resume decisions key on this rather than the image id alone, so
        re-running the same images under a different model or prompt produces a
        new derived record instead of being skipped as already finished.
        """
        payload = json.dumps(
            {"model": self.model_name, "prompt": self.prompt_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def describe(
        self,
        image_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 8,
        resume: bool = True,
        image_extensions: List[str] = None,
        recursive: bool = True,
        image_paths: Optional[List[Union[str, Path]]] = None,
    ) -> pd.DataFrame:
        """
        Describe a set of images.

        Args:
            image_dir: Directory to scan for images. Ignored when *image_paths*
                is given.
            output_path: Path to save results (Parquet). If None, returns
                without saving.
            batch_size: Number of images to process per batch
            resume: If True, skip images already processed *by this same model
                and prompt* (see :attr:`processing_id`). Failed records are
                always retried.
            image_extensions: Extensions to scan for (default: .jpg/.jpeg/.png)
            recursive: If True, search subdirectories recursively
            image_paths: Explicit list of images to describe. Use this to
                guarantee that only a selected subset is processed, rather than
                everything that happens to sit in *image_dir*.

        Returns:
            DataFrame with image paths, IDs, raw responses, parsed JSON fields
            and provenance columns (``model_name``, ``prompt_version``,
            ``processing_id``, ``quality_status``).
        """
        if image_paths is not None:
            paths = [Path(p) for p in image_paths]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} requested image(s) do not exist, "
                    f"first: {missing[0]}"
                )
            image_paths_list = sorted(set(paths))
            print(f"Describing {len(image_paths_list)} selected images")
        elif image_dir is not None:
            image_dir = Path(image_dir)
            extensions = image_extensions or [".jpg", ".jpeg", ".png"]

            collected = []
            pattern_prefix = "**/" if recursive else ""
            for ext in extensions:
                collected.extend(image_dir.glob(f"{pattern_prefix}*{ext}"))
                collected.extend(image_dir.glob(f"{pattern_prefix}*{ext.upper()}"))

            image_paths_list = sorted(set(collected))
            print(f"Found {len(image_paths_list)} images in {image_dir}")
        else:
            raise ValueError("Provide either image_dir or image_paths")

        if len(image_paths_list) == 0:
            return pd.DataFrame()

        # -- resume ---------------------------------------------------------
        # Only records produced by this same model+prompt, and which actually
        # succeeded, count as finished work.
        existing_df = None
        processed_ids = set()
        if output_path and Path(output_path).exists():
            existing_df = pd.read_parquet(output_path)
            if resume:
                done = existing_df
                if "processing_id" in done.columns:
                    done = done[done["processing_id"] == self.processing_id]
                if "parse_error" in done.columns:
                    done = done[~done["parse_error"].astype(bool)]
                processed_ids = set(done["image_id"].astype(str).tolist())
                print(f"Resuming: {len(processed_ids)} images already processed")

        pending = [p for p in image_paths_list if p.stem not in processed_ids]
        print(f"Images to process: {len(pending)}")

        if len(pending) == 0:
            print("All images already processed!")
            if existing_df is not None:
                return existing_df
            return pd.DataFrame()

        # -- process --------------------------------------------------------
        all_results: List[Dict[str, Any]] = []

        for batch_idx in tqdm(range(0, len(pending), batch_size), desc="Processing"):
            batch_paths = pending[batch_idx: batch_idx + batch_size]
            batch_path_strs = [str(p) for p in batch_paths]

            responses = self.backend.generate(
                batch_path_strs,
                self.system_prompt,
                self.user_prompt,
            )

            for img_path, response in zip(batch_paths, responses):
                all_results.append(
                    self._build_record(img_path, response)
                )

            if output_path:
                batch_df = pd.DataFrame(all_results[-len(batch_paths):])
                existing_df = _upsert_records(existing_df, batch_df)
                existing_df.to_parquet(output_path, index=False)

        if output_path and existing_df is not None:
            return existing_df

        return pd.DataFrame(all_results)

    def _build_record(
        self, img_path: Path, response: str
    ) -> Dict[str, Any]:
        """Turn one raw model response into a fully-provenanced record."""
        parsed = parse_json_response(response)
        failed = "error" in parsed

        record: Dict[str, Any] = {
            "image_path": str(img_path),
            "image_id": img_path.stem,
            "raw_response": response,
            "parsed_json": json.dumps(parsed),
            "parse_error": failed,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "processing_id": self.processing_id,
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        record.update(extract_summary_fields(parsed))
        return record

    def describe_single(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Describe a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Parsed JSON dictionary with description
        """
        responses = self.backend.generate(
            [str(image_path)],
            self.system_prompt,
            self.user_prompt,
        )
        
        return parse_json_response(responses[0])
