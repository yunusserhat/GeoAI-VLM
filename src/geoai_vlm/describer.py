# -*- coding: utf-8 -*-
"""
Image Describer Module for GeoAI-VLM
=====================================
VLM-based image description with VLLM (primary) and Transformers (fallback) backends.
"""

from __future__ import annotations

import glob
import json
import os
from abc import ABC, abstractmethod
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
        
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_response": text}


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
    
    def describe(
        self,
        image_dir: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 8,
        resume: bool = True,
        image_extensions: List[str] = None,
        recursive: bool = True,
    ) -> pd.DataFrame:
        """
        Describe images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save results (Parquet). If None, returns without saving.
            batch_size: Number of images to process per batch
            resume: If True, skip already-processed images
            image_extensions: List of image extensions to process (default: [".jpg", ".jpeg", ".png"])
            recursive: If True, search subdirectories recursively
            
        Returns:
            DataFrame with image paths, IDs, raw responses, and parsed JSON fields
        """
        image_dir = Path(image_dir)
        extensions = image_extensions or [".jpg", ".jpeg", ".png"]
        
        # Collect image paths (recursively if requested)
        image_paths = []
        pattern_prefix = "**/" if recursive else ""
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"{pattern_prefix}*{ext}"))
            image_paths.extend(image_dir.glob(f"{pattern_prefix}*{ext.upper()}"))
        
        image_paths = sorted(set(image_paths))
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        if len(image_paths) == 0:
            return pd.DataFrame()
        
        # Handle resume
        processed_ids = set()
        if resume and output_path and Path(output_path).exists():
            existing_df = pd.read_parquet(output_path)
            processed_ids = set(existing_df["image_id"].tolist())
            print(f"Resuming: {len(processed_ids)} images already processed")
        
        # Filter to unprocessed images
        image_paths = [
            p for p in image_paths
            if p.stem not in processed_ids
        ]
        print(f"Images to process: {len(image_paths)}")
        
        if len(image_paths) == 0:
            print("All images already processed!")
            return pd.read_parquet(output_path) if output_path else pd.DataFrame()
        
        # Process in batches
        all_results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(0, len(image_paths), batch_size), desc="Processing"):
            batch_paths = image_paths[batch_idx:batch_idx + batch_size]
            batch_path_strs = [str(p) for p in batch_paths]
            
            # Generate descriptions
            responses = self.backend.generate(
                batch_path_strs,
                self.system_prompt,
                self.user_prompt,
            )
            
            # Parse results
            for img_path, response in zip(batch_paths, responses):
                parsed = parse_json_response(response)
                
                result = {
                    "image_path": str(img_path),
                    "image_id": img_path.stem,
                    "raw_response": response,
                    "parsed_json": json.dumps(parsed),
                    "parse_error": "error" in parsed,
                }
                
                # Extract key fields from parsed JSON
                if "error" not in parsed:
                    result["scene_narrative"] = parsed.get("scene_narrative", "")
                    
                    # Handle semantic_tags
                    tags = parsed.get("semantic_tags", [])
                    if isinstance(tags, list):
                        result["semantic_tags"] = ",".join(str(t) for t in tags)
                    else:
                        result["semantic_tags"] = str(tags)
                    
                    # Extract nested fields
                    result["land_use_primary"] = parsed.get("land_use_character", {}).get("primary", "unknown")
                    result["street_type"] = parsed.get("urban_morphology", {}).get("street_type", "unknown")
                    result["place_character"] = parsed.get("place_character", {}).get("dominant_activity", "unknown")
                    result["usable"] = parsed.get("image_quality", {}).get("usable_for_analysis", True)
                else:
                    result["scene_narrative"] = ""
                    result["semantic_tags"] = ""
                    result["land_use_primary"] = "error"
                    result["street_type"] = "error"
                    result["place_character"] = "error"
                    result["usable"] = False
                
                all_results.append(result)
            
            # Save incrementally
            if output_path:
                batch_df = pd.DataFrame(all_results[-len(batch_paths):])
                
                if Path(output_path).exists():
                    existing_df = pd.read_parquet(output_path)
                    combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    combined_df.to_parquet(output_path, index=False)
                else:
                    batch_df.to_parquet(output_path, index=False)
        
        # Return final results
        if output_path and Path(output_path).exists():
            return pd.read_parquet(output_path)
        
        return pd.DataFrame(all_results)
    
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
