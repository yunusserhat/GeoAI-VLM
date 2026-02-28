# -*- coding: utf-8 -*-
"""
Embedding Module for GeoAI-VLM
================================
Multimodal embedding generation using Qwen3-VL-Embedding models.
Supports both Transformers and vLLM backends for text, image, and mixed-modal inputs.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


__all__ = [
    "ImageEmbedder",
    "TransformersEmbeddingBackend",
    "VLLMEmbeddingBackend",
]


class BaseEmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
        pass

    @abstractmethod
    def embed(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        """
        Generate embeddings for a list of multimodal inputs.

        Args:
            inputs: List of dicts, each with optional keys ``text``, ``image``.
            instruction: Task-specific instruction prepended as system prompt.

        Returns:
            2-D numpy array of shape ``(len(inputs), embed_dim)``.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class TransformersEmbeddingBackend(BaseEmbeddingBackend):
    """
    Transformers backend for Qwen3-VL-Embedding inference.

    Wraps the ``Qwen3VLEmbedder`` class from the official
    `Qwen3-VL-Embedding <https://github.com/QwenLM/Qwen3-VL-Embedding>`_ repo,
    using ``transformers`` for model loading and inference.

    Args:
        model_name: HuggingFace model name or local path.
        torch_dtype: Torch dtype string (e.g. ``"bfloat16"``).
        attn_implementation: Attention implementation (e.g. ``"flash_attention_2"``).
        max_length: Maximum context length.
        min_pixels: Minimum pixels for input images.
        max_pixels: Maximum pixels for input images.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        max_length: int = 8192,
        min_pixels: int = 4096,
        max_pixels: int = 1843200,
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self._model = None
        self._processor = None
        self._tokenizer = None

    def is_available(self) -> bool:
        """Check if Transformers is available."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        """Load the Qwen3-VL-Embedding model via Transformers."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        print(f"Loading Transformers embedding model: {self.model_name}")

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.torch_dtype:
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        else:
            kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._model = AutoModel.from_pretrained(
            self.model_name, device_map="auto", **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        print(f"Embedding model loaded on {self._model.device}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_conversation(
        self, inp: Dict[str, Any], instruction: str,
    ) -> List[Dict]:
        """Build a chat-template conversation list for one input."""
        content: List[Dict[str, Any]] = []

        # Image(s)
        image = inp.get("image")
        if image is not None:
            images = image if isinstance(image, list) else [image]
            for img in images:
                if isinstance(img, str):
                    if img.startswith(("http://", "https://")):
                        content.append({"type": "image", "image": img})
                    else:
                        abs_path = os.path.abspath(img)
                        content.append({"type": "image", "image": f"file://{abs_path}"})
                else:
                    # Assume PIL Image
                    content.append({"type": "image", "image": img})

        # Text
        text = inp.get("text")
        if text is not None:
            content.append({"type": "text", "text": text})

        if not content:
            content.append({"type": "text", "text": ""})

        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _last_token_pool(self, hidden_states, attention_mask):
        """Extract the last non-padding token's hidden state."""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]

    def embed(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        """Generate embeddings via Transformers."""
        self.load_model()

        import torch
        from qwen_vl_utils import process_vision_info

        all_embeddings: List[np.ndarray] = []

        for inp in tqdm(inputs, desc="Embedding (Transformers)"):
            conversation = self._build_conversation(inp, instruction)
            text = self._processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(conversation)

            proc_kwargs: Dict[str, Any] = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True,
                "max_length": self.max_length,
                "truncation": True,
            }
            if image_inputs:
                proc_kwargs["images"] = image_inputs
            if video_inputs:
                proc_kwargs["videos"] = video_inputs

            model_inputs = self._processor(**proc_kwargs).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**model_inputs)

            emb = self._last_token_pool(
                outputs.last_hidden_state, model_inputs["attention_mask"],
            )
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().float().numpy())

        return np.vstack(all_embeddings)


class VLLMEmbeddingBackend(BaseEmbeddingBackend):
    """
    vLLM backend for high-throughput embedding generation.

    Uses ``vllm.LLM`` with ``runner="pooling"`` and ``llm.embed()`` as described
    in the `Qwen3-VL-Embedding vLLM usage guide
    <https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B>`_.

    Args:
        model_name: HuggingFace model name or local path.
        dtype: Data type string (``"bfloat16"``, ``"float16"``, …).
        gpu_memory_utilization: Fraction of GPU memory to reserve.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

        self._llm = None

    def is_available(self) -> bool:
        """Check if vLLM is available with CUDA."""
        try:
            import vllm  # noqa: F401
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_model(self) -> None:
        """Load the vLLM pooling model."""
        if self._llm is not None:
            return

        import torch
        from vllm import LLM

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        tp_size = self.tensor_parallel_size or torch.cuda.device_count()

        print(f"Loading vLLM embedding model: {self.model_name}")

        self._llm = LLM(
            model=self.model_name,
            runner="pooling",
            dtype=self.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
        )

        print(f"Embedding model loaded on {tp_size} GPU(s) (vLLM pooling)")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_conversation(
        self, inp: Dict[str, Any], instruction: str,
    ) -> List[Dict]:
        """Build a chat-template conversation list for one input."""
        content: List[Dict[str, Any]] = []

        image = inp.get("image")
        if image is not None:
            images = image if isinstance(image, list) else [image]
            for img in images:
                if isinstance(img, str):
                    if img.startswith(("http://", "https://")):
                        content.append({"type": "image", "image": img})
                    else:
                        abs_path = os.path.abspath(img)
                        content.append({"type": "image", "image": f"file://{abs_path}"})
                else:
                    content.append({"type": "image", "image": img})

        text = inp.get("text")
        if text is not None:
            content.append({"type": "text", "text": text})

        if not content:
            content.append({"type": "text", "text": ""})

        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _prepare_vllm_input(
        self, inp: Dict[str, Any], instruction: str,
    ) -> Dict[str, Any]:
        """Prepare a single input dict for vLLM embed."""
        conversation = self._build_conversation(inp, instruction)

        prompt_text = self._llm.llm_engine.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        result: Dict[str, Any] = {"prompt": prompt_text}

        image = inp.get("image")
        if image is not None:
            images = image if isinstance(image, list) else [image]
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    if img.startswith(("http://", "https://")):
                        from vllm.multimodal.utils import fetch_image
                        pil_images.append(fetch_image(img))
                    else:
                        pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_images.append(img)
            if pil_images:
                result["multi_modal_data"] = {
                    "image": pil_images[0] if len(pil_images) == 1 else pil_images,
                }

        return result

    def embed(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        """Generate embeddings via vLLM pooling."""
        self.load_model()

        vllm_inputs = [self._prepare_vllm_input(inp, instruction) for inp in inputs]
        outputs = self._llm.embed(vllm_inputs)

        embeddings = np.array([o.outputs.embedding for o in outputs])

        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings


class ImageEmbedder:
    """
    Multimodal embedder using Qwen3-VL-Embedding models.

    Generates semantically rich vectors from text, images, or mixed-modal inputs.
    Uses the Qwen3-VL-Embedding model family by default — replacing lightweight
    text-only sentence-transformers with true multimodal representations.

    Args:
        model_name: HuggingFace model name or local path.
        backend: Backend to use (``"vllm"``, ``"transformers"``, or ``"auto"``).
        instruction: Default instruction prepended to every input.
        **backend_kwargs: Extra keyword arguments forwarded to the backend constructor.

    Example:
        >>> embedder = ImageEmbedder()
        >>> vecs = embedder.embed_images("./images")
        >>> print(vecs.shape)
        (120, 2048)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        backend: str = "auto",
        instruction: str = "Represent the user's input.",
        **backend_kwargs,
    ):
        self.model_name = model_name
        self.backend_name = backend
        self.instruction = instruction
        self.backend_kwargs = backend_kwargs

        self._backend: Optional[BaseEmbeddingBackend] = None

    @property
    def backend(self) -> BaseEmbeddingBackend:
        """Get or initialise the embedding backend (lazy)."""
        if self._backend is not None:
            return self._backend

        if self.backend_name == "auto":
            vllm_be = VLLMEmbeddingBackend(self.model_name, **self.backend_kwargs)
            if vllm_be.is_available():
                print("Using vLLM embedding backend")
                self._backend = vllm_be
            else:
                print("vLLM not available, falling back to Transformers embedding backend")
                self._backend = TransformersEmbeddingBackend(
                    self.model_name, **self.backend_kwargs,
                )
        elif self.backend_name == "vllm":
            self._backend = VLLMEmbeddingBackend(self.model_name, **self.backend_kwargs)
        elif self.backend_name == "transformers":
            self._backend = TransformersEmbeddingBackend(
                self.model_name, **self.backend_kwargs,
            )
        else:
            raise ValueError(f"Unknown embedding backend: {self.backend_name}")

        return self._backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: Plain text strings to embed.
            instruction: Task instruction (defaults to ``self.instruction``).
            batch_size: Number of texts per batch.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), embed_dim)``.
        """
        instr = instruction or self.instruction
        inputs = [{"text": t} for t in texts]
        return self._embed_batched(inputs, instr, batch_size)

    def embed_images(
        self,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        instruction: Optional[str] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Embed images from a directory path or list of file paths.

        Args:
            image_paths: A directory path (all images inside will be discovered)
                or a list of individual image file paths.
            instruction: Task instruction (defaults to ``self.instruction``).
            batch_size: Number of images per batch.

        Returns:
            ``np.ndarray`` of shape ``(n_images, embed_dim)``.
        """
        instr = instruction or self.instruction
        paths = self._resolve_image_paths(image_paths)
        inputs = [{"image": str(p)} for p in paths]
        return self._embed_batched(inputs, instr, batch_size)

    def embed_multimodal(
        self,
        inputs: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Embed mixed-modal inputs (text + image combinations).

        Each input dict may have keys ``text`` and/or ``image``.

        Args:
            inputs: List of multimodal input dicts.
            instruction: Task instruction (defaults to ``self.instruction``).
            batch_size: Number of inputs per batch.

        Returns:
            ``np.ndarray`` of shape ``(len(inputs), embed_dim)``.
        """
        instr = instruction or self.instruction
        return self._embed_batched(inputs, instr, batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed_batched(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str,
        batch_size: int,
    ) -> np.ndarray:
        """Run embedding in batches and concatenate results."""
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            emb = self.backend.embed(batch, instruction=instruction)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    @staticmethod
    def _resolve_image_paths(
        source: Union[str, Path, List[Union[str, Path]]],
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """Resolve a directory or list of paths into sorted image file paths."""
        exts = extensions or [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_dir():
                paths = []
                for ext in exts:
                    paths.extend(source_path.rglob(f"*{ext}"))
                    paths.extend(source_path.rglob(f"*{ext.upper()}"))
                return sorted(set(paths))
            else:
                return [source_path]

        return [Path(p) for p in source]
