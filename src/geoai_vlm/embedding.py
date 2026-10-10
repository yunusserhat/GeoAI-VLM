# -*- coding: utf-8 -*-
"""
Embedding Module for GeoAI-VLM
================================
Embedding generation for text, image and mixed-modal inputs.

* Qwen3-VL-Embedding models through Transformers or vLLM (the default);
* any CLIP-family dual encoder on the Hugging Face Hub -- CLIP, SigLIP,
  SigLIP 2, MetaCLIP, StreetCLIP and similar -- through
  :class:`ClipEmbeddingBackend` (``ImageEmbedder(backend="clip")``).

Every backend returns L2-normalised vectors, so inner product and cosine
similarity rank identically.
"""

from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


__all__ = [
    "ImageEmbedder",
    "TransformersEmbeddingBackend",
    "VLLMEmbeddingBackend",
    "ClipEmbeddingBackend",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_CLIP_MODEL",
]

#: Default model of the Qwen3-VL-Embedding backends.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
#: Default model of the CLIP-family backend.
DEFAULT_CLIP_MODEL = "google/siglip2-base-patch16-224"


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
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        max_length: int = 8192,
        min_pixels: int = 4096,
        max_pixels: int = 1843200,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # Qwen3-VL-Embedding loads with native transformers classes; custom
        # code only runs when the caller opts in.
        self.trust_remote_code = trust_remote_code

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
            "trust_remote_code": self.trust_remote_code,
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
            self.model_name, trust_remote_code=self.trust_remote_code,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code,
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

    @staticmethod
    def _last_token_pool(hidden_states, attention_mask):
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

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "The Qwen3-VL-Embedding Transformers backend uses Qwen's vision "
                "preprocessing helper (qwen-vl-utils). Install it with: "
                "pip install 'geoai-vlm[qwen]'"
            ) from exc

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
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code

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
            trust_remote_code=self.trust_remote_code,
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


# ---------------------------------------------------------------------------
# CLIP-family dual encoders
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _no_grad():
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.no_grad():
        yield


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _features(output, *names: str):
    """The projected embedding from a ``get_*_features`` call.

    transformers 4.x returns the tensor itself; transformers 5 returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds it.
    """
    if isinstance(output, Mapping) or hasattr(output, "pooler_output"):
        for name in names + ("pooler_output",):
            value = getattr(output, name, None)
            if value is None and isinstance(output, Mapping):
                value = output.get(name)
            if value is not None:
                return value
        raise TypeError(
            f"cannot find projected features in {type(output).__name__}; "
            f"looked for {names + ('pooler_output',)}"
        )
    return output


def _l2_normalise(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


class ClipEmbeddingBackend(BaseEmbeddingBackend):
    """Embeddings from a CLIP-family dual encoder on the Hugging Face Hub.

    Meant for any checkpoint whose model exposes ``get_image_features`` and
    ``get_text_features`` through ``AutoModel``: CLIP, SigLIP, SigLIP 2,
    MetaCLIP, StreetCLIP and similar. docs/models.md lists the checkpoints
    actually run. Image and text vectors share one space, so a text query
    retrieves images.

    Every vector is L2-normalised, so inner product equals cosine similarity.

    An input holding both an image and a text is embedded as the normalised
    weighted mean of its two normalised vectors (``fusion="mean"``,
    ``image_weight=0.5``). That keeps one dimension for every modality, so
    fused, image-only and text-only vectors can live in one index; it is a
    simple heuristic, not a learned joint representation.

    Args:
        model_name: Hugging Face model id or local path.
        device: ``"cuda"``, ``"cpu"``, ``"mps"``...; default: CUDA if available.
        dtype: Torch dtype name; default float32 (float16 on CUDA).
        text_padding: Tokeniser padding. Default ``"max_length"`` for the
            SigLIP family (as those models were trained) and ``True``
            otherwise.
        max_text_length: Token limit for text (default: 64 for SigLIP,
            the tokeniser's own limit otherwise).
        image_weight: Weight of the image vector when fusing (0..1).
        trust_remote_code: Run code shipped with the model. Off by default.
        revision: Model revision to load.

    The ``instruction`` argument of :meth:`embed` is accepted for interface
    compatibility and ignored: dual encoders take no instruction.
    """

    name = "clip"

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        text_padding: Optional[Union[str, bool]] = None,
        max_text_length: Optional[int] = None,
        image_weight: float = 0.5,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
    ):
        if not 0.0 <= image_weight <= 1.0:
            raise ValueError("image_weight must be between 0 and 1")
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.text_padding = text_padding
        self.max_text_length = max_text_length
        self.image_weight = image_weight
        self.trust_remote_code = trust_remote_code
        self.revision = revision

        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype:
            dtype = getattr(torch, self.dtype)
        else:
            dtype = torch.float16 if device.startswith("cuda") else torch.float32

        kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision

        print(f"Loading CLIP-family embedding model: {self.model_name}")
        processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        model = AutoModel.from_pretrained(self.model_name, **kwargs)
        if not (hasattr(model, "get_image_features") and hasattr(model, "get_text_features")):
            raise ValueError(
                f"{self.model_name!r} ({type(model).__name__}) is not a dual "
                "image/text encoder: it has no get_image_features/get_text_features."
            )
        self._model = model.to(device=device, dtype=dtype).eval()
        self._processor = processor
        self.device = device
        print(f"Embedding model loaded on {device}")

    # -- helpers ------------------------------------------------------------
    def _model_type(self) -> str:
        config = getattr(self._model, "config", None)
        return str(getattr(config, "model_type", "") or "")

    def _text_options(self) -> Dict[str, Any]:
        siglip = self._model_type().startswith("siglip")
        padding = self.text_padding
        if padding is None:
            padding = "max_length" if siglip else True
        options: Dict[str, Any] = {"padding": padding, "truncation": True}
        max_length = self.max_text_length or (64 if siglip else None)
        if max_length:
            options["max_length"] = max_length
        return options

    def _prepare(self, batch):
        model = self._model
        if not hasattr(batch, "to"):
            return batch
        device = getattr(model, "device", None)
        dtype = getattr(model, "dtype", None)
        try:
            return batch.to(device, dtype=dtype) if dtype is not None else batch.to(device)
        except TypeError:
            return batch.to(device)

    def _image_vectors(self, images: List[Image.Image]) -> np.ndarray:
        batch = self._prepare(self._processor(images=images, return_tensors="pt"))
        with _no_grad():
            out = self._model.get_image_features(**batch)
        return _l2_normalise(_to_numpy(_features(out, "image_embeds")))

    def _text_vectors(self, texts: List[str]) -> np.ndarray:
        batch = self._processor(text=texts, return_tensors="pt", **self._text_options())
        batch = self._prepare(batch)
        with _no_grad():
            out = self._model.get_text_features(**batch)
        return _l2_normalise(_to_numpy(_features(out, "text_embeds")))

    @staticmethod
    def _load(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("a dual encoder embeds one image per input")
            return ClipEmbeddingBackend._load(image[0])
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not fetched; pass a local path")
        with Image.open(image) as img:
            return img.convert("RGB")

    def embed(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str = "Represent the user's input.",
    ) -> np.ndarray:
        """Embed ``{"image": ..., "text": ...}`` dicts (either key optional)."""
        self.load_model()
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)

        image_slots = [i for i, inp in enumerate(inputs) if inp.get("image") is not None]
        text_slots = [i for i, inp in enumerate(inputs) if inp.get("text") is not None]
        # An input with neither is embedded as empty text, like the Qwen backends.
        empty = [i for i, inp in enumerate(inputs) if inp.get("image") is None and inp.get("text") is None]
        text_slots = sorted(set(text_slots) | set(empty))

        image_vecs = {}
        if image_slots:
            vectors = self._image_vectors([self._load(inputs[i]["image"]) for i in image_slots])
            image_vecs = dict(zip(image_slots, vectors))
        text_vecs = {}
        if text_slots:
            vectors = self._text_vectors([str(inputs[i].get("text") or "") for i in text_slots])
            text_vecs = dict(zip(text_slots, vectors))

        rows = []
        for i in range(len(inputs)):
            if i in image_vecs and i in text_vecs and inputs[i].get("text") is not None:
                fused = self.image_weight * image_vecs[i] + (1 - self.image_weight) * text_vecs[i]
                rows.append(fused)
            elif i in image_vecs:
                rows.append(image_vecs[i])
            else:
                rows.append(text_vecs[i])
        return _l2_normalise(np.vstack(rows).astype(np.float32))


class ImageEmbedder:
    """
    Multimodal embedder.

    Generates vectors from text, images, or mixed-modal inputs. Uses the
    Qwen3-VL-Embedding model family by default; ``backend="clip"`` uses any
    CLIP-family dual encoder (CLIP, SigLIP, SigLIP 2, MetaCLIP, StreetCLIP...).

    Args:
        model_name: HuggingFace model name or local path. Default:
            ``Qwen/Qwen3-VL-Embedding-2B``, or
            ``google/siglip2-base-patch16-224`` for ``backend="clip"``.
        backend: Backend to use (``"vllm"``, ``"transformers"``, ``"auto"``
            or ``"clip"``).
        instruction: Default instruction prepended to every input (ignored by
            the CLIP backend).
        **backend_kwargs: Extra keyword arguments forwarded to the backend constructor.

    Example:
        >>> embedder = ImageEmbedder()
        >>> vecs = embedder.embed_images("./images")
        >>> print(vecs.shape)
        (120, 2048)
        >>> clip = ImageEmbedder(backend="clip", model_name="google/siglip2-base-patch16-224")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "auto",
        instruction: str = "Represent the user's input.",
        **backend_kwargs,
    ):
        if model_name is None:
            model_name = DEFAULT_CLIP_MODEL if backend == "clip" else DEFAULT_EMBEDDING_MODEL
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
        elif self.backend_name == "clip":
            self._backend = ClipEmbeddingBackend(self.model_name, **self.backend_kwargs)
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
