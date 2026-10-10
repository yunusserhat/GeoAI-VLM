# -*- coding: utf-8 -*-
"""
Semantic segmentation measurements for GeoAI-VLM
=================================================
Per-image class pixel fractions from a semantic segmentation model trained on
Cityscapes (SegFormer, Mask2Former, ...) through Hugging Face Transformers.

Every output column names what was measured: ``vegetation_pixel_fraction`` is
the share of an image's labelled pixels that the model assigned to
"vegetation". It is an image measurement from one viewpoint -- not green-space
exposure, not canopy cover, not anything about the people who use the street.

Label mappings are versioned and named. Class ids are only meaningful together
with the mapping they come from: Cityscapes id 0 is "road" while Mapillary
Vistas id 13 is "Road", so a model is checked against the mapping before any
id is read, and a mismatch is an error rather than a silent relabelling.

Install the optional dependencies with ``pip install 'geoai-vlm[segment]'``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "LabelMapping",
    "LabelMappingError",
    "CITYSCAPES_19",
    "DEFAULT_SEGMENTATION_MODEL_CITYSCAPES",
    "mapping_from_id2label",
    "check_label_mapping",
    "class_pixel_fractions",
    "SemanticSegmenter",
]

#: Small (3.7M parameters) SegFormer trained on Cityscapes.
DEFAULT_SEGMENTATION_MODEL_CITYSCAPES = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


class LabelMappingError(ValueError):
    """A model's labels do not match the label mapping it is used with."""


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


@dataclass(frozen=True)
class LabelMapping:
    """A named, versioned mapping from class id to class name.

    Attributes:
        name: Mapping name, e.g. ``"cityscapes-19"``.
        version: Mapping version; change it whenever an id or name changes.
        id_to_name: Class id -> machine-readable class name.
        ignore_index: Mask value meaning "not labelled" (excluded from
            fractions).
        source: Where the definition comes from.
    """

    name: str
    version: str
    id_to_name: Mapping[int, str]
    ignore_index: Optional[int] = 255
    source: str = ""
    notes: Sequence[str] = field(default_factory=tuple)

    @property
    def tag(self) -> str:
        """``name@version``, recorded with every measurement."""
        return f"{self.name}@{self.version}"

    def column(self, class_id: int) -> str:
        """Output column for a class: ``<name>_pixel_fraction``."""
        return f"{self.id_to_name[class_id]}_pixel_fraction"


#: The 19 Cityscapes evaluation classes (train ids 0-18).
CITYSCAPES_19 = LabelMapping(
    name="cityscapes-19",
    version="1",
    id_to_name={
        0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole",
        6: "traffic_light", 7: "traffic_sign", 8: "vegetation", 9: "terrain", 10: "sky",
        11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train",
        17: "motorcycle", 18: "bicycle",
    },
    ignore_index=255,
    source="Cityscapes benchmark, 19 evaluation classes (train ids)",
    notes=("Not interchangeable with Mapillary Vistas ids.",),
)


def mapping_from_id2label(id2label: Mapping[Any, str], name: str, version: str = "1") -> LabelMapping:
    """Build a mapping from a model's own ``config.id2label``.

    Use this for a model trained on another label set; name it after the
    model (and revision) so its measurements cannot be mixed up with
    Cityscapes ones.
    """
    return LabelMapping(
        name=name,
        version=version,
        id_to_name={int(k): _normalise(v) for k, v in id2label.items()},
        ignore_index=None,
        source="model config id2label",
    )


def check_label_mapping(id2label: Mapping[Any, str], mapping: LabelMapping) -> None:
    """Raise :class:`LabelMappingError` unless the model's labels match *mapping*.

    Names are compared after normalisation (case, spaces and punctuation), id
    by id; the label sets must also have the same size.
    """
    model = {int(k): _normalise(v) for k, v in id2label.items()}
    expected = {int(k): _normalise(v) for k, v in mapping.id_to_name.items()}
    if model == expected:
        return
    mismatched = [
        f"{i}: model {model.get(i)!r} vs mapping {expected.get(i)!r}"
        for i in sorted(set(model) | set(expected))
        if model.get(i) != expected.get(i)
    ]
    raise LabelMappingError(
        f"model labels do not match {mapping.tag} ({len(model)} vs {len(expected)} classes); "
        f"first differences: {mismatched[:4]}. Use mapping_from_id2label() for this model "
        "instead of assuming the ids are equal."
    )


def class_pixel_fractions(mask: np.ndarray, mapping: LabelMapping = CITYSCAPES_19) -> Dict[str, Any]:
    """Class pixel fractions of one label mask.

    Fractions are over labelled pixels (``ignore_index`` excluded), so they
    sum to one when any pixel is labelled. Every class of the mapping gets a
    column, zero included; ids not in the mapping are counted as
    ``unmapped_pixel_fraction`` instead of being dropped.

    Returns:
        ``{"<class>_pixel_fraction": float, ..., "unmapped_pixel_fraction",
        "labelled_pixels", "label_mapping"}``.
    """
    values = np.asarray(mask).ravel()
    if mapping.ignore_index is not None:
        values = values[values != mapping.ignore_index]
    total = int(values.size)
    counts = np.bincount(values.astype(np.int64), minlength=max(mapping.id_to_name) + 1) if total else np.zeros(0)
    row: Dict[str, Any] = {}
    mapped = 0
    for class_id in sorted(mapping.id_to_name):
        count = int(counts[class_id]) if class_id < len(counts) else 0
        mapped += count
        row[mapping.column(class_id)] = count / total if total else np.nan
    row["unmapped_pixel_fraction"] = (total - mapped) / total if total else np.nan
    row["labelled_pixels"] = total
    row["label_mapping"] = mapping.tag
    return row


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


class SemanticSegmenter:
    """Run a Transformers semantic segmentation model and measure class fractions.

    Args:
        model_name: A semantic segmentation checkpoint (default: SegFormer-B0
            trained on Cityscapes).
        mapping: The label mapping the model must match (default
            Cityscapes-19). A mismatch raises when the model loads.
        device: ``"cuda"``, ``"cpu"``...; default CUDA if available.
        trust_remote_code: Run code shipped with the model. Off by default.
        revision: Model revision to load.

    Example:
        >>> seg = SemanticSegmenter()
        >>> table = seg.measure(["a.jpg", "b.jpg"])
        >>> table[["image_id", "vegetation_pixel_fraction", "sky_pixel_fraction"]]
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEGMENTATION_MODEL_CITYSCAPES,
        mapping: LabelMapping = CITYSCAPES_19,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
    ):
        self.model_name = model_name
        self.mapping = mapping
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "SemanticSegmenter needs transformers and torch. Install them with: "
                "pip install 'geoai-vlm[segment]'"
            ) from exc
        kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision
        processor = AutoImageProcessor.from_pretrained(self.model_name, **kwargs)
        model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name, **kwargs)
        check_label_mapping(model.config.id2label, self.mapping)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device

    def segment(self, image) -> np.ndarray:
        """Label mask (H x W class ids) for one image, at the image's own size."""
        from .chat import load_image

        self.load_model()
        pil = load_image(image)
        inputs = self.processor(images=pil, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        try:
            import torch

            with torch.no_grad():
                outputs = self.model(**inputs)
        except ImportError:  # pragma: no cover - torch is required in practice
            outputs = self.model(**inputs)
        masks = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(pil.height, pil.width)]
        )
        return _to_numpy(masks[0]).astype(np.int64)

    def measure(self, images: Sequence[Any], ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Class pixel fractions for each image.

        Args:
            images: Paths or PIL images.
            ids: Row ids; by default the file stem of each path, or its
                position.

        Returns:
            One row per image: ``image_id``, every ``<class>_pixel_fraction``,
            ``unmapped_pixel_fraction``, ``labelled_pixels``,
            ``label_mapping`` and ``segmentation_model``.
        """
        from pathlib import Path

        rows: List[Dict[str, Any]] = []
        for i, image in enumerate(images):
            if ids is not None:
                image_id = ids[i]
            elif isinstance(image, (str, Path)):
                image_id = Path(image).stem
            else:
                image_id = str(i)
            row = {"image_id": image_id}
            row.update(class_pixel_fractions(self.segment(image), self.mapping))
            row["segmentation_model"] = self.model_name
            rows.append(row)
        return pd.DataFrame(rows)
