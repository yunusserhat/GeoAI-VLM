"""
Segmentation model wrapper for Vision2Slope pipeline.
"""

import logging
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .core.interfaces import SegmentationProvider


class SegmentationModel(SegmentationProvider):
    """Wrapper for semantic segmentation model."""

    def __init__(self, model_name: str, device=None, cache_dir=None):
        """
        Initialize segmentation model.

        Args:
            model_name: HuggingFace model identifier
            device: Torch device or device string. ``None`` auto-detects CUDA.
            cache_dir: Directory for downloaded model weights.

        Note:
            ``device`` and ``cache_dir`` were previously accepted on
            ``ModelConfig`` but never reached the model, so configuring either
            had no effect on the pipeline path.
        """
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        """Load the pre-trained segmentation model."""
        try:
            self.logger.info(f"Loading segmentation model: {self.model_name}")
            load_kwargs = {}
            if self.cache_dir is not None:
                load_kwargs["cache_dir"] = self.cache_dir
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Perform semantic segmentation on an image.

        Args:
            image: PIL Image object

        Returns:
            Semantic segmentation map as numpy array
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            predicted_map = (
                self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                .cpu()
                .numpy()
            )

            return predicted_map

        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the segmentation model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_type": "Mask2Former",
        }
