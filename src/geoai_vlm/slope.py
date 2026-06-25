# -*- coding: utf-8 -*-
"""
Road slope estimation helpers for GeoAI-VLM.

The implementation follows the measurement idea used by Vision2Slope:
derive a road mask from Mapillary Vistas semantic labels, trace the upper
road boundary, fit a robust line, and report the road edge angle in degrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import RANSACRegressor

MAPILLARY_ROAD_CLASS_ID = 13
MAPILLARY_ROAD_CROSSWALK_CLASS_ID = 23
MAPILLARY_ROAD_MARKING_CLASS_ID = 24
DEFAULT_ROAD_CLASS_IDS = (
    MAPILLARY_ROAD_CLASS_ID,
    MAPILLARY_ROAD_CROSSWALK_CLASS_ID,
    MAPILLARY_ROAD_MARKING_CLASS_ID,
)
DEFAULT_SEGMENTATION_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
INVALID_SLOPE_VALUE = -999.0


@dataclass
class SlopeConfig:
    """Configuration for road slope estimation from semantic maps."""

    road_class_ids: Tuple[int, ...] = DEFAULT_ROAD_CLASS_IDS
    morphology_kernel_size: int = 15
    min_edge_points: int = 10
    ransac_residual_threshold: float = 1.0
    ransac_max_trials: int = 1000
    ransac_random_state: int = 42
    invalid_value: float = INVALID_SLOPE_VALUE


@dataclass
class SlopeResult:
    """Container for one road slope measurement."""

    road_edge_line_slope: float
    road_edge_line_intercept: float
    road_edge_line_angle: float
    road_area: int
    edge_point_count: int
    status: str = "success"
    road_mask: Optional[np.ndarray] = None
    edge_points: Optional[np.ndarray] = None

    @property
    def angle_degrees(self) -> float:
        """Return the fitted road edge angle in degrees."""
        return self.road_edge_line_angle

    def to_dict(self, include_arrays: bool = False) -> dict:
        """Convert the result to a serializable dictionary."""
        data = {
            "road_edge_line_slope": self.road_edge_line_slope,
            "road_edge_line_intercept": self.road_edge_line_intercept,
            "road_edge_line_angle": self.road_edge_line_angle,
            "road_area": self.road_area,
            "edge_point_count": self.edge_point_count,
            "slope_status": self.status,
        }
        if include_arrays:
            data["road_mask"] = self.road_mask
            data["edge_points"] = self.edge_points
        return data


def create_road_mask(
    semantic_map: np.ndarray,
    road_class_ids: Sequence[int] = DEFAULT_ROAD_CLASS_IDS,
    morphology_kernel_size: int = 15,
) -> np.ndarray:
    """
    Build a binary road mask from a semantic segmentation label map.

    Defaults use Mapillary Vistas class IDs for road, crosswalk lane marking,
    and general lane marking, matching the Vision2Slope convention.
    """
    semantic_array = np.asarray(semantic_map)
    if semantic_array.ndim != 2:
        raise ValueError("semantic_map must be a 2D label array")

    mask = np.isin(semantic_array, list(road_class_ids)).astype(np.uint8)
    return _open_binary_mask(mask, morphology_kernel_size)


def extract_road_edge(road_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the upper road boundary as ``(row, column)`` points.

    The returned coordinates use image convention: row/y first, column/x
    second. Empty masks return ``None``.
    """
    mask = np.asarray(road_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("road_mask must be a 2D array")

    road_rows, road_cols = np.where(mask)
    if road_rows.size == 0:
        return None

    points = []
    for col in np.unique(road_cols):
        rows = road_rows[road_cols == col]
        points.append((float(rows.min()), float(col)))

    return np.asarray(points, dtype=np.float64)


def fit_road_edge_line(
    edge_points: np.ndarray,
    config: Optional[SlopeConfig] = None,
) -> Tuple[float, float, float]:
    """
    Fit a robust line to road edge points.

    Returns ``(slope, intercept, angle_degrees)`` where the line is
    ``row = slope * column + intercept``.
    """
    cfg = config or SlopeConfig()
    points = np.asarray(edge_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("edge_points must be an array of (row, column) pairs")
    if len(points) < cfg.min_edge_points:
        raise ValueError(
            f"At least {cfg.min_edge_points} edge points are required, got {len(points)}"
        )

    x = points[:, 1].reshape(-1, 1)
    y = points[:, 0]
    ransac = RANSACRegressor(
        residual_threshold=cfg.ransac_residual_threshold,
        max_trials=cfg.ransac_max_trials,
        random_state=cfg.ransac_random_state,
    )
    ransac.fit(x, y)

    slope = float(np.ravel(ransac.estimator_.coef_)[0])
    intercept = float(np.ravel([ransac.estimator_.intercept_])[0])
    angle_degrees = float(np.degrees(np.arctan(slope)))
    return slope, intercept, angle_degrees


def estimate_slope_from_mask(
    road_mask: np.ndarray,
    config: Optional[SlopeConfig] = None,
    include_arrays: bool = False,
) -> SlopeResult:
    """Estimate road slope from a binary road mask."""
    cfg = config or SlopeConfig()
    mask = np.asarray(road_mask).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("road_mask must be a 2D array")

    road_area = int(mask.sum())
    if road_area == 0:
        return _invalid_result(cfg, "no_road", road_area, include_arrays, mask, None)

    edge_points = extract_road_edge(mask)
    edge_point_count = 0 if edge_points is None else int(len(edge_points))
    if edge_points is None or edge_point_count < cfg.min_edge_points:
        return _invalid_result(
            cfg,
            "not_enough_edge_points",
            road_area,
            include_arrays,
            mask,
            edge_points,
        )

    try:
        slope, intercept, angle = fit_road_edge_line(edge_points, cfg)
    except Exception:
        return _invalid_result(cfg, "slope_failed", road_area, include_arrays, mask, edge_points)

    return SlopeResult(
        road_edge_line_slope=slope,
        road_edge_line_intercept=intercept,
        road_edge_line_angle=angle,
        road_area=road_area,
        edge_point_count=edge_point_count,
        status="success",
        road_mask=mask if include_arrays else None,
        edge_points=edge_points if include_arrays else None,
    )


def estimate_slope_from_semantic_map(
    semantic_map: np.ndarray,
    config: Optional[SlopeConfig] = None,
    include_arrays: bool = False,
) -> SlopeResult:
    """Estimate road slope from a semantic segmentation label map."""
    cfg = config or SlopeConfig()
    road_mask = create_road_mask(
        semantic_map,
        road_class_ids=cfg.road_class_ids,
        morphology_kernel_size=cfg.morphology_kernel_size,
    )
    return estimate_slope_from_mask(road_mask, config=cfg, include_arrays=include_arrays)


class ImageSlopeEstimator:
    """
    Segment images with a Mapillary Vistas semantic model and estimate slope.

    The transformer model is loaded lazily on first use. Reuse one estimator
    instance for batches to avoid repeated model initialization.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEGMENTATION_MODEL,
        config: Optional[SlopeConfig] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.config = config or SlopeConfig()
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.processor = None
        self.model = None
        self._torch = None

    def segment(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Return a semantic segmentation label map for an image."""
        self._ensure_model()
        pil_image = _load_pil_image(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        predicted_map = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        return predicted_map.cpu().numpy()

    def estimate(
        self,
        image: Union[str, Path, Image.Image],
        include_arrays: bool = False,
    ) -> SlopeResult:
        """Estimate road slope directly from an image."""
        semantic_map = self.segment(image)
        return estimate_slope_from_semantic_map(
            semantic_map,
            config=self.config,
            include_arrays=include_arrays,
        )

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self._torch = torch
        self._device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.model.to(self._device)
        self.model.eval()


def estimate_image_slope(
    image: Union[str, Path, Image.Image],
    model_name: str = DEFAULT_SEGMENTATION_MODEL,
    config: Optional[SlopeConfig] = None,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    include_arrays: bool = False,
) -> SlopeResult:
    """Convenience wrapper for one-off image slope estimation."""
    estimator = ImageSlopeEstimator(
        model_name=model_name,
        config=config,
        device=device,
        cache_dir=cache_dir,
    )
    return estimator.estimate(image, include_arrays=include_arrays)


def estimate_slopes_from_images(
    image_paths: Iterable[Union[str, Path]],
    estimator: Optional[ImageSlopeEstimator] = None,
    model_name: str = DEFAULT_SEGMENTATION_MODEL,
    config: Optional[SlopeConfig] = None,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Estimate slope for a collection of images and return a DataFrame."""
    slope_estimator = estimator or ImageSlopeEstimator(
        model_name=model_name,
        config=config,
        device=device,
        cache_dir=cache_dir,
    )

    records = []
    for image_path in image_paths:
        path = Path(image_path)
        result = slope_estimator.estimate(path)
        record = {
            "filename": path.name,
            "image_id": path.stem,
            "pano_id": extract_pano_id(path.name),
            **result.to_dict(),
        }
        try:
            record["perspective_angle"] = extract_perspective_angle(path.name)
        except ValueError:
            record["perspective_angle"] = np.nan
        records.append(record)

    return pd.DataFrame.from_records(records)


def angle_difference(angle1: float, angle2: float) -> float:
    """Return the minimum absolute difference between two bearings."""
    diff = (float(angle1) - float(angle2)) % 360
    return float(360 - diff if diff > 180 else diff)


def compute_signed_slope(
    road_edge_angle: float,
    perspective_angle: float,
    heading: float,
    edge_bearing: float,
) -> float:
    """
    Convert an image-space road edge angle to a signed edge-direction slope.

    Positive values indicate uphill along ``edge_bearing``. The left/right
    perspective convention follows Vision2Slope: right views use the fitted
    angle sign, while left views invert it before aligning to the edge bearing.
    """
    magnitude = abs(float(road_edge_angle))
    if float(perspective_angle) == 270:
        slope_along_heading = -magnitude if road_edge_angle > 0 else magnitude
    else:
        slope_along_heading = magnitude if road_edge_angle > 0 else -magnitude

    if angle_difference(heading, edge_bearing) > 90:
        slope_along_heading = -slope_along_heading
    return float(slope_along_heading)


def aggregate_pano_slopes(
    df: pd.DataFrame,
    angle_column: str = "road_edge_line_angle",
    area_column: str = "road_area",
    filename_column: str = "filename",
    pano_id_column: str = "pano_id",
    perspective_column: str = "perspective_angle",
    heading_column: str = "heading",
    edge_bearing_column: str = "edge_bearing",
    angle_threshold: float = 10.0,
    use_weighted_average: bool = True,
) -> pd.DataFrame:
    """
    Add panorama-level slope estimates to a slope result table.

    If heading and edge bearing columns are present, signed slopes are computed.
    Otherwise the grouped absolute road edge angle is used.
    """
    result = df.copy()
    if angle_column not in result.columns:
        raise ValueError(f"Missing required angle column: {angle_column}")

    if pano_id_column not in result.columns:
        if filename_column not in result.columns:
            raise ValueError(f"Missing {pano_id_column!r} or {filename_column!r} column")
        result[pano_id_column] = result[filename_column].apply(extract_pano_id)

    if perspective_column not in result.columns and filename_column in result.columns:
        result[perspective_column] = result[filename_column].apply(
            lambda value: _extract_perspective_or_nan(value)
        )

    valid = result[angle_column].notna() & (result[angle_column].abs() <= angle_threshold)
    slope_values = result[angle_column].abs().astype(float)
    has_signed_metadata = {
        perspective_column,
        heading_column,
        edge_bearing_column,
    }.issubset(result.columns)

    if has_signed_metadata:
        complete_metadata = (
            result[perspective_column].notna()
            & result[heading_column].notna()
            & result[edge_bearing_column].notna()
        )
        signed_mask = valid & complete_metadata
        result["signed_slope"] = np.nan
        result.loc[signed_mask, "signed_slope"] = result.loc[signed_mask].apply(
            lambda row: compute_signed_slope(
                row[angle_column],
                row[perspective_column],
                row[heading_column],
                row[edge_bearing_column],
            ),
            axis=1,
        )
        slope_values = result["signed_slope"]
        valid = signed_mask

    estimates = {}
    for pano_id, group in result.loc[valid].groupby(pano_id_column):
        values = slope_values.loc[group.index].dropna()
        if values.empty:
            continue
        if use_weighted_average and area_column in group.columns:
            weights = group.loc[values.index, area_column].fillna(0).astype(float)
            if weights.sum() > 0:
                estimates[pano_id] = float(np.average(values, weights=weights))
            else:
                estimates[pano_id] = float(values.mean())
        else:
            estimates[pano_id] = float(values.mean())

    result["road_estimated_slope"] = result[pano_id_column].map(estimates)
    if has_signed_metadata:
        result["road_estimated_slope_abs"] = result["road_estimated_slope"].abs()
    return result


def extract_pano_id(filename: Union[str, Path]) -> str:
    """Extract a panorama id from a Vision2Slope-style filename."""
    name = Path(filename).name
    if "_Direction_" in name:
        return name.split("_Direction_")[0]
    return Path(name).stem


def extract_perspective_angle(filename: Union[str, Path]) -> float:
    """Extract ``_Direction_<angle>_FOV_`` from a filename."""
    match = re.search(r"_Direction_(\d+(?:\.\d+)?)_FOV_", Path(filename).name)
    if match is None:
        raise ValueError(f"Perspective angle not found in filename: {filename}")
    return float(match.group(1))


def _open_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size is None or kernel_size <= 1:
        return mask.astype(np.uint8)

    try:
        import cv2

        kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    except Exception:
        return mask.astype(np.uint8)


def _invalid_result(
    config: SlopeConfig,
    status: str,
    road_area: int,
    include_arrays: bool,
    road_mask: Optional[np.ndarray],
    edge_points: Optional[np.ndarray],
) -> SlopeResult:
    return SlopeResult(
        road_edge_line_slope=config.invalid_value,
        road_edge_line_intercept=config.invalid_value,
        road_edge_line_angle=config.invalid_value,
        road_area=road_area,
        edge_point_count=0 if edge_points is None else int(len(edge_points)),
        status=status,
        road_mask=road_mask if include_arrays else None,
        edge_points=edge_points if include_arrays else None,
    )


def _load_pil_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    return Image.open(image).convert("RGB")


def _extract_perspective_or_nan(filename: Union[str, Path]) -> float:
    try:
        return extract_perspective_angle(filename)
    except ValueError:
        return np.nan
