"""
Vision2Slope pipeline with clean architecture and parallel processing support.
"""

import logging
from dataclasses import dataclass
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .core.types import ProcessingResult
from .core.exceptions import ConfigurationError
from .processor import StandardImageProcessor
from .models import SegmentationModel
from .detectors import SkewDetector
from .correctors import ImageCorrector
from .analyzers import RoadSlopeAnalyzer
from .visualizers import Visualizer
from .utils import Utils
from .pano2perspective import PanoramaTransformer



# ---------------------------------------------------------------------------
# Parallel worker state
# ---------------------------------------------------------------------------
# The worker used to be a closure defined inside process_batch_parallel, which
# captured `self`. multiprocessing sends the callable to the child by pickle, and
# a local function cannot be pickled, so the parallel path could not start. The
# closure body also rebuilt the segmentation model on every image; at city scale
# that dominates the run.
#
# The worker is now a module-level function, and the processor is built once per
# process by the pool initialiser and reused for every image that process takes.
_WORKER_STATE: Dict[str, Any] = {}


@dataclass
class LegacyComponentConfig:
    """Settings the legacy detector/corrector/analyzer components read.

    A plain module-level dataclass so it can be pickled into worker processes;
    the previous adapter built an instance of a class defined inside a method,
    which cannot cross a process boundary.
    """

    canny_threshold1: float = 50.0
    canny_threshold2: float = 150.0
    hough_threshold: int = 50
    min_line_length: int = 50
    max_line_gap: int = 10
    angle_tolerance: float = 5.0
    min_edge_points: int = 10
    use_weighted_average: bool = True
    morphology_kernel_size: int = 5
    ransac_residual_threshold: float = 10.0
    ransac_max_trials: int = 100
    ransac_random_state: Optional[int] = None


def _init_slope_worker(model_name, device, cache_dir, component_config):
    """Build one processor per worker process and keep it for reuse."""
    from .analyzers import RoadSlopeAnalyzer
    from .correctors import ImageCorrector
    from .detectors import SkewDetector

    segmentation_provider = SegmentationModel(
        model_name, device=device, cache_dir=cache_dir
    )

    # No visualizer in workers, to avoid I/O contention between processes.
    _WORKER_STATE["processor"] = StandardImageProcessor(
        segmentation_provider=segmentation_provider,
        skew_detector=SkewDetector(component_config),
        corrector=ImageCorrector(component_config),
        slope_analyzer=RoadSlopeAnalyzer(component_config),
        visualizer=None,
        logger=logging.getLogger("vision2slope.worker"),
    )


def _slope_worker(image_path: str):
    """Process one image with this process's already-loaded model."""
    processor = _WORKER_STATE.get("processor")
    if processor is None:
        raise RuntimeError(
            "slope worker used before _init_slope_worker ran; the pool must be "
            "created with initializer=_init_slope_worker"
        )
    return processor.process(image_path)


# ---------------------------------------------------------------------------
# Record-level accounting
# ---------------------------------------------------------------------------
#: Outcome of evaluating one processed image.
EVALUATION_STATUSES = (
    "measured",           # produced a usable slope measurement
    "processing_failed",  # the image never produced a result
    "angle_filtered",     # measured, but outside the accepted angle range
    "angle_missing",      # processed successfully but no road edge angle
)


def build_record_table(df: "pd.DataFrame", angle_threshold: float) -> "pd.DataFrame":
    """Return every processed image with an explicit ``evaluation_status``.

    The evaluation table previously kept only successful, within-threshold rows,
    and the run summary was computed from it -- so failures and filtered steep
    angles disappeared from their own denominator. Keeping every input here
    separates "what happened to each image" from "which measurements qualify".
    """
    records = df.copy()
    if len(records) == 0:
        records["evaluation_status"] = pd.Series(dtype="object")
        return records

    status = records.get("status")
    angle = records.get("road_edge_line_angle")

    failed = status.astype(str).ne("success") if status is not None else False
    missing = angle.isna() if angle is not None else True
    steep = (
        angle.abs() > float(angle_threshold)
        if angle is not None
        else False
    )

    evaluation = np.where(
        failed,
        "processing_failed",
        np.where(missing, "angle_missing", np.where(steep, "angle_filtered", "measured")),
    )
    records["evaluation_status"] = evaluation
    return records


def summarise_records(records: "pd.DataFrame") -> Dict[str, Any]:
    """Summarise a record table over *every* processed image."""
    total = int(len(records))
    counts = (
        records["evaluation_status"].value_counts().to_dict()
        if total and "evaluation_status" in records.columns
        else {}
    )
    measured = int(counts.get("measured", 0))
    return {
        "n_processed": total,
        "n_measured": measured,
        "n_processing_failed": int(counts.get("processing_failed", 0)),
        "n_angle_filtered": int(counts.get("angle_filtered", 0)),
        "n_angle_missing": int(counts.get("angle_missing", 0)),
        "measured_rate": (measured / total) if total else 0.0,
    }


def select_pano_metadata(metadata_df: "pd.DataFrame") -> "pd.DataFrame":
    """Keep every distinct panorama/road-direction pair.

    Dropping duplicates on ``pano_id`` alone kept the first row per panorama, so
    a panorama matching several road segments lost every direction but one --
    which is exactly the information a signed slope needs.
    """
    subset = ["pano_id"]
    for column in ("edge_bearing", "heading"):
        if column in metadata_df.columns:
            subset.append(column)
    return metadata_df.drop_duplicates(subset=subset, keep="first")


class Vision2SlopePipeline:
    """
    Vision2Slope pipeline with clean architecture.

    Features:
    - Hierarchical configuration management
    - Dependency injection for testability
    - Optimized parallel processing
    - Comprehensive error handling
    - Modular component design
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize Vision2Slope pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_path = Path(config.output_dir)
        self.logger = self._setup_logging()

        # Handle panorama transformation if needed
        if self.config.processing_config.is_panorama:
            self._prepare_panorama_images()

        self.processor = self._create_processor()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file_path = self.output_path / "pipeline.log"

        # Clear any existing handlers
        logger = logging.getLogger("vision2slope")
        logger.handlers = []
        logger.setLevel(getattr(logging, self.config.processing_config.log_level))

        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.processing_config.log_level))
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _prepare_panorama_images(self):
        """
        Convert panoramic images to perspective views if needed.

        This method will:
        1. Create a subdirectory for perspective views
        2. Transform panoramic images to left (90°) and right (270°) views
        3. Update input_dir to point to the perspective views directory
        """
        self.logger.info("=" * 60)
        self.logger.info("PANORAMA PREPROCESSING")
        self.logger.info("=" * 60)

        # Create output directory for perspective views
        perspective_dir = self.output_path / self.config.processing_config.panorama_output_dir
        perspective_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Input directory (panoramic images): {self.config.input_dir}")
        self.logger.info(f"Output directory (perspective views): {perspective_dir}")

        # Initialize panorama transformer
        panorama_transformer = PanoramaTransformer(self.config.processing_config)

        try:
            # Transform panoramic images to perspective views
            generated_files = panorama_transformer.transform_panorama(
                input_dir=self.config.input_dir,
                output_dir=str(perspective_dir),
                generate_left_right=True,  # Generate left (90°) and right (270°) views
            )

            self.logger.info(f"Generated {len(generated_files)} perspective views")
            self.logger.info("Panorama preprocessing completed successfully")
            self.logger.info("=" * 60)

            # Update config to process perspective views instead of originals
            # Note: zensvi.transform.ImageTransformer creates nested subdirectories
            # Check for the actual location of generated files
            actual_perspective_dir = self._find_actual_perspective_dir(perspective_dir)
            self.config.input_dir = str(actual_perspective_dir)
            self.logger.info(f"Updated input directory to: {self.config.input_dir}")

        except Exception as e:
            self.logger.error(f"Panorama preprocessing failed: {e}")
            raise ConfigurationError(f"Failed to process panoramic images: {e}")

    def _find_actual_perspective_dir(self, base_dir: Path) -> Path:
        """
        Find the actual directory containing perspective images.

        zensvi.transform.ImageTransformer creates nested subdirectories,
        so we need to search for the actual location of generated files.

        Args:
            base_dir: Base directory to search from

        Returns:
            Path to directory containing perspective images
        """
        # Search for directories containing perspective images
        for root, dirs, files in os.walk(base_dir):
            # Check if this directory contains image files
            image_files = [
                f
                for f in files
                if Path(f).suffix.lower() in self.config.processing_config.image_extensions
            ]
            if image_files:
                found_dir = Path(root)
                self.logger.debug(f"Found {len(image_files)} perspective images in: {found_dir}")
                return found_dir

        # If no images found, return the original directory
        self.logger.warning(f"No perspective images found in subdirectories, using: {base_dir}")
        return base_dir

    def _create_processor(self) -> StandardImageProcessor:
        """
        Create image processor with all components.

        Returns:
            Configured StandardImageProcessor
        """
        self.logger.info("Initializing pipeline components...")

        # Create components (adapting old classes to new interfaces)
        segmentation_provider = SegmentationModel(
            self.config.model_config.model_name,
            device=self.config.model_config.device,
            cache_dir=self.config.model_config.cache_dir,
        )

        # Create legacy-compatible config for old components
        legacy_config = self._create_legacy_config()

        skew_detector = SkewDetector(legacy_config)
        corrector = ImageCorrector(legacy_config)
        slope_analyzer = RoadSlopeAnalyzer(legacy_config)
        visualizer = Visualizer(self.config.viz_config, self.config.output_dir)

        processor = StandardImageProcessor(
            segmentation_provider=segmentation_provider,
            skew_detector=skew_detector,
            corrector=corrector,
            slope_analyzer=slope_analyzer,
            visualizer=visualizer,
            logger=self.logger,
        )

        self.logger.info("Pipeline components initialized successfully")
        return processor

    def _create_legacy_config(self) -> LegacyComponentConfig:
        """Build the settings object the legacy components read.

        Returns a module-level dataclass so it can be pickled into worker
        processes; the previous version defined its class inside this method,
        which cannot cross a process boundary.
        """
        detection = self.config.detection_config
        analysis = self.config.analysis_config
        return LegacyComponentConfig(
            canny_threshold1=detection.canny_threshold1,
            canny_threshold2=detection.canny_threshold2,
            hough_threshold=detection.hough_threshold,
            min_line_length=detection.min_line_length,
            max_line_gap=detection.max_line_gap,
            angle_tolerance=detection.angle_tolerance,
            min_edge_points=analysis.min_edge_points,
            use_weighted_average=analysis.use_weighted_average,
            morphology_kernel_size=analysis.morphology_kernel_size,
            ransac_residual_threshold=analysis.ransac_residual_threshold,
            ransac_max_trials=analysis.ransac_max_trials,
            ransac_random_state=analysis.ransac_random_state,
        )

    def process_batch(self) -> pd.DataFrame:
        """
        Process all images in the input directory.

        Returns:
            DataFrame with processing results
        """
        image_files = self._find_image_files()

        if not image_files:
            self.logger.warning(f"No image files found in {self.config.input_dir}")
            return pd.DataFrame()

        self.logger.info(f"Found {len(image_files)} images to process")

        # Process all images
        results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            result = self.processor.process(str(image_file))
            results.append(result)

        # Convert results to DataFrame and save
        df = self._results_to_dataframe(results)

        csv_path = self._save_results(df)

        records = self._build_and_save_records(df)

        # Post-process for bi-directional slope estimation
        bi_estimate_df = self._bi_slope_estimate(df, csv_path)

        # Save intermediate results if requested
        if self.config.viz_config.save_intermediate_results:
            self._save_intermediate_results(df)

        # Summarise over every processed image, not over the filtered subset.
        self._print_summary(records)

        return bi_estimate_df

    def process_batch_parallel(self, num_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Process all images using multiprocessing.

        Args:
            num_workers: Number of worker processes (None = use config)

        Returns:
            DataFrame with processing results
        """
        if num_workers is None:
            num_workers = self.config.processing_config.num_workers
        if num_workers <= 0:
            num_workers = cpu_count()

        self.logger.info(f"Using {num_workers} worker processes")

        image_files = self._find_image_files()

        if not image_files:
            self.logger.warning(f"No image files found in {self.config.input_dir}")
            return pd.DataFrame()

        self.logger.info(f"Found {len(image_files)} images to process")

        # The processor is built once per worker process by the initialiser and
        # reused for every image that process handles, instead of being rebuilt
        # for each image as the previous closure did.
        model_config = self.config.model_config
        init_args = (
            model_config.model_name,
            model_config.device,
            model_config.cache_dir,
            self._create_legacy_config(),
        )

        with Pool(
            processes=num_workers,
            initializer=_init_slope_worker,
            initargs=init_args,
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_slope_worker, [str(f) for f in image_files]),
                    total=len(image_files),
                    desc="Processing images (parallel)",
                )
            )

        # Convert results to DataFrame and save
        df = self._results_to_dataframe(results)
        csv_path = self._save_results(df)

        records = self._build_and_save_records(df)

        # Post-process
        bi_estimate_df = self._bi_slope_estimate(df, csv_path)

        if self.config.viz_config.save_intermediate_results:
            self._save_intermediate_results(df)

        # Summarise over every processed image, not over the filtered subset.
        self._print_summary(records)

        return bi_estimate_df

    def _find_image_files(self) -> List[Path]:
        """Find all image files in input directory."""
        input_path = Path(self.config.input_dir)
        image_files = [
            f
            for f in input_path.iterdir()
            if f.suffix.lower() in self.config.processing_config.image_extensions
        ]
        return sorted(image_files)

    def _results_to_dataframe(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """Convert list of ProcessingResult to DataFrame."""
        results_data = [result.to_dict() for result in results]
        return pd.DataFrame(results_data)

    def _save_results(self, df: pd.DataFrame) -> Path:
        """Save results DataFrame to CSV."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"vision2slope_results_{timestamp}.csv"
        csv_path = self.output_path / csv_filename
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to: {csv_path}")
        return csv_path

    def _bi_slope_estimate(self, df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        """
        Estimate slope using bi-directional approach.

        If metadata_csv is provided (with heading and edge_bearing columns),
        computes signed slope along the OSM edge (u→v) direction.
        Otherwise falls back to absolute-value estimation.

        Args:
            df: DataFrame with results
            csv_path: Path where original CSV was saved

        Returns:
            DataFrame with adjusted slopes
        """
        filtered_df = df[df["status"] == "success"].copy()

        if filtered_df.empty:
            self.logger.warning("No successful results for bi-directional slope estimation")
            return df

        # Extract pano_id and perspective_angle from filename
        filtered_df["pano_id"] = filtered_df["filename"].apply(Utils.get_pano_id_from_path)
        filtered_df["perspective_angle"] = filtered_df["filename"].apply(
            lambda x: Utils.get_perspective_angle_from_path(x) if "_Direction_" in x else 0.0
        )

        # Filter by angle threshold
        angle_threshold = self.config.analysis_config.filter_slope_angle
        filtered_df = filtered_df[filtered_df["road_edge_line_angle"].abs() <= angle_threshold]

        if filtered_df.empty:
            self.logger.warning("No valid segments after angle filtering")
            return df

        self.logger.info(f"Found {len(filtered_df)} valid segments for bi-directional estimation")

        # Check if metadata is available for signed slope estimation
        metadata_csv_path = self.config.processing_config.metadata_csv
        use_signed = False

        if metadata_csv_path is not None:
            metadata_df = pd.read_csv(metadata_csv_path)
            metadata_df["pano_id"] = metadata_df["pano_id"].astype(str)
            metadata_df = select_pano_metadata(metadata_df)

            required_cols = {"pano_id", "heading", "edge_bearing"}
            if required_cols.issubset(metadata_df.columns):
                filtered_df = filtered_df.merge(
                    metadata_df[["pano_id", "heading", "edge_bearing"]], on="pano_id", how="left"
                )
                has_metadata = filtered_df["heading"].notna() & filtered_df["edge_bearing"].notna()

                if has_metadata.any():
                    use_signed = True
                    self.logger.info(
                        "Using signed slope estimation with heading and edge_bearing metadata"
                    )
                else:
                    self.logger.warning(
                        "No matching metadata found for any pano_id, falling back to absolute slope"
                    )
            else:
                missing = required_cols - set(metadata_df.columns)
                self.logger.warning(
                    f"metadata_csv missing columns {missing}, falling back to absolute slope"
                )

        if use_signed:
            # Compute signed slope for each row
            filtered_df["signed_slope"] = filtered_df.apply(
                lambda row: (
                    Utils.compute_signed_slope(
                        row["road_edge_line_angle"],
                        row["perspective_angle"],
                        row["heading"],
                        row["edge_bearing"],
                    )
                    if pd.notna(row.get("heading")) and pd.notna(row.get("edge_bearing"))
                    else np.nan
                ),
                axis=1,
            )

            # Aggregate signed slope per pano_id
            estimate_slope_list = {}
            grouped = filtered_df.groupby("pano_id")

            for pano_id, group in tqdm(grouped, desc="Computing signed road slopes"):
                valid = group["signed_slope"].dropna()
                if valid.empty:
                    continue
                if self.config.analysis_config.use_weighted_average:
                    valid_mask = group["signed_slope"].notna()
                    slope_per_pano = np.average(
                        group.loc[valid_mask, "signed_slope"],
                        weights=group.loc[valid_mask, "road_area"],
                    )
                else:
                    slope_per_pano = valid.mean()
                estimate_slope_list[pano_id] = slope_per_pano

            filtered_df["road_estimated_slope"] = filtered_df["pano_id"].map(estimate_slope_list)
            filtered_df["road_estimated_slope_abs"] = filtered_df["road_estimated_slope"].abs()
        else:
            # Original absolute-value behavior
            estimate_slope_list = {}
            grouped = filtered_df.groupby("pano_id")

            for pano_id, group in tqdm(grouped, desc="Computing road slopes"):
                if self.config.analysis_config.use_weighted_average:
                    slope_per_pano = np.average(
                        np.abs(group["road_edge_line_angle"]), weights=group["road_area"]
                    )
                else:
                    slope_per_pano = np.abs(group["road_edge_line_angle"]).mean()
                estimate_slope_list[pano_id] = slope_per_pano

            filtered_df["road_estimated_slope"] = np.abs(
                filtered_df["pano_id"].map(estimate_slope_list)
            )

        # Save estimated results
        estimate_csv_path = csv_path.parent / csv_path.name.replace(".csv", "_estimate.csv")
        filtered_df.to_csv(estimate_csv_path, index=False)
        self.logger.info(f"Estimated results saved to: {estimate_csv_path}")

        return filtered_df

    def _save_intermediate_results(self, df: pd.DataFrame):
        """Save intermediate processing results."""
        intermediate_dir = self.output_path / "intermediate_results"
        intermediate_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Skew detection results
        skew_df = df[
            ["filename", "pano_id", "skew_angle", "skew_confidence", "num_lines_detected"]
        ].copy()
        skew_path = intermediate_dir / f"skew_detection_{timestamp}.csv"
        skew_df.to_csv(skew_path, index=False)

        # Slope estimation results
        slope_df = df[
            [
                "filename",
                "pano_id",
                "road_edge_line_slope",
                "road_edge_line_intercept",
                "road_edge_line_angle",
                "road_area",
            ]
        ].copy()
        slope_path = intermediate_dir / f"slope_estimation_{timestamp}.csv"
        slope_df.to_csv(slope_path, index=False)

        self.logger.info(f"Intermediate results saved to: {intermediate_dir}")

    def _build_and_save_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the record-level table and write it next to the results CSV.

        The measurement subset returned by :meth:`_bi_slope_estimate` answers
        "which measurements qualify"; this table answers "what happened to every
        image", and is what the run summary is computed from.
        """
        records = build_record_table(
            df, angle_threshold=self.config.analysis_config.filter_slope_angle
        )
        self.records_ = records

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        records_path = self.output_path / f"vision2slope_records_{timestamp}.csv"
        records.to_csv(records_path, index=False)
        self.logger.info(f"Record-level table saved to: {records_path}")
        return records

    def _print_summary(self, df: pd.DataFrame):
        """Print processing summary statistics."""
        if df is None or df.empty:
            self.logger.warning("No data available for summary")
            return

        total = len(df)
        status_counts = df["status"].value_counts()

        self.logger.info("=" * 60)
        self.logger.info("VISION2SLOPE PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total images: {total}")

        for status, count in status_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            self.logger.info(f"  {status}: {count} ({percentage:.1f}%)")

        # Every processed image is accounted for here, including the ones that
        # failed or were filtered out by the angle threshold.
        if "evaluation_status" in df.columns:
            summary = summarise_records(df)
            self.logger.info("\nEvaluation outcome (denominator = all processed):")
            self.logger.info(f"  processed:         {summary['n_processed']}")
            self.logger.info(f"  measured:          {summary['n_measured']}")
            self.logger.info(f"  processing_failed: {summary['n_processing_failed']}")
            self.logger.info(f"  angle_filtered:    {summary['n_angle_filtered']}")
            self.logger.info(f"  angle_missing:     {summary['n_angle_missing']}")
            self.logger.info(f"  measured rate:     {summary['measured_rate'] * 100:.1f}%")

        # Statistics for successful results
        successful_df = df[df["status"] == "success"]
        if len(successful_df) > 0:
            self.logger.info("\nSuccessful Results Statistics:")
            # Report whichever statistics are present. The summary runs after
            # all the work is done, so a missing column must not discard the run.
            stats = (
                ("Avg skew angle", "skew_angle", "{:.2f}°"),
                ("Avg road edge line slope", "road_edge_line_slope", "{:.4f}"),
                ("Avg road edge line angle", "road_edge_line_angle", "{:.2f}°"),
                ("Avg road area", "road_area", "{:.0f} pixels"),
                ("Avg road estimated slope", "road_estimated_slope", "{:.2f}°"),
            )
            for label, column, fmt in stats:
                if column not in successful_df.columns:
                    continue
                value = pd.to_numeric(successful_df[column], errors="coerce").mean()
                if pd.isna(value):
                    continue
                self.logger.info(f"  {label}: {fmt.format(value)}")

        self.logger.info("=" * 60)
