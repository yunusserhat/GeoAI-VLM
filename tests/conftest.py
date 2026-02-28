# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GeoAI-VLM tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Synthetic GeoDataFrame
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_gdf() -> gpd.GeoDataFrame:
    """Small GeoDataFrame mimicking GeoAI-VLM pipeline output (30 rows)."""
    rng = np.random.RandomState(42)
    n = 30

    land_uses = ["residential", "commercial", "mixed", "institutional", "green_space"]
    street_types = ["arterial", "collector", "local", "pedestrian", "alley"]
    characters = ["busy", "quiet", "touristic", "residential", "industrial"]

    data = {
        "image_id": [f"img_{i:04d}" for i in range(n)],
        "scene_narrative": [
            f"A {'busy' if i % 3 == 0 else 'quiet'} street with "
            f"{'shops' if i % 2 == 0 else 'apartments'} and some trees."
            for i in range(n)
        ],
        "semantic_tags": [
            ",".join(rng.choice(["urban", "green", "historic", "modern"], size=3).tolist())
            for _ in range(n)
        ],
        "land_use_primary": rng.choice(land_uses, n).tolist(),
        "street_type": rng.choice(street_types, n).tolist(),
        "place_character": rng.choice(characters, n).tolist(),
        "lat": (41.0 + rng.rand(n) * 0.02).tolist(),
        "lon": (28.97 + rng.rand(n) * 0.02).tolist(),
    }

    df = pd.DataFrame(data)
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Synthetic images directory
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_images_dir(tmp_path: Path, sample_gdf: gpd.GeoDataFrame) -> Path:
    """Create a temporary directory with small synthetic JPEG images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img_id in sample_gdf["image_id"]:
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(img_dir / f"{img_id}.jpg")
    return img_dir


# ---------------------------------------------------------------------------
# Mock embedder (no GPU / model download required)
# ---------------------------------------------------------------------------
MOCK_EMBED_DIM = 128


class _MockEmbeddingBackend:
    """Deterministic mock: returns fixed-seed random embeddings."""

    def __init__(self, dim: int = MOCK_EMBED_DIM):
        self.dim = dim

    def load_model(self):
        pass

    def is_available(self):
        return True

    def embed(
        self,
        inputs: List[Dict[str, Any]],
        instruction: str = "",
    ) -> np.ndarray:
        rng = np.random.RandomState(len(inputs))
        emb = rng.randn(len(inputs), self.dim).astype(np.float32)
        # L2-normalise
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms


class MockImageEmbedder:
    """Drop-in replacement for :class:`geoai_vlm.embedding.ImageEmbedder`."""

    def __init__(self, dim: int = MOCK_EMBED_DIM):
        self.instruction = "mock"
        self._backend = _MockEmbeddingBackend(dim=dim)

    @property
    def backend(self):
        return self._backend

    def embed_texts(self, texts, instruction=None, batch_size=32):
        inputs = [{"text": t} for t in texts]
        return self._embed_batched(inputs, instruction or self.instruction, batch_size)

    def embed_images(self, image_paths, instruction=None, batch_size=8):
        if isinstance(image_paths, (str, Path)):
            p = Path(image_paths)
            if p.is_dir():
                paths = sorted(p.rglob("*.jpg"))
            else:
                paths = [p]
        else:
            paths = list(image_paths)
        inputs = [{"image": str(p)} for p in paths]
        return self._embed_batched(inputs, instruction or self.instruction, batch_size)

    def embed_multimodal(self, inputs, instruction=None, batch_size=8):
        return self._embed_batched(inputs, instruction or self.instruction, batch_size)

    def _embed_batched(self, inputs, instruction, batch_size):
        all_embs = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start: start + batch_size]
            all_embs.append(self._backend.embed(batch, instruction))
        return np.vstack(all_embs)


@pytest.fixture
def mock_embedder() -> MockImageEmbedder:
    """A mock ImageEmbedder that requires no GPU or model."""
    return MockImageEmbedder()


# ---------------------------------------------------------------------------
# Temporary parquet file
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_parquet(tmp_path: Path, sample_gdf: gpd.GeoDataFrame) -> Path:
    """Write sample_gdf to a temporary GeoParquet file and return its path."""
    path = tmp_path / "sample.parquet"
    sample_gdf.to_parquet(path, index=False)
    return path
