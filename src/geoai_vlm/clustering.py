# -*- coding: utf-8 -*-
"""
Clustering Module for GeoAI-VLM
=================================
Semantic clustering of VLM descriptions using K-Means, with TF-IDF keyword
extraction and optional GeoAI category analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


__all__ = [
    "SemanticClusterer",
    "ClusterConfig",
]


# ---------------------------------------------------------------------------
# Custom stop-words for street-view VLM descriptions
# ---------------------------------------------------------------------------
_GEOAI_STOP_WORDS: List[str] = [
    # Location-specific terms (from VLM prompt – appear in almost all descriptions)
    "istanbul", "fatih", "district", "turkey", "türkiye", "turkish",
    "istanbul fatih", "fatih district", "istanbul district", "fatih istanbul",
    "in istanbul", "in fatih", "of istanbul", "of fatih",
    "historic district", "historic center", "old city", "peninsula",
    # Generic visual terms
    "visible", "scene", "image", "view", "appears", "shown", "showing", "display",
    "looking", "taken", "captured", "depicts", "shows", "featuring",
    "photograph", "photo", "picture", "streetview", "street view",
    # Common street elements (too generic)
    "street", "road", "sidewalk", "pavement", "lane", "path", "way",
    # Colours
    "white", "blue", "red", "black", "gray", "grey", "green", "yellow", "brown",
    # Directions and positions
    "left", "right", "front", "back", "side", "center", "middle", "near", "far",
    "foreground", "background", "distance",
    # Generic descriptors
    "large", "small", "tall", "short", "old", "new", "various", "several", "multiple",
    # Technical / metadata terms from VLM output
    "unknown", "false", "true", "occluded", "count", "position", "bearing_deg",
    "distance_bucket", "attributes", "color",
    # Common but non-distinctive
    "area", "part", "section", "located", "seen", "one", "two", "including",
    "nearby", "along", "next", "typical", "common", "urban",
]


@dataclass
class ClusterConfig:
    """
    Configuration for semantic clustering.

    Attributes:
        n_clusters: Number of K-Means clusters (``None`` → auto via elbow method).
        random_state: Random seed for reproducibility.
        embedding_columns: DataFrame columns joined to form the text fed to the embedder.
        separator: String used to join embedding columns.
        k_range: Range of *k* values to evaluate during the elbow method.
        n_keywords: Number of TF-IDF keywords per cluster.
        stop_words: Additional stop words for keyword extraction.
    """

    n_clusters: Optional[int] = 10
    random_state: int = 42
    embedding_columns: List[str] = field(
        default_factory=lambda: ["scene_narrative", "semantic_tags", "place_character"],
    )
    separator: str = " | "
    k_range: Tuple[int, int] = (2, 16)
    n_keywords: int = 10
    stop_words: Optional[List[str]] = None


class SemanticClusterer:
    """
    Semantic clustering pipeline for GeoAI-VLM descriptions.

    Combines an :class:`~geoai_vlm.embedding.ImageEmbedder` with K-Means
    clustering, TF-IDF keyword extraction and GeoAI category analysis.

    Args:
        embedder: An :class:`~geoai_vlm.embedding.ImageEmbedder` instance.
            If ``None``, the caller must supply pre-computed embeddings.
        config: Clustering configuration.

    Example:
        >>> from geoai_vlm import ImageEmbedder, SemanticClusterer, ClusterConfig
        >>> embedder = ImageEmbedder()
        >>> clusterer = SemanticClusterer(embedder, ClusterConfig(n_clusters=12))
        >>> gdf = clusterer.cluster(gdf_input)
    """

    def __init__(
        self,
        embedder=None,
        config: Optional[ClusterConfig] = None,
    ):
        self.embedder = embedder
        self.config = config or ClusterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_embedding_text(
        self,
        gdf: gpd.GeoDataFrame,
        columns: Optional[List[str]] = None,
        separator: Optional[str] = None,
    ) -> pd.Series:
        """
        Combine multiple DataFrame columns into a single embedding-ready text field.

        Args:
            gdf: Input GeoDataFrame.
            columns: Columns to concatenate (defaults to ``config.embedding_columns``).
            separator: Join separator (defaults to ``config.separator``).

        Returns:
            ``pd.Series`` of concatenated text strings.
        """
        cols = columns or self.config.embedding_columns
        sep = separator or self.config.separator

        def _join(row):
            parts: List[str] = []
            for col in cols:
                val = row.get(col, "")
                if pd.isna(val):
                    val = ""
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                val = str(val).strip()
                if val:
                    parts.append(val)
            return sep.join(parts)

        return gdf.apply(_join, axis=1)

    def find_optimal_k(
        self,
        gdf: gpd.GeoDataFrame,
        k_range: Optional[Tuple[int, int]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Evaluate the elbow method across a range of *k* values.

        Args:
            gdf: GeoDataFrame (used to build embedding text if *embeddings* is ``None``).
            k_range: ``(min_k, max_k)`` range to test (defaults to ``config.k_range``).
            embeddings: Pre-computed embeddings (skips embedding step).

        Returns:
            ``(k_values, inertias)`` tuple.
        """
        from sklearn.cluster import KMeans

        kr = k_range or self.config.k_range
        k_values = list(range(kr[0], kr[1]))

        if embeddings is None:
            embeddings = self._get_embeddings(gdf)

        inertias: List[float] = []
        for k in tqdm(k_values, desc="Elbow method"):
            km = KMeans(
                n_clusters=k, random_state=self.config.random_state, n_init=10,
            )
            km.fit(embeddings)
            inertias.append(float(km.inertia_))

        return k_values, inertias

    def cluster(
        self,
        gdf: gpd.GeoDataFrame,
        n_clusters: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        column_name: str = "cluster",
    ) -> gpd.GeoDataFrame:
        """
        Run K-Means clustering and add a cluster label column.

        Args:
            gdf: Input GeoDataFrame with VLM description columns.
            n_clusters: Override ``config.n_clusters``.
            embeddings: Pre-computed embeddings (skips embedding step).
            column_name: Name of the cluster column added to *gdf*.

        Returns:
            Copy of *gdf* with an integer ``cluster`` column.
        """
        from sklearn.cluster import KMeans

        k = n_clusters or self.config.n_clusters
        if k is None:
            raise ValueError(
                "n_clusters must be specified either in config or in the method call."
            )

        if embeddings is None:
            embeddings = self._get_embeddings(gdf)

        print(f"Performing K-Means clustering with k={k} ...")
        km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
        labels = km.fit_predict(embeddings)

        gdf = gdf.copy()
        gdf[column_name] = labels

        # Print distribution
        counts = Counter(labels)
        for cid, cnt in sorted(counts.items()):
            print(f"  Cluster {cid}: {cnt:,} ({cnt / len(labels) * 100:.1f}%)")

        return gdf

    def extract_keywords(
        self,
        gdf: gpd.GeoDataFrame,
        n_keywords: Optional[int] = None,
        stop_words: Optional[List[str]] = None,
        text_column: str = "embedding_text",
        cluster_column: str = "cluster",
    ) -> Dict[int, List[str]]:
        """
        Extract TF-IDF keywords per cluster.

        Args:
            gdf: GeoDataFrame with *text_column* and *cluster_column*.
            n_keywords: Number of keywords per cluster (default: ``config.n_keywords``).
            stop_words: Extra stop words (merged with built-in list).
            text_column: Column with text for TF-IDF.
            cluster_column: Column with cluster labels.

        Returns:
            ``{cluster_id: [keyword, …]}`` mapping.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

        n_kw = n_keywords or self.config.n_keywords
        extra = stop_words or self.config.stop_words or []
        all_sw = list(ENGLISH_STOP_WORDS) + _GEOAI_STOP_WORDS + extra

        # Ensure the text column exists
        if text_column not in gdf.columns:
            gdf = gdf.copy()
            gdf[text_column] = self.build_embedding_text(gdf)

        cluster_keywords: Dict[int, List[str]] = {}
        for cid in sorted(gdf[cluster_column].unique()):
            texts = gdf.loc[gdf[cluster_column] == cid, text_column].tolist()
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=all_sw,
                ngram_range=(1, 2),
                min_df=max(3, len(texts) // 50),
            )
            try:
                tfidf = vectorizer.fit_transform(texts)
                names = vectorizer.get_feature_names_out()
                mean_scores = np.asarray(tfidf.mean(axis=0)).flatten()
                top_idx = mean_scores.argsort()[-n_kw:][::-1]
                kws = [names[i] for i in top_idx]
            except Exception:
                kws = []
            cluster_keywords[cid] = kws

        return cluster_keywords

    def analyze_categories(
        self,
        gdf: gpd.GeoDataFrame,
        category_columns: Optional[List[str]] = None,
        cluster_column: str = "cluster",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Cross-tabulate GeoAI categorical fields per cluster.

        Args:
            gdf: GeoDataFrame with cluster labels.
            category_columns: Columns to profile (default: ``land_use_primary``,
                ``street_type``, ``place_character``).
            cluster_column: Column with cluster labels.

        Returns:
            ``{cluster_id: {"size": N, "land_use_primary_top": …, …}}``
        """
        cats = category_columns or ["land_use_primary", "street_type", "place_character"]
        profiles: Dict[int, Dict[str, Any]] = {}

        for cid in sorted(gdf[cluster_column].unique()):
            subset = gdf[gdf[cluster_column] == cid]
            profile: Dict[str, Any] = {"size": len(subset)}
            for col in cats:
                if col in subset.columns:
                    vc = subset[col].value_counts()
                    if len(vc) > 0:
                        profile[f"{col}_top"] = vc.index[0]
                        profile[f"{col}_pct"] = round(
                            vc.values[0] / len(subset) * 100, 1,
                        )
            profiles[cid] = profile

        return profiles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_embeddings(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Build embedding text and generate embeddings via the embedder."""
        if self.embedder is None:
            raise RuntimeError(
                "No embedder set.  Either pass an ImageEmbedder or supply "
                "pre-computed embeddings."
            )
        texts = self.build_embedding_text(gdf).tolist()
        return self.embedder.embed_texts(texts)
