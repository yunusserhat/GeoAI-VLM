# -*- coding: utf-8 -*-
"""
Vector Store Module for GeoAI-VLM
===================================
Abstract vector database layer with ChromaDB and FAISS backends for
similarity search over multimodal embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


__all__ = [
    "VectorDB",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "RESULT_KEYS",
]



# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------
# Backends disagree about what their score field means. ChromaDB always returns
# a *distance* (lower is closer). FAISS with metric="ip" returns a raw inner
# product (higher is closer), and with metric="l2" a squared distance. Reporting
# all three in one field called "distance" meant that sorting results ascending
# silently reversed the ranking on the FAISS default.
#
# Every query result therefore carries, alongside the backend's raw numbers:
#   metric      the backend's own metric name
#   direction   "lower_is_closer" or "higher_is_closer", describing `distances`
#   similarity  a uniform score where higher is always closer
#   rank        0-based position, already ordered best-first
#
# `similarity` is the true cosine similarity where the metric allows it
# (ChromaDB cosine/ip, FAISS ip). For an L2 metric it is the negated distance:
# order-preserving and directly comparable within one store, but not a
# calibrated cosine value and not comparable across metrics.
RESULT_KEYS = (
    "ids",
    "distances",
    "metadatas",
    "documents",
    "metric",
    "direction",
    "similarity",
    "rank",
)


def _build_result(
    metric: str,
    direction: str,
    similarity: List[float],
    ids: List[str],
    distances: List[float],
    metadatas: List[Dict[str, Any]],
    documents: List[str],
) -> Dict[str, Any]:
    """Assemble a query result that states its own metric and direction."""
    return {
        "ids": list(ids),
        "distances": [float(d) for d in distances],
        "metadatas": list(metadatas),
        "documents": list(documents),
        "metric": metric,
        "direction": direction,
        "similarity": [float(v) for v in similarity],
        "rank": list(range(len(ids))),
    }


def _matches(metadata: Optional[Dict[str, Any]], where: Optional[Dict[str, Any]]) -> bool:
    """Equality-only metadata predicate, used to filter FAISS results."""
    if not where:
        return True
    if not metadata:
        return False
    return all(metadata.get(key) == value for key, value in where.items())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseVectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add vectors with optional metadata and documents."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the store for nearest neighbours, best match first.

        Args:
            query_embedding: The query vector.
            n_results: Maximum number of results.
            where: Optional equality filter over stored metadata,
                e.g. ``{"land_use_primary": "commercial"}``.

        Returns:
            Dict with the keys listed in :data:`RESULT_KEYS`. Use ``similarity``
            (higher is always closer) or ``rank`` rather than ``distances``,
            whose meaning and direction depend on the backend and metric.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the store to disk (no-op for already-persistent backends)."""
        pass


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------
class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store with persistent storage.

    Args:
        persist_directory: Path to the directory for persistent storage.
        collection_name: Name of the ChromaDB collection.
        distance_fn: Distance function (``"cosine"``, ``"l2"``, or ``"ip"``).
    """

    def __init__(
        self,
        persist_directory: Union[str, Path] = "./chroma_db",
        collection_name: str = "geoai_embeddings",
        distance_fn: str = "cosine",
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.distance_fn = distance_fn

        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy-initialising ChromaDB client."""
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    @property
    def collection(self):
        """Lazy-initialising ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn},
            )
        return self._collection

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Add vectors in batches of *batch_size*."""
        total = len(ids)
        for start in tqdm(
            range(0, total, batch_size), desc="Adding to ChromaDB",
        ):
            end = min(start + batch_size, total)
            kwargs: Dict[str, Any] = {
                "ids": ids[start:end],
                "embeddings": embeddings[start:end].tolist(),
            }
            if metadatas is not None:
                kwargs["metadatas"] = metadatas[start:end]
            if documents is not None:
                kwargs["documents"] = documents[start:end]
            self.collection.upsert(**kwargs)

    @property
    def direction(self) -> str:
        """ChromaDB always returns a distance, whichever space is configured."""
        return "lower_is_closer"

    def _to_similarity(self, distances: List[float]) -> List[float]:
        # Chroma's "cosine" and "ip" spaces both return 1 - <similarity>, so the
        # true similarity is recoverable; "l2" is a squared distance, for which
        # only an order-preserving score is available.
        if self.distance_fn in ("cosine", "ip"):
            return [1.0 - float(d) for d in distances]
        return [-float(d) for d in distances]

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query nearest neighbours, best match first."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.count() == 0:
            return _build_result(
                self.distance_fn, self.direction, [], [], [], [], []
            )

        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": min(n_results, self.count()),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        ids = results["ids"][0] if results.get("ids") else []
        distances = results["distances"][0] if results.get("distances") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        documents = results["documents"][0] if results.get("documents") else []

        metadatas = [m or {} for m in metadatas] or [{} for _ in ids]
        documents = list(documents) or ["" for _ in ids]

        return _build_result(
            self.distance_fn,
            self.direction,
            self._to_similarity(distances),
            ids,
            distances,
            metadatas,
            documents,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def persist(self) -> None:
        # PersistentClient auto-persists; nothing to do.
        pass


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------
class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store for fast in-memory similarity search.

    Args:
        dimension: Embedding dimensionality (inferred on first ``add`` if *None*).
        index_type: ``"flat"`` (brute-force) or ``"ivf"`` (approximate).
        metric: ``"ip"`` (inner product / cosine on L2-normed vecs) or ``"l2"``.
        nlist: Number of IVF cells when ``index_type="ivf"``.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "ip",
        nlist: int = 100,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist

        self._index = None
        self._id_list: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._documents: List[str] = []

    @property
    def direction(self) -> str:
        """Inner product ranks descending; L2 ranks ascending."""
        return "higher_is_closer" if self.metric == "ip" else "lower_is_closer"

    def _to_similarity(self, distances: List[float]) -> List[float]:
        # For unit-norm vectors an inner product *is* the cosine similarity.
        # A squared L2 distance only yields an order-preserving score.
        if self.metric == "ip":
            return [float(d) for d in distances]
        return [-float(d) for d in distances]

    def _build_index(self, dim: int):
        """Build the FAISS index."""
        import faiss

        if self.metric == "ip":
            if self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dim)
                # IndexIVFFlat defaults to METRIC_L2 regardless of the
                # quantizer, so omitting this built an L2 index for a caller who
                # asked for inner product and returned ascending L2 distances
                # under the inner-product contract.
                self._index = faiss.IndexIVFFlat(
                    quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT,
                )
            else:
                self._index = faiss.IndexFlatIP(dim)
        else:
            if self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer, dim, self.nlist, faiss.METRIC_L2,
                )
            else:
                self._index = faiss.IndexFlatL2(dim)

        if self.index_type == "ivf":
            # Default nprobe is 1, which searches a single cell and loses recall.
            self._index.nprobe = max(1, min(self.nlist, 10))

        self.dimension = dim

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Re-adding an existing id updates it, matching ChromaDB's upsert.
        # Appending instead left two entries under one id, so the stale vector
        # stayed searchable and count() no longer matched the number of ids.
        known = set(self._id_list)
        duplicates = [i for i in ids if i in known]
        if duplicates:
            self.delete(duplicates)

        if self._index is None:
            self._build_index(embeddings.shape[1])

        self._train_if_needed(embeddings)

        self._index.add(embeddings)
        self._id_list.extend(ids)
        self._metadatas.extend(list(metadatas) if metadatas else [{} for _ in ids])
        self._documents.extend(list(documents) if documents else ["" for _ in ids])

    def _train_if_needed(self, embeddings: np.ndarray) -> None:
        """Train an IVF index, refusing the case faiss reports as a bare crash."""
        if not (hasattr(self._index, "is_trained") and not self._index.is_trained):
            return

        if embeddings.shape[0] < self.nlist:
            raise ValueError(
                f"index_type='ivf' with nlist={self.nlist} needs at least "
                f"{self.nlist} vectors to train, but got {embeddings.shape[0]}. "
                f"Use index_type='flat' for a collection this size, or reduce "
                f"nlist to at most {embeddings.shape[0]}."
            )

        self._index.train(embeddings)

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._index is None or self._index.ntotal == 0:
            return _build_result(self.metric, self.direction, [], [], [], [], [])

        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32,
        )

        # FAISS has no metadata predicate, so filtering happens after the
        # search; over-fetch to the whole index so a selective filter cannot
        # return fewer rows than the caller asked for.
        k = self._index.ntotal if where else min(n_results, self._index.ntotal)
        distances, indices = self._index.search(query_embedding, k)

        result_ids: List[str] = []
        result_dists: List[float] = []
        result_meta: List[Dict[str, Any]] = []
        result_docs: List[str] = []

        # Keep each distance with its own index: faiss pads missing slots with
        # -1, and slicing the distances separately could misalign the pairs.
        for idx, dist in zip(indices[0].tolist(), distances[0].tolist()):
            if not (0 <= idx < len(self._id_list)):
                continue
            metadata = self._metadatas[idx]
            if not _matches(metadata, where):
                continue
            result_ids.append(self._id_list[idx])
            result_dists.append(dist)
            result_meta.append(metadata)
            result_docs.append(self._documents[idx])
            if len(result_ids) >= n_results:
                break

        return _build_result(
            self.metric,
            self.direction,
            self._to_similarity(result_dists),
            result_ids,
            result_dists,
            result_meta,
            result_docs,
        )

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def delete(self, ids: List[str]) -> None:
        # FAISS does not support random deletion natively; rebuild.
        keep = [i for i, id_ in enumerate(self._id_list) if id_ not in set(ids)]
        if not keep or self._index is None:
            self._id_list.clear()
            self._metadatas.clear()
            self._documents.clear()
            self._index = None
            return

        # An IVF index cannot reconstruct by id until a direct map exists.
        if hasattr(self._index, "make_direct_map"):
            try:
                self._index.make_direct_map()
            except Exception:  # pragma: no cover - flat indexes do not need it
                pass

        vecs = np.vstack([self._index.reconstruct(i) for i in keep])
        self._id_list = [self._id_list[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._build_index(vecs.shape[1])
        self._train_if_needed(np.ascontiguousarray(vecs, dtype=np.float32))
        self._index.add(vecs)

    def persist(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the FAISS index and metadata to disk."""
        if path is None:
            return

        import faiss
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self._id_list,
                    "metadatas": self._metadatas,
                    "documents": self._documents,
                    # Without these the reloaded store fell back to the default
                    # metric, so it reported the wrong direction for numbers the
                    # index was still computing under the original metric.
                    "metric": self.metric,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                },
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSVectorStore":
        """Load a previously persisted FAISS store."""
        import faiss
        import json

        path = Path(path)
        store = cls()
        store._index = faiss.read_index(str(path / "index.faiss"))
        store.dimension = store._index.d
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        store._id_list = meta["ids"]
        store._metadatas = meta["metadatas"]
        store._documents = meta["documents"]
        # Older files predate these keys; fall back to the constructor defaults.
        store.metric = meta.get("metric", store.metric)
        store.index_type = meta.get("index_type", store.index_type)
        store.nlist = meta.get("nlist", store.nlist)
        return store


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------
class VectorDB:
    """
    High-level vector database builder and search interface.

    Orchestrates an :class:`ImageEmbedder` and a :class:`BaseVectorStore`
    to build / query a searchable index from GeoAI-VLM pipeline output.

    Args:
        embedder: An :class:`~geoai_vlm.embedding.ImageEmbedder` instance
            (or ``None`` to defer embedding to the caller).
        store_backend: ``"chromadb"`` or ``"faiss"``.
        **store_kwargs: Extra keyword arguments forwarded to the store backend
            constructor (e.g. ``persist_directory``, ``collection_name``).

    Example:
        >>> from geoai_vlm import ImageEmbedder, VectorDB
        >>> embedder = ImageEmbedder()
        >>> vdb = VectorDB(embedder=embedder, store_backend="chromadb",
        ...                persist_directory="./my_db")
        >>> vdb.build(gdf, text_column="scene_narrative", image_dir="./images")
        >>> results = vdb.search(query_text="busy commercial street")
    """

    def __init__(
        self,
        embedder=None,
        store_backend: str = "chromadb",
        **store_kwargs,
    ):
        self.embedder = embedder
        self.store_backend = store_backend
        self.store_kwargs = store_kwargs

        self._store: Optional[BaseVectorStore] = None

    @property
    def store(self) -> BaseVectorStore:
        """Get or create the vector store."""
        if self._store is not None:
            return self._store

        if self.store_backend == "chromadb":
            self._store = ChromaVectorStore(**self.store_kwargs)
        elif self.store_backend == "faiss":
            self._store = FAISSVectorStore(**self.store_kwargs)
        else:
            raise ValueError(f"Unknown store backend: {self.store_backend}")
        return self._store

    def build(
        self,
        gdf,
        text_column: Optional[str] = "scene_narrative",
        image_dir: Optional[Union[str, Path]] = None,
        metadata_columns: Optional[List[str]] = None,
        id_column: str = "image_id",
        batch_size: int = 32,
    ) -> BaseVectorStore:
        """
        Build a vector index from a GeoDataFrame.

        Either *text_column* (text-only embedding) or *image_dir* (image embedding)
        should be provided.  When both are given the embedder receives
        multimodal ``{text, image}`` dicts.

        Args:
            gdf: GeoDataFrame produced by the GeoAI-VLM pipeline.
            text_column: Column with text to embed.
            image_dir: Directory containing the original images
                (file names must match *id_column* values).
            metadata_columns: Extra columns to store alongside each vector.
            id_column: Column used as the vector ID.
            batch_size: Batch size for embedding.

        Returns:
            The populated :class:`BaseVectorStore`.
        """
        import geopandas as gpd

        if self.embedder is None:
            raise RuntimeError(
                "No embedder set.  Pass an ImageEmbedder instance to VectorDB()."
            )

        # Build input dicts
        inputs: List[Dict[str, Any]] = []
        for _, row in gdf.iterrows():
            inp: Dict[str, Any] = {}
            if text_column and text_column in gdf.columns:
                inp["text"] = str(row[text_column])
            if image_dir is not None:
                img_id = str(row[id_column])
                # Try common extensions
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = Path(image_dir) / f"{img_id}{ext}"
                    if candidate.exists():
                        inp["image"] = str(candidate)
                        break
            inputs.append(inp)

        # Embed
        embeddings = self.embedder._embed_batched(
            inputs, self.embedder.instruction, batch_size,
        )

        # IDs
        ids = [str(v) for v in gdf[id_column]]

        # Metadata
        meta_cols = metadata_columns or []
        # Always include lat/lon if available
        if "lat" in gdf.columns and "lat" not in meta_cols:
            meta_cols = ["lat", "lon"] + meta_cols
        elif isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry is not None:
            gdf = gdf.copy()
            if "lat" not in gdf.columns:
                gdf["lat"] = gdf.geometry.y
                gdf["lon"] = gdf.geometry.x
            meta_cols = ["lat", "lon"] + meta_cols

        metadatas = None
        if meta_cols:
            metadatas = []
            for _, row in gdf.iterrows():
                meta = {}
                for col in meta_cols:
                    if col in gdf.columns:
                        val = row[col]
                        # ChromaDB only supports str/int/float/bool
                        if isinstance(val, (int, float, bool, str)):
                            meta[col] = val
                        else:
                            meta[col] = str(val)
                metadatas.append(meta)

        # Documents (the text used for embedding)
        documents = None
        if text_column and text_column in gdf.columns:
            documents = [str(v) for v in gdf[text_column]]

        self.store.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        self.store.persist()

        print(f"Built vector index with {self.store.count()} vectors.")
        return self.store

    def search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, Path]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Search for similar items, best match first.

        Args:
            query_text: Text query string.
            query_image: Path to a query image.
            n_results: Number of results to return.
            where: Optional equality filter over stored metadata,
                e.g. ``{"land_use_primary": "commercial"}``.

        Returns:
            DataFrame with columns ``id``, ``rank``, ``similarity``,
            ``distance`` and any stored metadata, already ordered best-first.

            Rank by ``similarity`` (higher is always closer) or by ``rank``.
            ``distance`` is the backend's raw score, whose direction depends on
            the metric -- sorting by it ascending reverses the ranking on a
            FAISS inner-product index. The metric and its direction are
            reported in ``df.attrs["metric"]`` and ``df.attrs["direction"]``.
        """
        if self.embedder is None:
            raise RuntimeError("No embedder set.")

        inp: Dict[str, Any] = {}
        if query_text is not None:
            inp["text"] = query_text
        if query_image is not None:
            inp["image"] = str(query_image)

        query_emb = self.embedder.backend.embed([inp], instruction=self.embedder.instruction)

        results = self.store.query(query_emb[0], n_results=n_results, where=where)

        rows = []
        for i, id_ in enumerate(results["ids"]):
            row = {
                "id": id_,
                "rank": results["rank"][i],
                "similarity": results["similarity"][i],
                "distance": results["distances"][i],
            }
            if results["metadatas"] and i < len(results["metadatas"]):
                row.update(results["metadatas"][i])
            rows.append(row)

        df = pd.DataFrame(rows, columns=None if rows else ["id", "rank", "similarity", "distance"])
        df.attrs["metric"] = results["metric"]
        df.attrs["direction"] = results["direction"]
        return df
