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
]


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
    ) -> Dict[str, Any]:
        """
        Query the store for nearest neighbours.

        Returns:
            Dict with keys ``ids``, ``distances``, ``metadatas``, ``documents``.
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

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        """Query nearest neighbours."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
        )
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
        }

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

    def _build_index(self, dim: int):
        """Build the FAISS index."""
        import faiss

        if self.metric == "ip":
            if self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFFlat(quantizer, dim, self.nlist)
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

        self.dimension = dim

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self._index is None:
            self._build_index(embeddings.shape[1])

        # Train index if required (IVF)
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._id_list.extend(ids)
        self._metadatas.extend(metadatas or [{} for _ in ids])
        self._documents.extend(documents or ["" for _ in ids])

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        if self._index is None or self._index.ntotal == 0:
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}

        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32,
        )
        distances, indices = self._index.search(query_embedding, n_results)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        result_ids = []
        result_meta = []
        result_docs = []
        for idx in indices:
            if 0 <= idx < len(self._id_list):
                result_ids.append(self._id_list[idx])
                result_meta.append(self._metadatas[idx])
                result_docs.append(self._documents[idx])

        return {
            "ids": result_ids,
            "distances": distances[: len(result_ids)],
            "metadatas": result_meta,
            "documents": result_docs,
        }

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

        import faiss

        # Reconstruct vectors
        vecs = np.vstack([self._index.reconstruct(i) for i in keep])
        self._id_list = [self._id_list[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._build_index(vecs.shape[1])
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
    ) -> pd.DataFrame:
        """
        Search for similar items.

        Args:
            query_text: Text query string.
            query_image: Path to a query image.
            n_results: Number of results to return.

        Returns:
            DataFrame with columns ``id``, ``distance``, and any stored metadata.
        """
        if self.embedder is None:
            raise RuntimeError("No embedder set.")

        inp: Dict[str, Any] = {}
        if query_text is not None:
            inp["text"] = query_text
        if query_image is not None:
            inp["image"] = str(query_image)

        query_emb = self.embedder.backend.embed([inp], instruction=self.embedder.instruction)

        results = self.store.query(query_emb[0], n_results=n_results)

        rows = []
        for i, id_ in enumerate(results["ids"]):
            row = {"id": id_, "distance": results["distances"][i]}
            if results["metadatas"] and i < len(results["metadatas"]):
                row.update(results["metadatas"][i])
            rows.append(row)

        return pd.DataFrame(rows)
