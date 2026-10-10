# -*- coding: utf-8 -*-
"""
Build a small demo SceneIndex from a published street-level dataset.

Reads only the columns it needs from the published Hugging Face dataset
``yunusserhat/fatih`` -- image ids, coordinates, capture metadata and the
model-generated scene narratives -- for the first ``--n`` described images,
embeds the narratives with a CLIP-family text encoder, and writes a local
index for ``geoai-vlm app``. No image is downloaded.

Data and attribution
--------------------
* Dataset: yunusserhat/fatih on the Hugging Face Hub, CC BY-SA 4.0,
  doi:10.57967/hf/10144. Street-level imagery and its metadata originate from
  Mapillary contributors (CC BY-SA 4.0).
* The narratives are model-generated (see the dataset card), not
  human-verified observations.
* Keep this attribution with anything you derive from the index; the index
  metadata carries it, together with the dataset revision that was read.

The dataset id is a parameter: any dataset with the same layer layout works,
and nothing in GeoAI-VLM is tied to one place.

Retrieval quality depends on the embedding model. CLIP-family text encoders
are trained to match images with captions and separate similar captions
poorly (similar narratives all score close to each other), so for question
answering over descriptions a multimodal embedding model such as
Qwen3-VL-Embedding (``--embedding-backend transformers``) is the better
choice where hardware allows.

Usage::

    pip install 'geoai-vlm[transformers]'
    python examples/build_demo_index.py --n 300 --out ./demo_index
    geoai-vlm app --index ./demo_index
"""

from __future__ import annotations

import argparse

import pandas as pd
import pyarrow.parquet as pq


DESCRIPTIONS = "data/derived/descriptions/geoai/v1/part-00000.parquet"
MANIFEST = "data/raw/manifest/train.parquet"
ATTRIBUTION = (
    "Descriptions and metadata: yunusserhat/fatih (Hugging Face), CC BY-SA 4.0, "
    "doi:10.57967/hf/10144. Imagery and image metadata: Mapillary contributors, "
    "CC BY-SA 4.0. Scene narratives are model-generated."
)


def _read_columns(fs, path: str, columns, n=None) -> pd.DataFrame:
    """Read selected columns, stopping after *n* rows when given."""
    with fs.open(path, "rb", block_size=1 << 20) as handle:
        parquet = pq.ParquetFile(handle)
        frames, total = [], 0
        for batch in parquet.iter_batches(batch_size=min(n or 65536, 65536), columns=list(columns)):
            frames.append(batch.to_pandas())
            total += batch.num_rows
            if n is not None and total >= n:
                break
    frame = pd.concat(frames, ignore_index=True)
    return frame.head(n) if n is not None else frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="yunusserhat/fatih")
    parser.add_argument("--n", type=int, default=300, help="number of described images")
    parser.add_argument("--out", default="demo_index")
    parser.add_argument(
        "--embedding-backend",
        default="clip",
        help="clip (default, small, CPU-friendly) or transformers (Qwen3-VL-Embedding)",
    )
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    from geoai_vlm import ImageEmbedder
    from geoai_vlm.service import SceneIndex

    revision = HfApi().dataset_info(args.dataset).sha
    fs = HfFileSystem()
    root = f"datasets/{args.dataset}@{revision}"

    described = _read_columns(
        fs,
        f"{root}/{DESCRIPTIONS}",
        ["image_id", "scene_narrative", "land_use_primary", "street_type", "usable"],
        n=args.n,
    )
    described = described[described["scene_narrative"].fillna("").str.len() > 0]
    manifest = _read_columns(fs, f"{root}/{MANIFEST}", ["image_id", "lat", "lon", "sequence_id", "captured_at"])
    records = described.merge(manifest, on="image_id", how="inner").dropna(subset=["lat", "lon"])
    records["attribution"] = ATTRIBUTION

    embedder = ImageEmbedder(backend=args.embedding_backend, model_name=args.embedding_model)
    index = SceneIndex.build(
        records,
        embedder,
        text_column="scene_narrative",
        metadata={
            "dataset": args.dataset,
            "dataset_revision": revision,
            "license": "CC-BY-SA-4.0",
            "doi": "10.57967/hf/10144",
            "attribution": ATTRIBUTION,
            "embedding_backend": args.embedding_backend,
            "embedding_model": embedder.model_name,
        },
    )
    index.save(args.out)
    print(f"indexed {len(index)} scenes from {args.dataset}@{revision[:8]} -> {args.out}")
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
