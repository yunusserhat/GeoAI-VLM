# -*- coding: utf-8 -*-
"""
API contract tests.

These pin the public signatures that the README and the example notebooks
call. The 2026-09-12 review found several documented calls that would raise
``TypeError`` against the real implementation (``save_path=`` on the plotting
helpers, ``output_dir=`` on ``generate_report``, ``embedding_model=`` on
``embed_place``). A documented call that cannot run is a defect in its own
right, so the contract is asserted here rather than left to the reader.

No model, network or API key is used.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _params(func) -> set:
    return set(inspect.signature(func).parameters)


def _accepts_kwargs(func) -> bool:
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(func).parameters.values()
    )


def _required_params(func) -> set:
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


# ---------------------------------------------------------------------------
# Documented keyword arguments must exist
# ---------------------------------------------------------------------------
DOCUMENTED_KWARGS = [
    ("geoai_vlm.visualization", "plot_elbow_curve", {"k_values", "inertias", "save_path"}),
    ("geoai_vlm.visualization", "plot_cluster_map", {"gdf", "cluster_column", "save_path"}),
    ("geoai_vlm.visualization", "plot_lisa_map", {"gdf", "save_path"}),
    (
        "geoai_vlm.visualization",
        "plot_category_distribution",
        {"gdf", "category_columns"},
    ),
    ("geoai_vlm.visualization", "generate_report", {"gdf", "output_path"}),
    (
        "geoai_vlm.pipeline",
        "embed_place",
        {"place_name", "mly_api_key", "model_name", "embedding_modality"},
    ),
    (
        "geoai_vlm.pipeline",
        "cluster_descriptions",
        {"gdf_or_path", "n_clusters", "embedding_model"},
    ),
    ("geoai_vlm.pipeline", "analyze_spatial", {"gdf", "column", "k_neighbors"}),
    (
        "geoai_vlm.pipeline",
        "describe_query",
        {"query", "mly_api_key", "output_dir", "output_path", "export_formats"},
    ),
    (
        "geoai_vlm.io",
        "merge_metadata_and_descriptions",
        {"metadata_gdf", "descriptions_df", "on", "on_duplicate"},
    ),
    (
        "geoai_vlm.describer",
        "ImageDescriber.describe",
        {"image_dir", "image_paths", "output_path", "resume"},
    ),
]


@pytest.mark.parametrize(
    "module_name,qualname,expected",
    DOCUMENTED_KWARGS,
    ids=[f"{m.split('.')[-1]}.{q}" for m, q, _ in DOCUMENTED_KWARGS],
)
def test_documented_keywords_exist(module_name, qualname, expected):
    import importlib

    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)

    actual = _params(obj)
    missing = expected - actual
    assert not missing, (
        f"{module_name}.{qualname} does not accept documented argument(s): "
        f"{sorted(missing)}. Present: {sorted(actual)}"
    )


# ---------------------------------------------------------------------------
# generate_report must be callable the way the README shows it
# ---------------------------------------------------------------------------
def test_generate_report_has_no_hidden_required_arguments():
    """The README calls generate_report(gdf, output_path=...) with nothing else."""
    from geoai_vlm.visualization import generate_report

    required = _required_params(generate_report)
    assert required <= {"gdf"}, (
        "generate_report requires arguments the documentation never supplies: "
        f"{sorted(required - {'gdf'})}"
    )


def test_generate_report_returns_markdown_not_html(sample_gdf, tmp_path):
    """The README described this as an HTML report; it produces Markdown."""
    from geoai_vlm.visualization import generate_report

    gdf = sample_gdf.copy()
    # generate_report summarises clusters, so it needs a cluster column.
    gdf["cluster"] = [i % 3 for i in range(len(gdf))]

    out = tmp_path / "report.md"
    text = generate_report(gdf, output_path=out)
    assert isinstance(text, str)
    assert "<html" not in text.lower(), "report claims to be Markdown"
    assert out.exists()


# ---------------------------------------------------------------------------
# The README must not show calls that cannot run
# ---------------------------------------------------------------------------
def test_readme_python_blocks_use_real_keywords():
    """Scan README python blocks for calls into the package's public API.

    For every ``name(...)`` call whose name is exported by the package, every
    ``keyword=`` used must exist in the real signature (unless the function
    takes ``**kwargs``). This is a syntactic check, not an execution check.
    """
    import geoai_vlm

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks, "no python examples found in README"

    problems = []
    for block in blocks:
        for match in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(([^()]*)\)", block):
            name, args = match.group(1), match.group(2)
            func = getattr(geoai_vlm, name, None)
            if func is None or not callable(func):
                continue
            if _accepts_kwargs(func):
                continue
            used = set(re.findall(r"(\w+)\s*=", args))
            unknown = used - _params(func)
            if unknown:
                problems.append(f"{name}(): unknown argument(s) {sorted(unknown)}")

    assert not problems, "README shows calls that would raise TypeError:\n" + "\n".join(
        sorted(set(problems))
    )


# ---------------------------------------------------------------------------
# Notebook calls must match too
# ---------------------------------------------------------------------------
def test_demo_notebook_plot_calls_match_signature():
    """demo_new_features.ipynb called plot_elbow_curve with a k_range argument."""
    import json

    import geoai_vlm

    nb_path = REPO_ROOT / "examples" / "demo_new_features.ipynb"
    if not nb_path.exists():
        pytest.skip("example notebook not present")

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    problems = []
    for match in re.finditer(r"\b(plot_\w+|generate_report)\s*\(([^()]*)\)", source):
        name, args = match.group(1), match.group(2)
        func = getattr(geoai_vlm, name, None)
        if func is None or _accepts_kwargs(func):
            continue
        unknown = set(re.findall(r"(\w+)\s*=", args)) - _params(func)
        if unknown:
            problems.append(f"{name}(): unknown argument(s) {sorted(unknown)}")

    assert not problems, "notebook calls do not match signatures:\n" + "\n".join(
        sorted(set(problems))
    )
