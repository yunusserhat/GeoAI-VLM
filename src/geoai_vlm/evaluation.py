# -*- coding: utf-8 -*-
"""
Evaluation tools for GeoAI-VLM
==============================
Tools for checking street-level measurements against references and against
themselves. Nothing here produces a finding on its own; it gives the numbers
and their uncertainty so that a finding can be checked.

* **Splits** that cannot leak: sequence-disjoint and spatial-block splits,
  with a combined mode in which neither a capture sequence nor a block spans
  train and test, and :func:`check_split_leakage` to verify any split.
* **Agreement**: percent agreement and Cohen's kappa (unweighted or
  weighted) for categories; ICC(2,1) and ICC(3,1) for continuous values;
  R-squared and Bland-Altman limits between two measurement methods (for
  example two segmentation models).
* **Uncertainty**: a cluster bootstrap that resamples whole sequences or
  blocks, since images from one drive or one block are not independent.
* **Generation stability**: run the same images several times with sampling
  on and report, per field, how often the repeats agree.
* **Reference sampling**: a seeded, quota-based stratified sample for manual
  labelling, with a daylight proxy computed from capture time and solar
  elevation.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


__all__ = [
    "SplitLeakageError",
    "grouped_split",
    "sequence_split",
    "assign_spatial_blocks",
    "spatial_block_split",
    "check_split_leakage",
    "percent_agreement",
    "cohen_kappa",
    "icc",
    "r_squared",
    "bland_altman",
    "cluster_bootstrap_ci",
    "agreement_report",
    "repeated_generation",
    "generation_stability",
    "solar_elevation",
    "light_condition",
    "stratified_reference_sample",
]


# =============================================================================
# Splits
# =============================================================================
class SplitLeakageError(ValueError):
    """A sequence, block or other group appears on both sides of a split."""


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def grouped_split(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    test_size: float = 0.2,
    seed: int = 0,
    split_column: str = "split",
) -> pd.DataFrame:
    """Train/test split that keeps every group on one side.

    Rows sharing a value in *any* of ``group_columns`` are linked, and the
    linked components are assigned whole. With
    ``group_columns=["sequence_id", "block_id"]`` a drive that crosses two
    blocks pulls both blocks to the same side, so neither leaks. A missing
    group value links nothing (the row is its own group for that column).

    Components are shuffled with ``seed`` and added to the test side until it
    holds at least ``test_size`` of the rows.

    Returns:
        A copy with ``split_column`` (``"train"``/``"test"``) and
        ``split_component``. ``attrs["split"]`` records the seed, the
        requested and achieved test share and the number of components.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise ValueError(f"group columns not found: {missing}")

    n = len(df)
    uf = _UnionFind(n)
    for column in group_columns:
        first_row: Dict[Any, int] = {}
        for i, value in enumerate(df[column].tolist()):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            if value in first_row:
                uf.union(first_row[value], i)
            else:
                first_row[value] = i
    components = np.array([uf.find(i) for i in range(n)])
    unique = np.unique(components)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    sizes = pd.Series(components).value_counts()

    target = test_size * n
    test_components, test_rows = set(), 0
    for component in order:
        if test_rows >= target:
            break
        test_components.add(component)
        test_rows += int(sizes[component])

    out = df.copy()
    out["split_component"] = components
    out[split_column] = np.where(np.isin(components, list(test_components)), "test", "train")
    out.attrs["split"] = {
        "group_columns": list(group_columns),
        "seed": seed,
        "test_size_requested": test_size,
        "test_size_achieved": test_rows / n if n else float("nan"),
        "n_components": int(len(unique)),
    }
    return out


def sequence_split(
    df: pd.DataFrame,
    sequence_column: str = "sequence_id",
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Sequence-disjoint split: no capture sequence spans train and test."""
    if sequence_column in df.columns and df[sequence_column].isna().any():
        warnings.warn(
            f"{int(df[sequence_column].isna().sum())} row(s) have no {sequence_column}; "
            "each is treated as its own sequence",
            stacklevel=2,
        )
    return grouped_split(df, [sequence_column], test_size=test_size, seed=seed)


def assign_spatial_blocks(
    gdf,
    block_size_m: float = 500.0,
    column: str = "block_id",
    metric_crs: Any = None,
):
    """Label each point with the square grid block (``block_size_m``) it falls in.

    Blocks are aligned to the origin of a projected CRS (the data's own, or an
    estimated UTM zone), so the same point always gets the same block.
    """
    if block_size_m <= 0:
        raise ValueError("block_size_m must be > 0")
    if gdf.crs is None:
        raise ValueError("points need a CRS")
    crs = metric_crs or (gdf.crs if gdf.crs.is_projected else gdf.estimate_utm_crs())
    projected = gdf.to_crs(crs)
    cx = np.floor(projected.geometry.x / block_size_m).astype(int)
    cy = np.floor(projected.geometry.y / block_size_m).astype(int)
    out = gdf.copy()
    out[column] = [f"{x}_{y}" for x, y in zip(cx, cy)]
    out.attrs["blocks"] = {"block_size_m": block_size_m, "crs": str(crs)}
    return out


def spatial_block_split(
    gdf,
    block_size_m: float = 500.0,
    test_size: float = 0.2,
    seed: int = 0,
    sequence_column: Optional[str] = None,
    metric_crs: Any = None,
):
    """Spatial-block split; optionally also sequence-disjoint.

    Args:
        gdf: Image points with a CRS.
        block_size_m: Block edge length in metres.
        test_size: Target share of rows in the test split.
        seed: Random seed (recorded in ``attrs``).
        sequence_column: When given, sequences are kept whole as well, so
            neither a block nor a sequence spans the split.
        metric_crs: Projected CRS for the grid.
    """
    blocked = assign_spatial_blocks(gdf, block_size_m=block_size_m, metric_crs=metric_crs)
    groups = ["block_id"] + ([sequence_column] if sequence_column else [])
    out = grouped_split(blocked, groups, test_size=test_size, seed=seed)
    out.attrs["split"]["block_size_m"] = block_size_m
    return out


def check_split_leakage(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    split_column: str = "split",
    raise_on_leak: bool = True,
) -> Dict[str, List[Any]]:
    """Find group values present on more than one side of a split.

    Returns:
        ``{column: [leaking values]}`` (empty lists when clean).

    Raises:
        SplitLeakageError: When anything leaks and ``raise_on_leak`` is True.
    """
    report: Dict[str, List[Any]] = {}
    for column in group_columns:
        if column not in df.columns:
            raise ValueError(f"column {column!r} not found")
        sides = df.dropna(subset=[column]).groupby(column)[split_column].nunique()
        report[column] = sorted(sides[sides > 1].index.tolist(), key=str)
    leaks = {c: v for c, v in report.items() if v}
    if leaks and raise_on_leak:
        summary = "; ".join(f"{c}: {len(v)} value(s), e.g. {v[:3]}" for c, v in leaks.items())
        raise SplitLeakageError(f"groups span the split -- {summary}")
    return report


# =============================================================================
# Agreement
# =============================================================================
def _pairs(a, b) -> pd.DataFrame:
    frame = pd.DataFrame({"a": list(a), "b": list(b)})
    if len(frame) == 0:
        raise ValueError("no observations")
    return frame.dropna()


def percent_agreement(a: Iterable, b: Iterable) -> float:
    """Share of paired observations on which two raters agree (missing pairs dropped)."""
    pairs = _pairs(a, b)
    if len(pairs) == 0:
        return float("nan")
    return float((pairs["a"] == pairs["b"]).mean())


def cohen_kappa(
    a: Iterable,
    b: Iterable,
    labels: Optional[Sequence[Any]] = None,
    weights: Optional[str] = None,
) -> float:
    """Cohen's kappa between two raters (missing pairs dropped).

    Args:
        a, b: Paired categorical ratings.
        labels: Category order. Required for weighted kappa (ordinal
            categories); by default the sorted union of observed values.
        weights: ``None`` (unweighted), ``"linear"`` or ``"quadratic"``.

    Returns:
        Kappa, or NaN when chance agreement is 1 (both raters used a single
        identical category), where kappa is undefined.
    """
    pairs = _pairs(a, b)
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    if labels is None:
        if weights is not None:
            raise ValueError("weighted kappa needs an explicit ordinal label order")
        labels = sorted(set(pairs["a"]) | set(pairs["b"]), key=str)
    index = {label: i for i, label in enumerate(labels)}
    unknown = (set(pairs["a"]) | set(pairs["b"])) - set(index)
    if unknown:
        raise ValueError(f"values not in labels: {sorted(unknown, key=str)}")
    k = len(labels)
    observed = np.zeros((k, k))
    for x, y in zip(pairs["a"], pairs["b"]):
        observed[index[x], index[y]] += 1
    n = observed.sum()
    if n == 0:
        return float("nan")
    observed /= n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weights is None:
        w = 1.0 - np.eye(k)
    else:
        grid = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / max(k - 1, 1)
        w = grid if weights == "linear" else grid ** 2
    denominator = (w * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (w * observed).sum() / denominator)


def icc(data: Union[np.ndarray, pd.DataFrame], kind: str = "ICC(2,1)") -> float:
    """Intraclass correlation for a complete subjects x raters table.

    Two-way ANOVA estimates (Shrout & Fleiss, 1979):

    * ``ICC(2,1)`` -- two-way random effects, absolute agreement, single
      rater: ``(MSR - MSE) / (MSR + (k-1) MSE + k (MSC - MSE) / n)``
    * ``ICC(3,1)`` -- two-way mixed effects, consistency, single rater:
      ``(MSR - MSE) / (MSR + (k-1) MSE)``

    Rows with any missing value are dropped.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("data must be a 2-D subjects x raters table with at least 2 raters")
    values = values[~np.isnan(values).any(axis=1)]
    n, k = values.shape
    if n < 2:
        raise ValueError("ICC needs at least 2 complete subjects")
    grand = values.mean()
    ss_rows = k * ((values.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((values.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((values - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    if kind == "ICC(2,1)":
        denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    elif kind == "ICC(3,1)":
        denominator = msr + (k - 1) * mse
    else:
        raise ValueError("kind must be 'ICC(2,1)' or 'ICC(3,1)'")
    if denominator == 0:
        return float("nan")
    return float((msr - mse) / denominator)


def r_squared(x: Iterable[float], y: Iterable[float], method: str = "pearson") -> float:
    """R-squared between two measurement methods (missing pairs dropped).

    ``"pearson"`` is the squared Pearson correlation: symmetric, and blind to
    a constant offset or scale difference. ``"identity"`` is
    ``1 - sum((y - x)^2) / sum((y - mean(y))^2)``: agreement of *y* with *x*
    along the identity line, which does penalise bias. Report the
    Bland-Altman limits alongside either.
    """
    pairs = _pairs(x, y).astype(float)
    if len(pairs) < 2:
        return float("nan")
    xs, ys = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    if method == "pearson":
        if xs.std() == 0 or ys.std() == 0:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1] ** 2)
    if method == "identity":
        total = ((ys - ys.mean()) ** 2).sum()
        if total == 0:
            return float("nan")
        return float(1.0 - ((ys - xs) ** 2).sum() / total)
    raise ValueError("method must be 'pearson' or 'identity'")


def bland_altman(a: Iterable[float], b: Iterable[float], z: float = 1.96) -> Dict[str, float]:
    """Bland-Altman bias and limits of agreement for ``a - b``.

    Returns:
        ``bias`` (mean difference), ``sd`` (sample SD of the differences),
        ``loa_lower`` / ``loa_upper`` (``bias -/+ z * sd``) and ``n``.
    """
    pairs = _pairs(a, b).astype(float)
    diff = pairs["a"] - pairs["b"]
    n = len(diff)
    bias = float(diff.mean()) if n else float("nan")
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": bias - z * sd,
        "loa_upper": bias + z * sd,
        "n": n,
    }


def cluster_bootstrap_ci(
    metric: Callable[[pd.DataFrame], float],
    data: pd.DataFrame,
    cluster_column: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile bootstrap CI that resamples whole clusters.

    Images from one capture sequence or one spatial block are correlated;
    resampling rows as if independent understates uncertainty. Here the
    clusters (sequences, blocks) are drawn with replacement and the metric
    is recomputed on their pooled rows.

    Args:
        metric: Function of a DataFrame returning a number.
        data: The rows the metric is computed on.
        cluster_column: Cluster id column (rows without one are dropped).
        n_boot: Bootstrap replicates.
        ci: Confidence level.
        seed: Random seed (recorded).

    Returns:
        ``estimate``, ``ci_low``, ``ci_high``, ``n_boot``, ``n_valid``
        (replicates with a finite value), ``n_clusters``, ``seed``, ``ci`` and
        ``method``.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    data = data.dropna(subset=[cluster_column])
    groups = {key: frame for key, frame in data.groupby(cluster_column, sort=True)}
    keys = list(groups)
    estimate = float(metric(data))
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(keys), size=len(keys))
        sample = pd.concat([groups[keys[i]] for i in draw], ignore_index=True)
        value = metric(sample)
        if value is not None and np.isfinite(value):
            replicates.append(float(value))
    alpha = (1 - ci) / 2
    low, high = (
        (float(np.quantile(replicates, alpha)), float(np.quantile(replicates, 1 - alpha)))
        if replicates
        else (float("nan"), float("nan"))
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "n_boot": n_boot,
        "n_valid": len(replicates),
        "n_clusters": len(keys),
        "seed": seed,
        "ci": ci,
        "method": f"cluster percentile bootstrap over {cluster_column!r}",
    }


def agreement_report(
    df: pd.DataFrame,
    a_column: str,
    b_column: str,
    kind: str = "categorical",
    cluster_column: Optional[str] = None,
    labels: Optional[Sequence[Any]] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Agreement metrics between two columns, with cluster-bootstrap CIs.

    ``kind="categorical"``: percent agreement and Cohen's kappa.
    ``kind="continuous"``: ICC(2,1), ICC(3,1), Pearson R-squared, Bland-Altman
    bias and limits. CIs are reported when ``cluster_column`` is given.
    """
    if kind == "categorical":
        metrics = {
            "percent_agreement": lambda d: percent_agreement(d[a_column], d[b_column]),
            "cohen_kappa": lambda d: cohen_kappa(d[a_column], d[b_column], labels=labels),
        }
    elif kind == "continuous":
        metrics = {
            "icc_2_1": lambda d: icc(d[[a_column, b_column]].to_numpy(dtype=float), "ICC(2,1)"),
            "icc_3_1": lambda d: icc(d[[a_column, b_column]].to_numpy(dtype=float), "ICC(3,1)"),
            "r_squared": lambda d: r_squared(d[a_column], d[b_column]),
            "bland_altman_bias": lambda d: bland_altman(d[a_column], d[b_column])["bias"],
            "bland_altman_loa_lower": lambda d: bland_altman(d[a_column], d[b_column])["loa_lower"],
            "bland_altman_loa_upper": lambda d: bland_altman(d[a_column], d[b_column])["loa_upper"],
        }
    else:
        raise ValueError("kind must be 'categorical' or 'continuous'")

    rows = []
    for name, fn in metrics.items():
        row: Dict[str, Any] = {"metric": name, "estimate": fn(df), "n": int(df[[a_column, b_column]].dropna().shape[0])}
        if cluster_column:
            boot = cluster_bootstrap_ci(fn, df, cluster_column, n_boot=n_boot, seed=seed)
            row.update(ci_low=boot["ci_low"], ci_high=boot["ci_high"], n_clusters=boot["n_clusters"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["agreement"] = {"a": a_column, "b": b_column, "kind": kind, "cluster_column": cluster_column, "seed": seed}
    return out


# =============================================================================
# Repeated-generation stability
# =============================================================================
def repeated_generation(
    describer,
    image_paths: Sequence[Any],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Describe the same images ``n_repeats`` times; one row per image and repeat.

    Sampling must be on (``temperature > 0``) for this to say anything: with
    greedy decoding the repeats agree by construction, and a fixed sampling
    seed makes them identical. Both cases raise a warning.
    """
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")
    backend = describer.backend
    temperature = getattr(backend, "temperature", None)
    if temperature is not None and temperature <= 0:
        warnings.warn(
            "temperature is 0: decoding is greedy and repeats agree trivially; "
            "set temperature > 0 to measure generation stability",
            stacklevel=2,
        )
    if getattr(backend, "seed", None) is not None and (temperature or 0) > 0:
        warnings.warn(
            "a fixed sampling seed can make repeated generations identical",
            stacklevel=2,
        )
    runs = []
    for repeat in range(n_repeats):
        frame = describer.describe(image_paths=list(image_paths), resume=False)
        frame = frame.copy()
        frame["repeat"] = repeat
        runs.append(frame)
    return pd.concat(runs, ignore_index=True)


def generation_stability(
    runs: pd.DataFrame,
    fields: Sequence[str],
    id_column: str = "image_id",
    repeat_column: str = "repeat",
) -> pd.DataFrame:
    """Per-field agreement across repeated generations.

    For each image and field the modal share is the fraction of repeats that
    gave the most common value (a missing value counts as its own value).

    Returns:
        One row per field: ``n_images``, ``n_repeats`` (median per image),
        ``mean_modal_share``, ``min_modal_share``, ``share_fully_stable``
        (images whose repeats all agree) and ``mean_n_distinct``.
    """
    missing = [f for f in fields if f not in runs.columns]
    if missing:
        raise ValueError(f"fields not found: {missing}")
    rows = []
    for field in fields:
        per_image = []
        for _, group in runs.groupby(id_column, sort=True):
            values = group[field].astype(object).where(group[field].notna(), "<missing>")
            counts = values.value_counts()
            per_image.append(
                {"modal_share": counts.iloc[0] / len(values), "n_distinct": len(counts), "n": len(values)}
            )
        table = pd.DataFrame(per_image)
        rows.append(
            {
                "field": field,
                "n_images": len(table),
                "n_repeats": float(table["n"].median()) if len(table) else float("nan"),
                "mean_modal_share": float(table["modal_share"].mean()) if len(table) else float("nan"),
                "min_modal_share": float(table["modal_share"].min()) if len(table) else float("nan"),
                "share_fully_stable": float((table["modal_share"] == 1).mean()) if len(table) else float("nan"),
                "mean_n_distinct": float(table["n_distinct"].mean()) if len(table) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Daylight proxy and stratified reference sampling
# =============================================================================
def _utc_times(times) -> pd.DatetimeIndex:
    series = pd.Series(times)
    if pd.api.types.is_numeric_dtype(series):
        idx = pd.to_datetime(series, unit="ms", utc=True)  # Mapillary epoch ms
    else:
        idx = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx)


def solar_elevation(times, lat, lon) -> np.ndarray:
    """Approximate solar elevation angle in degrees.

    NOAA's general solar position equations (fractional-year series), good
    to roughly half a degree -- ample for telling day from twilight from
    night. Naive times are taken as UTC; Mapillary epoch milliseconds are
    understood.

    Args:
        times: Capture times (datetimes, ISO strings or epoch ms).
        lat, lon: Latitude and longitude in degrees (scalars or arrays).
    """
    idx = _utc_times(times)
    lat = np.broadcast_to(np.asarray(lat, dtype=float), (len(idx),))
    lon = np.broadcast_to(np.asarray(lon, dtype=float), (len(idx),))
    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    hours = (idx.hour + idx.minute / 60.0 + idx.second / 3600.0).to_numpy(dtype=float)
    days_in_year = np.where(idx.is_leap_year, 366.0, 365.0)
    gamma = 2 * np.pi / days_in_year * (day_of_year - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_minutes = hours * 60 + eqtime + 4 * lon
    hour_angle = np.radians(true_solar_minutes / 4 - 180)
    phi = np.radians(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    elevation = 90 - np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    elevation[np.asarray(idx.isna())] = np.nan
    return elevation


def light_condition(
    times,
    lat,
    lon,
    day_threshold: float = 6.0,
    night_threshold: float = -6.0,
) -> np.ndarray:
    """Daylight proxy from capture time and position: day / twilight / night.

    ``day`` when the sun is at least ``day_threshold`` degrees up,
    ``night`` below ``night_threshold`` (civil twilight ends at -6 degrees),
    ``twilight`` in between, ``unknown`` without a usable time. It is a proxy
    computed from metadata, not an observation of the image (weather,
    shade and timestamp errors are not seen).
    """
    elevation = solar_elevation(times, lat, lon)
    out = np.full(elevation.shape, "unknown", dtype=object)
    out[elevation >= day_threshold] = "day"
    out[(elevation < day_threshold) & (elevation >= night_threshold)] = "twilight"
    out[elevation < night_threshold] = "night"
    return out


def stratified_reference_sample(
    df: pd.DataFrame,
    strata_columns: Sequence[str],
    n_per_stratum: Optional[int] = None,
    total: Optional[int] = None,
    allocation: str = "equal",
    seed: int = 20261010,
    id_column: str = "image_id",
) -> pd.DataFrame:
    """Seeded, quota-based stratified sample for manual reference labelling.

    Strata are the combinations of ``strata_columns`` (e.g. area x road
    class x light condition); a missing value forms its own ``(missing)``
    stratum rather than disappearing.

    Quotas: either ``n_per_stratum`` for every stratum, or a ``total``
    allocated ``"equal"`` (remainder to strata in sorted order) or
    ``"proportional"`` (largest remainders). A stratum smaller than its quota
    is taken whole and its shortfall reported.

    Returns:
        The selected rows with ``stratum`` and ``sample_seed`` columns,
        ordered by stratum and id. ``attrs["sampling"]`` holds the seed, the
        allocation and a per-stratum table of available / quota / selected /
        shortfall.
    """
    if (n_per_stratum is None) == (total is None):
        raise ValueError("give exactly one of n_per_stratum or total")
    if allocation not in ("equal", "proportional"):
        raise ValueError("allocation must be 'equal' or 'proportional'")
    missing = [c for c in strata_columns if c not in df.columns]
    if missing:
        raise ValueError(f"strata columns not found: {missing}")

    labels = df[list(strata_columns)].astype(object).where(df[list(strata_columns)].notna(), "(missing)")
    stratum = labels.astype(str).agg(" | ".join, axis=1)
    available = stratum.value_counts().sort_index()
    names = list(available.index)

    if n_per_stratum is not None:
        quota = {s: int(n_per_stratum) for s in names}
    elif allocation == "equal":
        base, remainder = divmod(int(total), len(names))
        quota = {s: base + (1 if i < remainder else 0) for i, s in enumerate(names)}
    else:
        exact = {s: total * available[s] / available.sum() for s in names}
        quota = {s: int(math.floor(v)) for s, v in exact.items()}
        leftover = int(total) - sum(quota.values())
        for s in sorted(names, key=lambda s: (-(exact[s] - quota[s]), s))[:leftover]:
            quota[s] += 1

    rng = np.random.default_rng(seed)
    chosen = []
    report = []
    for s in names:
        rows = df.index[stratum == s]
        ordered = sorted(rows, key=lambda i: str(df.at[i, id_column]) if id_column in df.columns else str(i))
        take = min(quota[s], len(ordered))
        picked = [ordered[j] for j in sorted(rng.choice(len(ordered), size=take, replace=False))] if take else []
        chosen.extend(picked)
        report.append(
            {"stratum": s, "available": len(ordered), "quota": quota[s], "selected": take, "shortfall": quota[s] - take}
        )

    sample = df.loc[chosen].copy()
    sample["stratum"] = stratum.loc[chosen].values
    sample["sample_seed"] = seed
    sort_key = [c for c in ("stratum", id_column) if c in sample.columns]
    sample = sample.sort_values(sort_key, kind="mergesort")
    sample.attrs["sampling"] = {
        "seed": seed,
        "allocation": "fixed per stratum" if n_per_stratum is not None else allocation,
        "strata_columns": list(strata_columns),
        "per_stratum": report,
    }
    return sample
