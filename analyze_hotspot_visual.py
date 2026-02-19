"""
analyze_hotspot_visual.py
=========================
Inter-Annotator Agreement (IAA) analysis tool for Sign Language Recognition (SLR) datasets.

PURPOSE
-------
This script is a Quality Assurance (QA) tool designed to:
  - Measure agreement/disagreement between N annotators (N >= 2) on the same video
  - Detect temporal "hotspots" (zones of persistent disagreement or low coverage)
  - Compute a reliability score (0–100) to objectively qualify dataset quality
  - Produce a professional timeline visualization exportable as PNG

DESIGN PHILOSOPHY
-----------------
The analysis is based on a discrete time grid approach:
  - The video timeline is discretized at a fine resolution (dt = 0.05s by default)
  - At each time step, we record which gloss is active for each annotator
  - Agreement = all active annotators label the same gloss at the same instant
  - Hotspot = contiguous zone where disagreement (or low coverage) persists

RELIABILITY SCORE (0–100)
--------------------------
  score = 0.80 * agreement_ratio * 100
        + 0.20 * stability_bonus * 100

  Where:
    - agreement_ratio  = agreement_time / multi_annotator_time
    - stability_bonus  = 1 - (number_of_hotspots / total_grid_steps)
      (penalizes datasets with many fragmented disagreement zones)

  Thresholds:
    - score >= 80 → RELIABLE
    - score >= 60 → MEDIUM
    - score  < 60 → UNRELIABLE

USAGE
-----
  python analyze_hotspot_visual.py \\
      --video_id A_000123 \\
      --csv ann_A.csv ann_B.csv ann_C.csv \\
      --annotators Alice Bob Carol \\
      [--dt 0.05] \\
      [--min_overlap 2] \\
      [--output_dir .]

REQUIRED CSV FORMAT
-------------------
  segment_id,start,end,gloss
  1,0.35,0.92,WHAT
  2,1.10,1.85,FLIGHT
  ...

Author: Florian Meloux — Clerc / SLR Data Infrastructure
"""

import argparse
import sys
import os
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import colormaps


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(csv_paths: list[str], annotator_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load annotation CSVs into a dictionary keyed by annotator name.

    Parameters
    ----------
    csv_paths : list of str
        File paths to the annotation CSV files (one per annotator).
    annotator_names : list of str
        Human-readable names for each annotator (same order as csv_paths).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping annotator_name → DataFrame with columns
        [segment_id, start, end, gloss].

    Raises
    ------
    ValueError
        If required columns are missing or if start >= end for any row.
    SystemExit
        If a file cannot be read.
    """
    if len(csv_paths) != len(annotator_names):
        raise ValueError(
            f"Mismatch: {len(csv_paths)} CSV files but {len(annotator_names)} annotator names."
        )

    required_columns = {"start", "end", "gloss"}
    annotations = {}

    for path, name in zip(csv_paths, annotator_names):
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)

        def _load_elan_wide_csv(file_path: str) -> Optional[pd.DataFrame]:
            """Parse ELAN-like wide CSV (rows: Category/start/End/Labels)."""
            try:
                wide = pd.read_csv(file_path, sep=';', header=None, dtype=str, keep_default_na=False)
            except Exception:
                return None

            if wide.shape[0] < 5 or wide.shape[1] < 2:
                return None

            first_col = wide.iloc[:, 0].astype(str).str.strip().str.lower()
            if "category" not in set(first_col):
                return None
            if "start" not in set(first_col):
                return None
            if "end" not in set(first_col):
                return None
            if "labels" not in set(first_col):
                return None

            idx_category = int(first_col[first_col == "category"].index[0])
            idx_start = int(first_col[first_col == "start"].index[0])
            idx_end = int(first_col[first_col == "end"].index[0])
            idx_labels = int(first_col[first_col == "labels"].index[0])

            categories = wide.iloc[idx_category, 1:]
            starts = wide.iloc[idx_start, 1:]
            ends = wide.iloc[idx_end, 1:]
            labels = wide.iloc[idx_labels, 1:]

            records = []
            for cat, st, en, lb in zip(categories, starts, ends, labels):
                cat_s = str(cat).strip().lower()
                if cat_s != "gloss":
                    continue
                lb_s = str(lb).strip()
                if not lb_s:
                    continue
                try:
                    st_f = float(str(st).replace(",", "."))
                    en_f = float(str(en).replace(",", "."))
                except ValueError:
                    continue
                if st_f >= en_f:
                    continue
                records.append({"start": st_f, "end": en_f, "gloss": lb_s})

            if not records:
                return None
            return pd.DataFrame.from_records(records)

        try:
            df = pd.read_csv(path)
        except Exception:
            df = None

        if df is None:
            df = _load_elan_wide_csv(path)
            if df is None:
                print(f"[ERROR] Cannot read {path}: unsupported CSV format", file=sys.stderr)
                sys.exit(1)

        # Normalize column names (strip whitespace, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        missing = required_columns - set(df.columns)
        if missing:
            # Fallback to ELAN-like wide format
            df_alt = _load_elan_wide_csv(path)
            if df_alt is None:
                raise ValueError(
                    f"CSV '{path}' is missing required columns: {missing}. "
                    f"Found: {list(df.columns)}"
                )
            df = df_alt

        # Validate temporal integrity
        invalid = df[df["start"] >= df["end"]]
        if not invalid.empty:
            raise ValueError(
                f"Annotator '{name}': {len(invalid)} row(s) with start >= end:\n{invalid}"
            )

        # Normalize gloss to uppercase stripped strings
        df["gloss"] = df["gloss"].astype(str).str.strip().str.upper()
        df = df.sort_values("start").reset_index(drop=True)

        annotations[name] = df
        print(f"  [LOADED] {name}: {len(df)} segments from '{path}'")

    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TIME GRID CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_time_grid(
    annotations: dict[str, pd.DataFrame],
    dt: float = 0.05,
    timeline_start: Optional[float] = None,
    timeline_end: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Discretize the full timeline and map each annotator's labels to it.

    For each time step t and each annotator, the label is:
      - The gloss of the active segment at time t (if any)
      - None if no segment covers t

    Parameters
    ----------
    annotations : dict[str, pd.DataFrame]
        Loaded annotations per annotator.
    dt : float
        Time resolution in seconds (default: 0.05s = 50ms).

    Returns
    -------
    time_grid : np.ndarray
        1D array of time steps [t_min, t_min+dt, ..., t_max].
    label_grid : dict[str, np.ndarray of object]
        Per-annotator array of gloss labels (or None) at each time step.
    """
    # Compute global time range across all annotators
    all_starts = [df["start"].min() for df in annotations.values()]
    all_ends   = [df["end"].max()   for df in annotations.values()]
    t_min = min(all_starts)
    t_max = max(all_ends)
    if timeline_start is not None:
        t_min = float(timeline_start)
    if timeline_end is not None:
        t_max = float(timeline_end)
    if t_max <= t_min:
        raise ValueError(f"Invalid timeline bounds: start={t_min}, end={t_max}")

    time_grid = np.arange(t_min, t_max + dt, dt)
    n_steps = len(time_grid)

    label_grid: dict[str, np.ndarray] = {}

    for name, df in annotations.items():
        labels = np.empty(n_steps, dtype=object)
        labels[:] = None  # Default: no annotation

        # Vectorized interval lookup using pandas IntervalIndex for speed
        for _, row in df.iterrows():
            # Find indices where start <= t < end
            mask = (time_grid >= row["start"]) & (time_grid < row["end"])
            labels[mask] = row["gloss"]

        label_grid[name] = labels

    return time_grid, label_grid


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — AGREEMENT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_agreement(
    time_grid: np.ndarray,
    label_grid: dict[str, np.ndarray],
    min_overlap: int = 2,
    dt: float = 0.05
) -> dict:
    """
    Compute agreement, disagreement and coverage statistics across the timeline.

    Definitions
    -----------
    - multi_annotator : at least `min_overlap` annotators are active (not None)
    - agreement       : all active annotators (>= min_overlap) label the same gloss
    - disagreement    : active annotators label different glosses

    Parameters
    ----------
    time_grid : np.ndarray
    label_grid : dict[str, np.ndarray]
    min_overlap : int
        Minimum number of simultaneous annotators to consider a step (default: 2).
    dt : float
        Time resolution used for duration computation.

    Returns
    -------
    dict with keys:
        - total_steps         : int
        - multi_steps         : int   — steps with >= min_overlap active annotators
        - agree_steps         : int   — steps in agreement
        - disagree_steps      : int   — steps in disagreement
        - solo_steps          : int   — steps with < min_overlap annotators
        - multi_time          : float — duration (s) of multi-annotator coverage
        - agree_time          : float
        - disagree_time       : float
        - agree_ratio         : float — agree_time / multi_time
        - disagree_ratio      : float
        - step_labels         : list[set] — active labels at each step
        - step_active_count   : np.ndarray — number of active annotators per step
        - agreement_mask      : np.ndarray[bool]
        - disagreement_mask   : np.ndarray[bool]
        - multi_mask          : np.ndarray[bool]
    """
    annotators = list(label_grid.keys())
    n_steps = len(time_grid)

    # Stack labels into 2D array: shape (n_annotators, n_steps)
    label_matrix = np.array([label_grid[a] for a in annotators], dtype=object)

    # Active mask: True where annotator has a label
    active_matrix = (label_matrix != None)  # noqa: E711

    # Count active annotators at each step
    step_active_count = active_matrix.sum(axis=0).astype(int)

    # Multi-annotator mask: >= min_overlap active
    multi_mask = step_active_count >= min_overlap

    # For each step, collect the set of distinct active glosses
    step_label_sets = []
    for i in range(n_steps):
        active_labels = set(label_matrix[active_matrix[:, i], i])
        step_label_sets.append(active_labels)

    # Agreement: exactly 1 distinct gloss across all active annotators
    agreement_mask = np.array([
        multi_mask[i] and len(step_label_sets[i]) == 1
        for i in range(n_steps)
    ], dtype=bool)

    # Disagreement: >= 2 distinct glosses (subset of multi)
    disagreement_mask = np.array([
        multi_mask[i] and len(step_label_sets[i]) > 1
        for i in range(n_steps)
    ], dtype=bool)

    multi_steps    = int(multi_mask.sum())
    agree_steps    = int(agreement_mask.sum())
    disagree_steps = int(disagreement_mask.sum())
    solo_steps     = int(n_steps - multi_steps)

    multi_time    = multi_steps    * dt
    agree_time    = agree_steps    * dt
    disagree_time = disagree_steps * dt

    agree_ratio    = agree_time    / multi_time if multi_time > 0 else 0.0
    disagree_ratio = disagree_time / multi_time if multi_time > 0 else 0.0

    return {
        "total_steps":       n_steps,
        "multi_steps":       multi_steps,
        "agree_steps":       agree_steps,
        "disagree_steps":    disagree_steps,
        "solo_steps":        solo_steps,
        "multi_time":        multi_time,
        "agree_time":        agree_time,
        "disagree_time":     disagree_time,
        "agree_ratio":       agree_ratio,
        "disagree_ratio":    disagree_ratio,
        "step_label_sets":   step_label_sets,
        "step_active_count": step_active_count,
        "agreement_mask":    agreement_mask,
        "disagreement_mask": disagreement_mask,
        "multi_mask":        multi_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HOTSPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_hotspots(
    time_grid: np.ndarray,
    agreement_stats: dict,
    label_grid: dict[str, np.ndarray],
    min_duration: float = 0.10
) -> list[dict]:
    """
    Detect temporal hotspots: contiguous zones of disagreement.

    A hotspot is a maximal contiguous range of time steps where:
      - at least 2 annotators are active AND
      - their labels differ (disagreement_mask is True)
    AND the zone duration >= min_duration seconds.

    Parameters
    ----------
    time_grid : np.ndarray
    agreement_stats : dict
        Output of compute_agreement().
    label_grid : dict[str, np.ndarray]
        Per-annotator label arrays.
    min_duration : float
        Minimum hotspot duration in seconds (default: 100ms).

    Returns
    -------
    list of dict, each with:
        - index       : int    — hotspot number (1-based)
        - t_start     : float
        - t_end       : float
        - duration    : float
        - labels_per_annotator : dict[str, set[str]] — glosses seen per annotator
        - conflicting_glosses  : set[str] — union of all conflicting glosses
    """
    disagreement_mask = agreement_stats["disagreement_mask"]
    dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.05
    annotators = list(label_grid.keys())

    hotspots = []
    in_hotspot = False
    h_start_idx = 0

    for i in range(len(time_grid)):
        if disagreement_mask[i] and not in_hotspot:
            in_hotspot = True
            h_start_idx = i
        elif (not disagreement_mask[i]) and in_hotspot:
            # End of hotspot at i-1
            _register_hotspot(
                hotspots, time_grid, label_grid, annotators,
                h_start_idx, i - 1, dt, min_duration
            )
            in_hotspot = False

    # Handle hotspot that extends to end of timeline
    if in_hotspot:
        _register_hotspot(
            hotspots, time_grid, label_grid, annotators,
            h_start_idx, len(time_grid) - 1, dt, min_duration
        )

    # Re-index cleanly
    for k, h in enumerate(hotspots):
        h["index"] = k + 1

    return hotspots


def _register_hotspot(
    hotspots: list,
    time_grid: np.ndarray,
    label_grid: dict[str, np.ndarray],
    annotators: list[str],
    start_idx: int,
    end_idx: int,
    dt: float,
    min_duration: float
) -> None:
    """Internal helper: validate and register a candidate hotspot."""
    duration = (end_idx - start_idx + 1) * dt
    if duration < min_duration:
        return

    t_start = float(time_grid[start_idx])
    t_end   = float(time_grid[end_idx]) + dt  # end is exclusive

    labels_per_annotator: dict[str, set] = {}
    for ann in annotators:
        glosses = set(
            g for g in label_grid[ann][start_idx:end_idx + 1]
            if g is not None
        )
        if glosses:
            labels_per_annotator[ann] = glosses

    conflicting_glosses: set = set()
    for gs in labels_per_annotator.values():
        conflicting_glosses |= gs

    hotspots.append({
        "index":                len(hotspots) + 1,
        "t_start":              t_start,
        "t_end":                t_end,
        "duration":             duration,
        "labels_per_annotator": labels_per_annotator,
        "conflicting_glosses":  conflicting_glosses,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RELIABILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_reliability_score(
    agreement_stats: dict,
    hotspots: list[dict],
    time_grid: np.ndarray
) -> dict:
    """
    Compute a global reliability score (0–100) for the annotation session.

    Formula
    -------
    score = 0.80 * agreement_component + 0.20 * stability_component

    Where:
      agreement_component = agree_ratio * 100
        → Rewards high temporal agreement (main driver, 80% weight)

      stability_component = (1 - hotspot_density) * 100
        hotspot_density = total_hotspot_steps / total_grid_steps
        → Penalizes fragmented, numerous hotspot zones (20% weight)

    The stability component ensures that even if agreement is moderate,
    datasets with many scattered hotspots are penalized more than those
    with a single large disagreement zone.

    Thresholds
    ----------
      score >= 80 → RELIABLE
      score >= 60 → MEDIUM
      score  < 60 → UNRELIABLE

    Parameters
    ----------
    agreement_stats : dict
    hotspots : list[dict]
    time_grid : np.ndarray

    Returns
    -------
    dict with keys: score, label, agreement_component, stability_component, hotspot_density
    """
    agree_ratio = agreement_stats["agree_ratio"]
    n_steps     = len(time_grid)
    dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.05

    # Total steps covered by hotspots
    total_hotspot_steps = sum(
        int(round(h["duration"] / dt)) for h in hotspots
    )
    hotspot_density = total_hotspot_steps / n_steps if n_steps > 0 else 0.0

    # Components (industrial profile: more severe than default QA)
    agreement_component = agree_ratio * 100.0
    stability_component = (1.0 - hotspot_density) * 100.0
    disagreement_component = (1.0 - agreement_stats.get("disagree_ratio", 0.0)) * 100.0

    # Weighted score: agreement remains dominant, but instability/disagreement
    # are penalized more strongly than the original 80/20 formula.
    score = (
        0.60 * agreement_component
        + 0.25 * stability_component
        + 0.15 * disagreement_component
    )
    score = float(np.clip(score, 0.0, 100.0))

    # Industrial gate: binary OK/KO with strict criteria.
    # A high score alone is not enough; all gates must pass.
    gate_score = score >= 92.0
    gate_agree = agree_ratio >= 0.90
    gate_hotspot_density = hotspot_density <= 0.08
    gate_hotspot_count = len(hotspots) <= 2
    industrial_ok = gate_score and gate_agree and gate_hotspot_density and gate_hotspot_count
    label = "OK" if industrial_ok else "KO"

    return {
        "score":                 round(score, 1),
        "label":                 label,
        "agreement_component":   round(agreement_component, 2),
        "stability_component":   round(stability_component, 2),
        "disagreement_component": round(disagreement_component, 2),
        "hotspot_density":       round(hotspot_density, 4),
        "industrial_ok":         industrial_ok,
        "gates": {
            "score_ge_92": gate_score,
            "agreement_ge_90pct": gate_agree,
            "hotspot_density_le_8pct": gate_hotspot_density,
            "hotspot_count_le_2": gate_hotspot_count,
        },
    }


def compute_annotator_gap_metrics(annotations: dict[str, pd.DataFrame]) -> dict:
    """Compute per-annotator timing deltas vs per-segment median consensus."""
    annotators = list(annotations.keys())
    counts = [len(df) for df in annotations.values()]
    common_n = min(counts) if counts else 0

    if common_n == 0:
        return {
            "per_annotator": {},
            "common_n": 0,
            "global_mean_start_gap": 0.0,
            "global_mean_end_gap": 0.0,
        }

    starts = np.array([annotations[a]["start"].to_numpy()[:common_n] for a in annotators], dtype=float)
    ends = np.array([annotations[a]["end"].to_numpy()[:common_n] for a in annotators], dtype=float)
    durs = ends - starts

    # Consensus pivot per comparable segment:
    # start/end/duration are medians across annotators for each segment index.
    med_starts = np.median(starts, axis=0)
    med_ends = np.median(ends, axis=0)
    med_durs = np.median(durs, axis=0)

    per_ann = {}
    all_start = []
    all_end = []

    for i, ann in enumerate(annotators):
        start_abs = np.abs(starts[i] - med_starts)
        end_abs = np.abs(ends[i] - med_ends)
        dur_abs = np.abs(durs[i] - med_durs)
        all_start.extend(start_abs.tolist())
        all_end.extend(end_abs.tolist())
        per_ann[ann] = {
            "segments": int(len(annotations[ann])),
            "total_duration_s": float(np.sum(durs[i])),
            "avg_segment_duration_s": float(np.mean(durs[i])) if len(durs[i]) else 0.0,
            "mean_abs_start_gap_s": float(np.mean(start_abs)),
            "mean_abs_end_gap_s": float(np.mean(end_abs)),
            "mean_abs_duration_gap_s": float(np.mean(dur_abs)),
            "avg_delta_start_s": float(np.mean(start_abs)),
            "min_delta_start_s": float(np.min(start_abs)) if len(start_abs) else 0.0,
            "max_delta_start_s": float(np.max(start_abs)) if len(start_abs) else 0.0,
            "avg_delta_end_s": float(np.mean(end_abs)),
            "min_delta_end_s": float(np.min(end_abs)) if len(end_abs) else 0.0,
            "max_delta_end_s": float(np.max(end_abs)) if len(end_abs) else 0.0,
            "avg_delta_dur_s": float(np.mean(dur_abs)),
            "min_delta_dur_s": float(np.min(dur_abs)) if len(dur_abs) else 0.0,
            "max_delta_dur_s": float(np.max(dur_abs)) if len(dur_abs) else 0.0,
        }

    return {
        "per_annotator": per_ann,
        "common_n": int(common_n),
        "global_mean_start_gap": float(np.mean(all_start)) if all_start else 0.0,
        "global_mean_end_gap": float(np.mean(all_end)) if all_end else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Curated palette for annotator lanes (colorblind-friendly base)
_ANNOTATOR_PALETTE = [
    "#4C72B0",  # muted blue
    "#55A868",  # muted green
    "#C44E52",  # muted red
    "#8172B2",  # purple
    "#CCB974",  # yellow-brown
    "#64B5CD",  # light blue
]

# Color map for gloss labels (auto-generated)
def _build_gloss_colormap(all_glosses: list[str]) -> dict[str, tuple]:
    """Assign a unique pastel color to each unique gloss."""
    cmap = colormaps.get_cmap("tab20b")
    return {
        gloss: cmap(i / max(len(all_glosses), 1))
        for i, gloss in enumerate(sorted(all_glosses))
    }


def plot_timeline_with_hotspots(
    df_segments: pd.DataFrame,
    video_id: str,
    annotators: list[str],
    metrics: dict,
    hotspots: list[dict],
    output_path: str,
) -> None:
    """Industrial QA timeline with clear per-annotator rows and hotspot bands."""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#F8FAFC",
    })

    df = df_segments.copy()
    for col in ["annotator", "start", "end", "gloss"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    n_ann = max(1, len(annotators))
    fig_h = max(6.2, 5.0 + 0.85 * n_ann)
    fig, (ax, ax_footer) = plt.subplots(
        nrows=2,
        figsize=(16, fig_h),
        gridspec_kw={"height_ratios": [7.2, 2.8], "hspace": 0.14},
    )

    t_min = float(metrics.get("timeline_start_s", df["start"].min() if not df.empty else 0.0))
    t_max = float(metrics.get("timeline_end_s", df["end"].max() if not df.empty else 1.0))
    if t_max <= t_min:
        t_max = t_min + 1.0

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.5, n_ann - 0.5)
    ax.set_xlabel("Time (seconds)", fontsize=12, labelpad=8)
    # Force industrial, regular time ticks every 0.2s over the full timeline.
    xticks = np.arange(t_min, t_max + 1e-9, 0.2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=10)
    ax.set_yticks(range(n_ann))
    ax.set_yticklabels(annotators, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#64748B")
    ax_footer.axis("off")

    # Draw agreement bands first (background), then hotspot bands on top.
    # Both span full chart height to make time zones immediately visible.
    for z in metrics.get("agreement_zones", []):
        ax.axvspan(
            float(z["start"]), float(z["end"]),
            ymin=0, ymax=1, color="#2EAF6D", alpha=0.15, zorder=0, linewidth=0
        )
    for h in hotspots:
        hs = float(h.get("start", h.get("t_start", 0.0)))
        he = float(h.get("end", h.get("t_end", hs)))
        ax.axvspan(hs, he, ymin=0, ymax=1, color="#E1533D", alpha=0.15, zorder=1, linewidth=0)

    # Row geometry: one clear lane per annotator with vertical padding.
    lane_h = 0.72
    min_label_w = 0.12
    row_colors = ["#4F86C6", "#5B7CB7", "#6A9A8B", "#7A8FB3", "#5F95A8", "#6A8FA3"]

    for row_idx, ann in enumerate(annotators):
        ann_df = df[df["annotator"] == ann].sort_values("start")
        base_color = row_colors[row_idx % len(row_colors)]
        for _, r in ann_df.iterrows():
            s = float(r["start"])
            e = float(r["end"])
            w = max(0.0, e - s)
            gloss = str(r["gloss"]).strip().upper()
            rect = FancyBboxPatch(
                (s, row_idx - lane_h / 2),
                w, lane_h,
                boxstyle="round,pad=0.006,rounding_size=0.004",
                facecolor=base_color,
                edgecolor=_darken(base_color, 0.75),
                linewidth=1.0,
                alpha=0.88,
                zorder=3
            )
            ax.add_patch(rect)
            if w >= min_label_w:
                ax.text(
                    s + w / 2, row_idx, gloss,
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=4
                )

    score = float(metrics.get("score", 0.0))
    label = str(metrics.get("label", "UNKNOWN"))
    badge_color = "#1FA15A" if score >= 90 else ("#F39C12" if score >= 75 else "#D64545")

    title = (
        f"VIDEO: {video_id}  |  "
        f"Annotators: {', '.join(annotators)}  |  "
        f"Score: {score:.1f}/100"
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.text(
        0.99, 1.04, f"● {label}",
        transform=ax.transAxes, ha="right", va="center",
        fontsize=16, fontweight="bold", color=badge_color,
        bbox=dict(boxstyle="round,pad=0.35", fc="#EEF2F7", ec=badge_color, lw=1.8)
    )

    legend_items = [
        mpatches.Patch(facecolor="#2EAF6D", alpha=0.25, label="Agreement zone"),
        mpatches.Patch(facecolor="#E1533D", alpha=0.25, label="Hotspot (disagreement)"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper left", fontsize=11, framealpha=0.95, ncol=2)
    leg.get_frame().set_facecolor("#F8FAFC")

    if not hotspots:
        ax.text(
            0.01, 1.02, "No hotspots (perfect agreement over comparable time).",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color="#2E7D32", fontweight="bold"
        )

    summary = (
        f"Common comparable segments = {metrics.get('common_segments', 0)} | "
        f"Coverage = {metrics.get('coverage', 0.0):.2f} s | "
        f"Agreement = {metrics.get('agreement_ratio', 0.0)*100:.1f} % | "
        f"Disagreement = {metrics.get('disagreement_ratio', 0.0)*100:.1f} % | "
        f"Hotspots = {len(hotspots)} | Score = {score:.1f}/100 ({label})"
    )
    ax_footer.text(
        0.5, 0.70, summary,
        ha="center", va="center",
        fontsize=11, color="#334155", style="italic",
        transform=ax_footer.transAxes
    )
    ax_footer.text(
        0.5, 0.50,
        "Consensus boundaries = per-segment median across annotators. "
        "All Δ values below are mean absolute deviations vs this consensus.",
        ha="center", va="center",
        fontsize=10, color="#475569",
        transform=ax_footer.transAxes
    )

    per_ann = metrics.get("per_annotator", {})
    ann_txt = []
    for ann in annotators:
        v = per_ann.get(ann, {})
        ann_txt.append(
            f"{ann}: n={v.get('n', 0)}, dur={v.get('dur', 0.0):.2f} s | "
            f"avg Δstart={_ms(v.get('avg_delta_start_s', 0.0))} ms "
            f"({ _ms(v.get('min_delta_start_s', 0.0)) }–{ _ms(v.get('max_delta_start_s', 0.0)) }) | "
            f"avg Δend={_ms(v.get('avg_delta_end_s', 0.0))} ms "
            f"({ _ms(v.get('min_delta_end_s', 0.0)) }–{ _ms(v.get('max_delta_end_s', 0.0)) }) | "
            f"avg Δdur={_ms(v.get('avg_delta_dur_s', 0.0))} ms "
            f"({ _ms(v.get('min_delta_dur_s', 0.0)) }–{ _ms(v.get('max_delta_dur_s', 0.0)) })"
        )
    if ann_txt:
        # One annotator per line for strict readability.
        y = 0.28
        for line in ann_txt:
            ax_footer.text(
                0.01, y, line,
                ha="left", va="top",
                fontsize=10.2, color="#475569",
                transform=ax_footer.transAxes
            )
            y -= 0.22

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)


def plot_timeline(
    video_id: str,
    annotations: dict[str, pd.DataFrame],
    time_grid: np.ndarray,
    label_grid: dict[str, np.ndarray],
    agreement_stats: dict,
    hotspots: list[dict],
    reliability: dict,
    annotator_gaps: dict,
    output_dir: str = ".",
    dt: float = 0.05
) -> str:
    """Backward-compatible wrapper that feeds industrial plotting API."""
    annotators = list(annotations.keys())
    frames = []
    for ann in annotators:
        df = annotations[ann][["start", "end", "gloss"]].copy()
        df["annotator"] = ann
        frames.append(df)
    df_segments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["annotator", "start", "end", "gloss"]
    )

    # Convert agreement mask to contiguous zones for the new plotting API.
    agreement_zones = []
    mask = agreement_stats.get("agreement_mask", np.array([], dtype=bool))
    in_span = False
    span_start = 0.0
    for i, val in enumerate(mask):
        t = float(time_grid[i])
        if val and not in_span:
            in_span = True
            span_start = t
        elif (not val) and in_span:
            agreement_zones.append({"start": span_start, "end": t})
            in_span = False
    if in_span:
        agreement_zones.append({"start": span_start, "end": float(time_grid[-1]) + dt})

    hotspot_payload = [
        {
            "start": float(h["t_start"]),
            "end": float(h["t_end"]),
            "type": "disagreement",
            "details": {k: " | ".join(sorted(v)) for k, v in h.get("labels_per_annotator", {}).items()},
        }
        for h in hotspots
    ]

    metrics = {
        "score": reliability.get("score", 0.0),
        "label": f"INDUSTRIAL {reliability.get('label', 'KO')}",
        "agreement_ratio": agreement_stats.get("agree_ratio", 0.0),
        "disagreement_ratio": agreement_stats.get("disagree_ratio", 0.0),
        "coverage": agreement_stats.get("multi_time", 0.0),
        "common_segments": annotator_gaps.get("common_n", 0),
        "timeline_start_s": float(time_grid[0]) if len(time_grid) else 0.0,
        "timeline_end_s": (float(time_grid[-1]) + dt) if len(time_grid) else 1.0,
        "agreement_zones": agreement_zones,
        "per_annotator": {},
    }
    for ann, vals in annotator_gaps.get("per_annotator", {}).items():
        metrics["per_annotator"][ann] = {
            "n": vals.get("segments", 0),
            "dur": vals.get("total_duration_s", 0.0),
            "avg_delta_start_s": vals.get("avg_delta_start_s", vals.get("mean_abs_start_gap_s", 0.0)),
            "min_delta_start_s": vals.get("min_delta_start_s", 0.0),
            "max_delta_start_s": vals.get("max_delta_start_s", 0.0),
            "avg_delta_end_s": vals.get("avg_delta_end_s", vals.get("mean_abs_end_gap_s", 0.0)),
            "min_delta_end_s": vals.get("min_delta_end_s", 0.0),
            "max_delta_end_s": vals.get("max_delta_end_s", 0.0),
            "avg_delta_dur_s": vals.get("avg_delta_dur_s", vals.get("mean_abs_duration_gap_s", 0.0)),
            "min_delta_dur_s": vals.get("min_delta_dur_s", 0.0),
            "max_delta_dur_s": vals.get("max_delta_dur_s", 0.0),
        }

    output_path = os.path.join(output_dir, f"timeline_{video_id}.png")
    plot_timeline_with_hotspots(
        df_segments=df_segments,
        video_id=video_id,
        annotators=annotators,
        metrics=metrics,
        hotspots=hotspot_payload,
        output_path=output_path,
    )
    return output_path


def _draw_mask_overlay(
    ax, time_grid: np.ndarray, mask: np.ndarray,
    dt: float, color: str, alpha: float, n_ann: int, zorder: int
) -> None:
    """Draw contiguous spans where mask is True as colored vertical bands."""
    in_span = False
    span_start = 0.0

    for i, val in enumerate(mask):
        t = float(time_grid[i])
        if val and not in_span:
            in_span = True
            span_start = t
        elif (not val) and in_span:
            ax.axvspan(span_start, t, color=color, alpha=alpha,
                       zorder=zorder, linewidth=0)
            in_span = False

    if in_span:
        ax.axvspan(span_start, float(time_grid[-1]) + dt,
                   color=color, alpha=alpha, zorder=zorder, linewidth=0)


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Return a darkened version of a hex color."""
    try:
        rgba = matplotlib.colors.to_rgba(hex_color)
        darkened = tuple(min(1.0, c * factor) for c in rgba[:3]) + (rgba[3],)
        return matplotlib.colors.to_hex(darkened)
    except Exception:
        return hex_color


def _ms(sec: float) -> int:
    """Convert seconds to rounded milliseconds."""
    return int(round(float(sec) * 1000.0))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    video_id: str,
    annotators: list[str],
    agreement_stats: dict,
    hotspots: list[dict],
    reliability: dict,
    annotator_gaps: dict,
    output_png: Optional[str] = None
) -> None:
    """
    Print a structured, human-readable QA report to stdout.

    Sections
    --------
    1. Header (video + annotators)
    2. Temporal statistics
    3. Reliability score + label
    4. Hotspot report
    5. Visual output path
    """
    W = 64  # report width

    # ── Helpers ──────────────────────────────────────────────────────────────
    sep    = "─" * W
    thick  = "═" * W

    def _pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(thick)
    print(f"  IAA QUALITY REPORT — VIDEO: {video_id}")
    print(thick)
    print(f"  Annotators : {', '.join(annotators)}")
    print(f"  Count      : {len(annotators)}")
    print(sep)

    # ── Temporal stats ────────────────────────────────────────────────────────
    print()
    print("  TEMPORAL STATISTICS")
    print(f"  {'Multi-annotator coverage':<35} {agreement_stats['multi_time']:.3f} s")
    print(f"  {'Agreement time':<35} {agreement_stats['agree_time']:.3f} s  ({_pct(agreement_stats['agree_ratio'])})")
    print(f"  {'Disagreement time':<35} {agreement_stats['disagree_time']:.3f} s  ({_pct(agreement_stats['disagree_ratio'])})")
    print()

    # ── Score ─────────────────────────────────────────────────────────────────
    print(sep)
    print()
    score_bar = _ascii_bar(reliability["score"], 100, width=30)
    label_icon = {"OK": "✓", "KO": "✗"}.get(reliability["label"], "")
    print(f"  RELIABILITY SCORE : {reliability['score']:>6.1f} / 100   {score_bar}")
    print(f"  INDUSTRIAL STATUS : {label_icon} {reliability['label']}")
    print()
    print(f"    Component  Agreement    : {reliability['agreement_component']:>6.2f} / 100  (weight: 60%)")
    print(f"    Component  Stability    : {reliability['stability_component']:>6.2f} / 100  (weight: 25%)")
    print(f"    Component  Anti-disag.  : {reliability['disagreement_component']:>6.2f} / 100  (weight: 15%)")
    print(f"    Hotspot density       : {reliability['hotspot_density'] * 100:.2f}%")
    gates = reliability.get("gates", {})
    print(f"    Gate score ≥ 92       : {'OK' if gates.get('score_ge_92') else 'KO'}")
    print(f"    Gate agreement ≥ 90%  : {'OK' if gates.get('agreement_ge_90pct') else 'KO'}")
    print(f"    Gate hotspot ≤ 8%     : {'OK' if gates.get('hotspot_density_le_8pct') else 'KO'}")
    print(f"    Gate hotspots ≤ 2     : {'OK' if gates.get('hotspot_count_le_2') else 'KO'}")
    print()
    print("  ANNOTATOR GAP METRICS (vs median timeline)")
    print(f"    Comparable segments (all annotators): {annotator_gaps.get('common_n', 0)}")
    print(f"    Global mean |Δstart|: {annotator_gaps.get('global_mean_start_gap', 0.0):.3f}s")
    print(f"    Global mean |Δend|  : {annotator_gaps.get('global_mean_end_gap', 0.0):.3f}s")
    for ann, vals in annotator_gaps.get("per_annotator", {}).items():
        print(
            f"    - {ann}: n={vals['segments']}, dur={vals['total_duration_s']:.2f}s, "
            f"|Δstart|={vals['mean_abs_start_gap_s']:.3f}s, "
            f"|Δend|={vals['mean_abs_end_gap_s']:.3f}s, "
            f"|Δdur|={vals['mean_abs_duration_gap_s']:.3f}s"
        )
    print()

    # ── Hotspots ──────────────────────────────────────────────────────────────
    print(sep)
    print()
    if not hotspots:
        print("  HOTSPOTS : None detected  ✓")
    else:
        print(f"  HOTSPOTS DETECTED : {len(hotspots)}")
        print()
        for h in hotspots:
            print(f"  HOTSPOT {h['index']}")
            print(f"    Time     : {h['t_start']:.3f}s  →  {h['t_end']:.3f}s  "
                  f"(duration: {h['duration']:.3f}s)")
            for ann, glosses in h["labels_per_annotator"].items():
                gloss_str = " | ".join(sorted(glosses))
                print(f"    {ann:<16}: {gloss_str}")
            conflict_str = " vs ".join(sorted(h["conflicting_glosses"]))
            print(f"    Conflict : {conflict_str}")
            print()

    # ── Output ────────────────────────────────────────────────────────────────
    if output_png:
        print(sep)
        print(f"  Visual saved → {output_png}")

    print(thick)
    print()


def _ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    """Generate a compact ASCII progress bar."""
    filled = int(round(value / max_value * width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="analyze_hotspot_visual.py",
        description="Inter-Annotator Agreement (IAA) QA tool for SLR datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_hotspot_visual.py \\
      --video_id A_000123 \\
      --csv ann_Alice.csv ann_Bob.csv ann_Carol.csv \\
      --annotators Alice Bob Carol

  python analyze_hotspot_visual.py \\
      --video_id B_005 \\
      --csv a.csv b.csv \\
      --annotators Ann Ben \\
      --dt 0.02 \\
      --output_dir ./reports
        """
    )
    parser.add_argument("--video_id",   required=True,        help="Video identifier (e.g. A_000123)")
    parser.add_argument("--csv",        nargs="+", required=True, help="One CSV file per annotator")
    parser.add_argument("--annotators", nargs="+", required=True, help="Annotator names (same order as --csv)")
    parser.add_argument("--dt",         type=float, default=0.05, help="Time grid resolution in seconds (default: 0.05)")
    parser.add_argument("--min_overlap",type=int,   default=2,    help="Min simultaneous annotators to count as multi-annotated (default: 2)")
    parser.add_argument("--min_hotspot_duration", type=float, default=0.10,
                        help="Minimum hotspot duration in seconds (default: 0.10)")
    parser.add_argument("--video_duration", type=float, default=None,
                        help="Full video duration in seconds. If set, timeline spans 0.0 to this value.")
    parser.add_argument("--timeline_start", type=float, default=0.0,
                        help="Timeline start in seconds when --video_duration is set (default: 0.0).")
    parser.add_argument("--output_dir", default=".", help="Directory to save the PNG (default: current dir)")
    return parser.parse_args()


def main() -> None:
    """Main orchestration function."""
    args = parse_args()

    # ── Validate argument counts ──────────────────────────────────────────────
    if len(args.csv) != len(args.annotators):
        print(
            f"[ERROR] {len(args.csv)} CSV files but {len(args.annotators)} annotator names. "
            "They must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(args.csv) < 1:
        print("[ERROR] At least 1 annotator file is required.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[1/6] Loading {len(args.csv)} annotation files...")
    annotations = load_annotations(args.csv, args.annotators)

    timeline_start = None
    timeline_end = None
    if args.video_duration is not None:
        timeline_start = float(args.timeline_start)
        timeline_end = timeline_start + float(args.video_duration)

    print(f"[2/6] Building time grid (dt={args.dt}s)...")
    time_grid, label_grid = build_time_grid(
        annotations,
        dt=args.dt,
        timeline_start=timeline_start,
        timeline_end=timeline_end,
    )

    effective_min_overlap = 1 if len(args.csv) == 1 else args.min_overlap
    if len(args.csv) == 1:
        print("[INFO] Single annotator mode: min_overlap forced to 1.")
    print(f"[3/6] Computing agreement statistics (min_overlap={effective_min_overlap})...")
    stats = compute_agreement(time_grid, label_grid, min_overlap=effective_min_overlap, dt=args.dt)

    print("[4/6] Detecting hotspots...")
    hotspots = detect_hotspots(
        time_grid, stats, label_grid,
        min_duration=args.min_hotspot_duration
    )

    print("[5/6] Computing reliability score...")
    reliability = compute_reliability_score(stats, hotspots, time_grid)
    annotator_gaps = compute_annotator_gap_metrics(annotations)

    print("[6/6] Generating timeline visualization...")
    output_png = plot_timeline(
        video_id=args.video_id,
        annotations=annotations,
        time_grid=time_grid,
        label_grid=label_grid,
        agreement_stats=stats,
        hotspots=hotspots,
        reliability=reliability,
        annotator_gaps=annotator_gaps,
        output_dir=args.output_dir,
        dt=args.dt
    )

    print_summary(
        video_id=args.video_id,
        annotators=args.annotators,
        agreement_stats=stats,
        hotspots=hotspots,
        reliability=reliability,
        annotator_gaps=annotator_gaps,
        output_png=output_png
    )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
