"""Visualization theme presets for simulation renderers.

Themes are frozen dataclasses that group all styling constants together.
``visualize.py`` now accepts a ``Theme`` instance instead of referencing
hard-coded module-level constants, making it easy to swap palettes
via the ``--theme`` CLI argument or programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    """Complete collection of visualization style tokens."""

    # Per-metric display
    metric_labels: dict[str, str] = field(default_factory=dict)
    metric_colors: dict[str, str] = field(default_factory=dict)

    # Per-phase display
    phase_colors: dict[str, str] = field(default_factory=dict)
    phase_descriptions: dict[str, str] = field(default_factory=dict)

    # Agent-state grid
    state_colors: tuple[str, ...] = ("#2196F3", "#FF5722", "#4CAF50", "#FFC107")
    empty_cell_color: str = "#F0F0F0"
    empty_cell_color_dark: str = "#1A1A1A"
    grid_line_color: str = "#CCCCCC"
    grid_line_color_dark: str = "#333333"


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

_DEFAULT_METRIC_LABELS: dict[str, str] = {
    "state_entropy": "State Entropy",
    "neighbor_mutual_information": "Neighbor Mutual Information",
    "compression_ratio": "Compression Ratio",
    "morans_i": "Moran's I",
    "cluster_count": "Cluster Count",
    "predictability_hamming": "Hamming Distance",
    "quasi_periodicity_peaks": "Periodicity Peaks",
    "phase_transition_max_delta": "Phase Transition",
    "action_entropy_mean": "Action Entropy (mean)",
    "action_entropy_variance": "Action Entropy (var)",
    "block_ncd": "Block NCD",
}

_DEFAULT_METRIC_COLORS: dict[str, str] = {
    "state_entropy": "tab:blue",
    "neighbor_mutual_information": "tab:red",
    "compression_ratio": "tab:green",
    "morans_i": "tab:orange",
    "cluster_count": "tab:purple",
    "predictability_hamming": "tab:brown",
    "quasi_periodicity_peaks": "tab:pink",
    "phase_transition_max_delta": "tab:gray",
    "action_entropy_mean": "tab:olive",
    "action_entropy_variance": "tab:cyan",
    "block_ncd": "darkblue",
}

_DEFAULT_PHASE_COLORS: dict[str, str] = {
    "P1": "tab:blue",
    "P2": "tab:red",
    "Control": "tab:gray",
    "RW": "tab:olive",
}

_DEFAULT_PHASE_DESCRIPTIONS: dict[str, str] = {
    "P1": "Phase 1 (density)",
    "P2": "Phase 2 (state profile)",
    "Control": "Control (step-clock)",
    "RW": "Random Walk",
}

DEFAULT_THEME = Theme(
    metric_labels=_DEFAULT_METRIC_LABELS,
    metric_colors=_DEFAULT_METRIC_COLORS,
    phase_colors=_DEFAULT_PHASE_COLORS,
    phase_descriptions=_DEFAULT_PHASE_DESCRIPTIONS,
)

PAPER_THEME = Theme(
    metric_labels=_DEFAULT_METRIC_LABELS,
    metric_colors={
        "state_entropy": "#1f77b4",
        "neighbor_mutual_information": "#d62728",
        "compression_ratio": "#2ca02c",
        "morans_i": "#ff7f0e",
        "cluster_count": "#9467bd",
        "predictability_hamming": "#8c564b",
        "quasi_periodicity_peaks": "#e377c2",
        "phase_transition_max_delta": "#7f7f7f",
        "action_entropy_mean": "#bcbd22",
        "action_entropy_variance": "#17becf",
        "block_ncd": "#393b79",
    },
    phase_colors={
        "P1": "#1f77b4",
        "P2": "#d62728",
        "Control": "#7f7f7f",
        "RW": "#bcbd22",
    },
    phase_descriptions=_DEFAULT_PHASE_DESCRIPTIONS,
    state_colors=("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"),
    empty_cell_color="#FFFFFF",
    empty_cell_color_dark="#0D0D0D",
    grid_line_color="#E0E0E0",
    grid_line_color_dark="#2A2A2A",
)

REGISTERED_THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "paper": PAPER_THEME,
}


def get_theme(name: str) -> Theme:
    """Look up a theme by name (case-insensitive)."""
    key = name.lower()
    if key not in REGISTERED_THEMES:
        valid = ", ".join(sorted(REGISTERED_THEMES))
        raise ValueError(f"Unknown theme {name!r}; available: {valid}")
    return REGISTERED_THEMES[key]
