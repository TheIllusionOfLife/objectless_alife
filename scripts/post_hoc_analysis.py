"""Post-hoc analyses on stage_d data for supplementary tables.

Computes:
  1. Halt-window sweep (Table D) — survival + median delta_mi for {5,10,20}
  2. Alt null models (Table G) — mean null MI: shuffle, block, fixed-marginal
  3. Spatial scrambling (Section H) — mean scrambled MI for top-50 P2
  4. Transfer entropy (Section I) — per-condition median TE
  5. Capacity-matched controls (Section J) — Phase 5 & 6 delta_mi

Usage:
    uv run python scripts/post_hoc_analysis.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from objectless_alife.aggregation import (
    run_halt_window_sweep,
    select_top_rules_by_delta_mi,
)
from objectless_alife.config import HaltWindowSweepConfig
from objectless_alife.metrics import (
    block_shuffle_null_mi,
    fixed_marginal_null_mi,
    neighbor_mutual_information,
    neighbor_transfer_entropy,
    shuffle_null_mi,
    spatial_scramble_mi,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage_d"
P2_METRICS = DATA_DIR / "phase_2" / "logs" / "metrics_summary.parquet"
P2_RULES = DATA_DIR / "phase_2" / "rules"
P2_SIM_LOG = DATA_DIR / "phase_2" / "logs" / "simulation_log.parquet"
P1_SIM_LOG = DATA_DIR / "phase_1" / "logs" / "simulation_log.parquet"
CTRL_SIM_LOG = DATA_DIR / "control" / "logs" / "simulation_log.parquet"


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


GRID_W, GRID_H = 20, 20
HALT_WINDOWS = (5, 10, 20)
SAMPLE_SIZE = 5000
MAX_LABELS = 199
MI_COLUMNS = [
    "rule_id",
    "step",
    "neighbor_mutual_information",
    "mi_shuffle_null",
]


def _load_final_snapshots(
    sim_log_path: Path, rule_ids: set[str]
) -> dict[str, tuple[tuple[int, int, int, int], ...]]:
    """Load final-step snapshots for given rule_ids from a simulation log."""
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
    )
    rows = table.to_pylist()

    # Find max step per rule_id
    max_steps: dict[str, int] = {}
    for row in rows:
        rid = row["rule_id"]
        if rid not in rule_ids:
            continue
        step = int(row["step"])
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step

    # Collect final-step agents
    snapshots: dict[str, list[tuple[int, int, int, int]]] = {rid: [] for rid in rule_ids}
    for row in rows:
        rid = row["rule_id"]
        if rid not in rule_ids:
            continue
        if int(row["step"]) != max_steps[rid]:
            continue
        snapshots[rid].append(
            (int(row["agent_id"]), int(row["x"]), int(row["y"]), int(row["state"]))
        )

    return {rid: tuple(agents) for rid, agents in snapshots.items() if agents}


def _load_sim_log_tuples(
    sim_log_path: Path, rule_ids: set[str]
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """Load full sim log as (step, agent_id, x, y, state) tuples."""
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
    )
    rows = table.to_pylist()
    result: dict[str, list[tuple[int, int, int, int, int]]] = {rid: [] for rid in rule_ids}
    for row in rows:
        rid = row["rule_id"]
        if rid not in rule_ids:
            continue
        result[rid].append(
            (
                int(row["step"]),
                int(row["agent_id"]),
                int(row["x"]),
                int(row["y"]),
                int(row["state"]),
            )
        )
    return result


def _rule_ids_for_seeds(phase_value: int, seeds: list[int]) -> set[str]:
    """Build rule_id set from phase and seeds."""
    return {f"phase{phase_value}_rs{s}_ss{s}" for s in seeds}


def _median_delta_mi_from_metrics(metrics_path: Path) -> float:
    """Compute median delta_mi from a metrics summary parquet."""
    table = pq.read_table(metrics_path, columns=MI_COLUMNS)
    rows = table.to_pylist()
    max_steps: dict[str, int] = {}
    for row in rows:
        rid = row["rule_id"]
        step = int(row["step"])
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step
    delta_mi_vals = []
    for row in rows:
        rid = row["rule_id"]
        if int(row["step"]) != max_steps[rid]:
            continue
        mi = row["neighbor_mutual_information"]
        null = row["mi_shuffle_null"]
        if mi is not None and null is not None:
            delta_mi_vals.append(float(mi) - float(null))
    return statistics.median(delta_mi_vals) if delta_mi_vals else 0.0


# ---------------------------------------------------------------------------
# Analysis 1: Halt-window sweep
# ---------------------------------------------------------------------------
def run_halt_window_analysis(
    top_seeds: list[int],
) -> dict[int, dict[str, float]]:
    """Run halt-window sweep: {window: {survival_rate, median_delta_mi}}."""
    print("=== Halt-Window Sweep ===")
    config = HaltWindowSweepConfig(
        rule_seeds=tuple(top_seeds),
        halt_windows=HALT_WINDOWS,
        out_dir=PROJECT_ROOT / "data" / "post_hoc" / "halt_window",
    )
    output_path = run_halt_window_sweep(config)
    table = pq.read_table(output_path)
    rows = table.to_pylist()

    results: dict[int, dict[str, float]] = {}
    for window in HALT_WINDOWS:
        window_rows = [r for r in rows if r["halt_window"] == window]
        if not window_rows:
            results[window] = {"survival_rate": 0.0, "median_delta_mi": 0.0}
            print(f"  Window {window:2d}: no data")
            continue
        survived = sum(1 for r in window_rows if r["survived"])
        survival_rate = survived / len(window_rows) * 100
        mi_vals = [float(r["delta_mi"]) for r in window_rows if r["survived"]]
        median_mi = statistics.median(mi_vals) if mi_vals else 0.0
        results[window] = {
            "survival_rate": survival_rate,
            "median_delta_mi": median_mi,
        }
        print(
            f"  Window {window:2d}: survival={survival_rate:.1f}%, median delta_mi={median_mi:.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Alternative null models
# ---------------------------------------------------------------------------
def run_alt_null_analysis(
    snapshots: dict[str, tuple[tuple[int, int, int, int], ...]],
) -> dict[str, float]:
    """Mean null MI for three null models across top-50 P2 snapshots."""
    print("\n=== Alternative Null Models ===")
    state_shuffle_vals: list[float] = []
    block_shuffle_vals: list[float] = []
    fixed_marginal_vals: list[float] = []

    for i, (_rid, snap) in enumerate(snapshots.items()):
        rng = random.Random(i)
        state_shuffle_vals.append(shuffle_null_mi(snap, GRID_W, GRID_H, rng=rng))
        block_shuffle_vals.append(
            block_shuffle_null_mi(snap, GRID_W, GRID_H, block_size=4, rng=rng)
        )
        fixed_marginal_vals.append(fixed_marginal_null_mi(snap, GRID_W, GRID_H, rng=rng))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(snapshots)} rules")

    results = {
        "state_shuffle": _safe_mean(state_shuffle_vals),
        "block_shuffle": _safe_mean(block_shuffle_vals),
        "fixed_marginal": _safe_mean(fixed_marginal_vals),
    }
    for name, val in results.items():
        print(f"  {name}: {val:.4f} bits")
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Spatial scrambling
# ---------------------------------------------------------------------------
def run_spatial_scramble_analysis(
    snapshots: dict[str, tuple[tuple[int, int, int, int], ...]],
) -> dict[str, float]:
    """Mean observed MI and mean scrambled MI for top-50 P2."""
    print("\n=== Spatial Scrambling ===")
    observed_vals: list[float] = []
    scrambled_vals: list[float] = []

    for i, (_rid, snap) in enumerate(snapshots.items()):
        rng = random.Random(i + 1000)
        observed_vals.append(neighbor_mutual_information(snap, GRID_W, GRID_H))
        scrambled_vals.append(spatial_scramble_mi(snap, GRID_W, GRID_H, rng=rng))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(snapshots)} rules")

    results = {
        "observed_mi": _safe_mean(observed_vals),
        "scrambled_mi": _safe_mean(scrambled_vals),
    }
    print(f"  Mean observed MI:  {results['observed_mi']:.4f} bits")
    print(f"  Mean scrambled MI: {results['scrambled_mi']:.4f} bits")
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Transfer entropy
# ---------------------------------------------------------------------------
def run_transfer_entropy_analysis(
    top_seeds: list[int],
) -> dict[str, float]:
    """Per-condition median TE for top-50 P2, P1, and Control."""
    print("\n=== Transfer Entropy ===")
    conditions: dict[str, tuple[Path, int]] = {
        "Phase 2": (P2_SIM_LOG, 2),
        "Phase 1": (P1_SIM_LOG, 1),
        "Control": (CTRL_SIM_LOG, 3),
    }

    results: dict[str, float] = {}
    for cond_name, (sim_log_path, phase_val) in conditions.items():
        rule_ids = _rule_ids_for_seeds(phase_val, top_seeds)
        sim_data = _load_sim_log_tuples(sim_log_path, rule_ids)
        te_values: list[float] = []
        for _rid, tuples in sim_data.items():
            if tuples:
                te = neighbor_transfer_entropy(tuples, GRID_W, GRID_H)
                te_values.append(te)
        median_te = statistics.median(te_values) if te_values else 0.0
        results[cond_name] = median_te
        print(f"  {cond_name}: median TE = {median_te:.4f} bits (n={len(te_values)})")

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Capacity-matched controls (Phase 5 & 6)
# ---------------------------------------------------------------------------
def run_capacity_matched_analysis() -> dict[str, dict[str, float]]:
    """Run Phase 5 and 6 experiments; compare to P1/P2 from stage_d."""
    print("\n=== Capacity-Matched Controls ===")

    from objectless_alife.experiments.summaries import collect_final_metric_rows
    from objectless_alife.rules import ObservationPhase
    from objectless_alife.simulation import run_batch_search

    # Run Phase 5 (Capacity-matched Phase 1)
    print("  Running Phase 5 (capacity-matched P1) — 5000 rules...")
    phase5_out = PROJECT_ROOT / "data" / "post_hoc" / "phase_5"
    phase5_results = run_batch_search(
        n_rules=SAMPLE_SIZE,
        phase=ObservationPhase.PHASE1_CAPACITY_MATCHED,
        out_dir=phase5_out,
    )
    phase5_survived = sum(1 for r in phase5_results if r.survived)
    phase5_survival = phase5_survived / len(phase5_results) * 100 if phase5_results else 0.0

    phase5_metrics = collect_final_metric_rows(
        phase5_out / "logs" / "metrics_summary.parquet",
        MI_COLUMNS,
        phase5_results,
        MAX_LABELS,
    )
    phase5_delta_mi = []
    for row in phase5_metrics:
        mi = row.get("neighbor_mutual_information")
        null = row.get("mi_shuffle_null")
        if mi is not None and null is not None:
            phase5_delta_mi.append(float(mi) - float(null))
    phase5_med = statistics.median(phase5_delta_mi) if phase5_delta_mi else 0.0
    print(f"    Phase 5: survival={phase5_survival:.1f}%, median delta_mi={phase5_med:.3f}")

    # Run Phase 6 (Random-encoding Phase 2)
    print("  Running Phase 6 (random-encoding P2) — 5000 rules...")
    phase6_out = PROJECT_ROOT / "data" / "post_hoc" / "phase_6"
    phase6_results = run_batch_search(
        n_rules=SAMPLE_SIZE,
        phase=ObservationPhase.PHASE2_RANDOM_ENCODING,
        out_dir=phase6_out,
    )
    phase6_survived = sum(1 for r in phase6_results if r.survived)
    phase6_survival = phase6_survived / len(phase6_results) * 100 if phase6_results else 0.0

    phase6_metrics = collect_final_metric_rows(
        phase6_out / "logs" / "metrics_summary.parquet",
        MI_COLUMNS,
        phase6_results,
        MAX_LABELS,
    )
    phase6_delta_mi = []
    for row in phase6_metrics:
        mi = row.get("neighbor_mutual_information")
        null = row.get("mi_shuffle_null")
        if mi is not None and null is not None:
            phase6_delta_mi.append(float(mi) - float(null))
    phase6_med = statistics.median(phase6_delta_mi) if phase6_delta_mi else 0.0
    print(f"    Phase 6: survival={phase6_survival:.1f}%, median delta_mi={phase6_med:.3f}")

    # Reference: P1 and P2 from stage_d
    p1_path = DATA_DIR / "phase_1" / "logs" / "metrics_summary.parquet"
    p1_median = _median_delta_mi_from_metrics(p1_path)
    p2_median = _median_delta_mi_from_metrics(P2_METRICS)
    print(f"    Reference P1: median delta_mi={p1_median:.3f}")
    print(f"    Reference P2: median delta_mi={p2_median:.3f}")

    return {
        "Phase 5 (cap-matched P1)": {
            "survival_rate": phase5_survival,
            "median_delta_mi": phase5_med,
        },
        "Phase 6 (rand-encoding P2)": {
            "survival_rate": phase6_survival,
            "median_delta_mi": phase6_med,
        },
        "Phase 1 (reference)": {"median_delta_mi": p1_median},
        "Phase 2 (reference)": {"median_delta_mi": p2_median},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Post-Hoc Analysis — stage_d data")
    print("=" * 60)

    # Step 1: Get top-50 P2 seeds
    print("\nExtracting top-50 Phase 2 rule seeds...")
    top_seeds = select_top_rules_by_delta_mi(P2_METRICS, P2_RULES, top_k=50)
    print(f"  Found {len(top_seeds)} seeds")

    # Step 2: Load snapshots for top-50 P2 rules
    print("\nLoading final-step snapshots for top-50 P2 rules...")
    p2_rule_ids = _rule_ids_for_seeds(2, top_seeds)
    p2_snapshots = _load_final_snapshots(P2_SIM_LOG, p2_rule_ids)
    print(f"  Loaded {len(p2_snapshots)} snapshots")

    # Run analyses
    halt_results = run_halt_window_analysis(top_seeds)
    null_results = run_alt_null_analysis(p2_snapshots)
    scramble_results = run_spatial_scramble_analysis(p2_snapshots)
    te_results = run_transfer_entropy_analysis(top_seeds)
    cap_results = run_capacity_matched_analysis()

    # Print LaTeX-ready summary
    print("\n" + "=" * 60)
    print("LATEX-READY VALUES")
    print("=" * 60)

    print("\n--- Table D: Halt Window Sensitivity ---")
    for window in HALT_WINDOWS:
        r = halt_results[window]
        print(f"{window}  & {r['survival_rate']:.1f}\\% & {r['median_delta_mi']:.3f} \\\\")

    print("\n--- Table G: Alternative Null Models ---")
    print(f"State shuffle & {null_results['state_shuffle']:.4f} \\\\")
    print(f"Block shuffle & {null_results['block_shuffle']:.4f} \\\\")
    print(f"Fixed-marginal & {null_results['fixed_marginal']:.4f} \\\\")

    print("\n--- Section H: Spatial Scrambling ---")
    print(f"Mean observed MI (top-50 P2): {scramble_results['observed_mi']:.4f} bits")
    print(f"Mean scrambled MI (top-50 P2): {scramble_results['scrambled_mi']:.4f} bits")

    print("\n--- Section I: Transfer Entropy ---")
    for cond, te in te_results.items():
        print(f"{cond}: median TE = {te:.4f} bits")

    print("\n--- Section J: Capacity-Matched Controls ---")
    for cond, vals in cap_results.items():
        parts = [f"median delta_mi={vals['median_delta_mi']:.3f}"]
        if "survival_rate" in vals:
            parts.append(f"survival={vals['survival_rate']:.1f}%")
        print(f"  {cond}: {', '.join(parts)}")

    # Dump all results as JSON for programmatic use
    all_results = {
        "halt_window": {str(k): v for k, v in halt_results.items()},
        "alt_nulls": null_results,
        "spatial_scramble": scramble_results,
        "transfer_entropy": te_results,
        "capacity_matched": cap_results,
    }
    output_json = PROJECT_ROOT / "data" / "post_hoc" / "analysis_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_json}")


if __name__ == "__main__":
    main()
