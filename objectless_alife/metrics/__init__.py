"""Simulation metrics: spatial, temporal, and information-theoretic measures."""

from objectless_alife.metrics.information import (
    block_ncd,
    block_shuffle_null_mi,
    compression_ratio,
    fixed_marginal_null_mi,
    neighbor_mutual_information,
    neighbor_transfer_entropy,
    normalized_hamming_distance,
    serialize_snapshot,
    shuffle_null_mi,
    spatial_scramble_mi,
    state_entropy,
    transfer_entropy_excess,
    transfer_entropy_shuffle_null,
)
from objectless_alife.metrics.spatial import (
    cluster_count_by_state,
    morans_i_occupied,
    neighbor_pair_count,
    same_state_adjacency_fraction,
)
from objectless_alife.metrics.temporal import (
    action_entropy,
    action_entropy_variance,
    phase_transition_max_delta,
    quasi_periodicity_peak_count,
)

__all__ = [
    "action_entropy",
    "action_entropy_variance",
    "block_ncd",
    "block_shuffle_null_mi",
    "cluster_count_by_state",
    "compression_ratio",
    "fixed_marginal_null_mi",
    "morans_i_occupied",
    "neighbor_mutual_information",
    "neighbor_pair_count",
    "neighbor_transfer_entropy",
    "normalized_hamming_distance",
    "phase_transition_max_delta",
    "quasi_periodicity_peak_count",
    "same_state_adjacency_fraction",
    "serialize_snapshot",
    "shuffle_null_mi",
    "spatial_scramble_mi",
    "state_entropy",
    "transfer_entropy_excess",
    "transfer_entropy_shuffle_null",
]
