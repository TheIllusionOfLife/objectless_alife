"""Backward-compatibility shim: metrics module moved to metrics/ subpackage."""

from objectless_alife.metrics.information import (
    block_ncd as block_ncd,
)
from objectless_alife.metrics.information import (
    block_shuffle_null_mi as block_shuffle_null_mi,
)
from objectless_alife.metrics.information import (
    compression_ratio as compression_ratio,
)
from objectless_alife.metrics.information import (
    fixed_marginal_null_mi as fixed_marginal_null_mi,
)
from objectless_alife.metrics.information import (
    neighbor_mutual_information as neighbor_mutual_information,
)
from objectless_alife.metrics.information import (
    neighbor_transfer_entropy as neighbor_transfer_entropy,
)
from objectless_alife.metrics.information import (
    normalized_hamming_distance as normalized_hamming_distance,
)
from objectless_alife.metrics.information import (
    serialize_snapshot as serialize_snapshot,
)
from objectless_alife.metrics.information import (
    shuffle_null_mi as shuffle_null_mi,
)
from objectless_alife.metrics.information import (
    spatial_scramble_mi as spatial_scramble_mi,
)
from objectless_alife.metrics.information import (
    state_entropy as state_entropy,
)
from objectless_alife.metrics.information import (
    transfer_entropy_excess as transfer_entropy_excess,
)
from objectless_alife.metrics.information import (
    transfer_entropy_shuffle_null as transfer_entropy_shuffle_null,
)
from objectless_alife.metrics.spatial import (
    cluster_count_by_state as cluster_count_by_state,
)
from objectless_alife.metrics.spatial import (
    morans_i_occupied as morans_i_occupied,
)
from objectless_alife.metrics.spatial import (
    neighbor_pair_count as neighbor_pair_count,
)
from objectless_alife.metrics.spatial import (
    same_state_adjacency_fraction as same_state_adjacency_fraction,
)
from objectless_alife.metrics.temporal import (
    action_entropy as action_entropy,
)
from objectless_alife.metrics.temporal import (
    action_entropy_variance as action_entropy_variance,
)
from objectless_alife.metrics.temporal import (
    phase_transition_max_delta as phase_transition_max_delta,
)
from objectless_alife.metrics.temporal import (
    quasi_periodicity_peak_count as quasi_periodicity_peak_count,
)
