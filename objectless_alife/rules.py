"""Backward-compatibility shim: rules module moved to domain/rules.py."""

from objectless_alife.config.constants import CLOCK_PERIOD as CLOCK_PERIOD
from objectless_alife.domain.rules import (
    ObservationPhase as ObservationPhase,
)
from objectless_alife.domain.rules import (
    compute_capacity_matched_index as compute_capacity_matched_index,
)
from objectless_alife.domain.rules import (
    compute_control_index as compute_control_index,
)
from objectless_alife.domain.rules import (
    compute_phase1_index as compute_phase1_index,
)
from objectless_alife.domain.rules import (
    compute_phase2_index as compute_phase2_index,
)
from objectless_alife.domain.rules import (
    dominant_neighbor_state as dominant_neighbor_state,
)
from objectless_alife.domain.rules import (
    generate_rule_table as generate_rule_table,
)
from objectless_alife.domain.rules import (
    rule_table_size as rule_table_size,
)
