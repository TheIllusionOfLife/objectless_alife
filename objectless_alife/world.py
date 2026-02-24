"""Backward-compatibility shim: world module moved to domain/world.py."""

from objectless_alife.config.constants import CLOCK_PERIOD as CLOCK_PERIOD
from objectless_alife.domain.world import (
    NOOP_ACTION as NOOP_ACTION,
)
from objectless_alife.domain.world import (
    NUM_ACTIONS as NUM_ACTIONS,
)
from objectless_alife.domain.world import (
    Agent as Agent,
)
from objectless_alife.domain.world import (
    World as World,
)
from objectless_alife.domain.world import (
    WorldConfig as WorldConfig,
)
