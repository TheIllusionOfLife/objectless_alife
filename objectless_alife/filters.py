"""Backward-compatibility shim: filters module moved to domain/filters.py."""

from objectless_alife.config.constants import ACTION_SPACE_SIZE as ACTION_SPACE_SIZE
from objectless_alife.domain.filters import (
    HaltDetector as HaltDetector,
)
from objectless_alife.domain.filters import (
    LowActivityDetector as LowActivityDetector,
)
from objectless_alife.domain.filters import (
    ShortPeriodDetector as ShortPeriodDetector,
)
from objectless_alife.domain.filters import (
    StateUniformDetector as StateUniformDetector,
)
from objectless_alife.domain.filters import (
    TerminationReason as TerminationReason,
)
