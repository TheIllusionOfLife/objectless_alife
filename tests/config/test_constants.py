from objectless_alife.config.constants import (
    ACTION_SPACE_SIZE,
    BLOCK_NCD_WINDOW,
    CLOCK_PERIOD,
    FLUSH_THRESHOLD,
    GRID_HEIGHT,
    GRID_WIDTH,
    HALT_WINDOW,
    MAX_EXPERIMENT_WORK_UNITS,
    NUM_AGENTS,
    NUM_STATES,
    NUM_STEPS,
    SHUFFLE_NULL_N,
)


def test_grid_dimensions_are_positive_ints() -> None:
    assert isinstance(GRID_WIDTH, int) and GRID_WIDTH > 0
    assert isinstance(GRID_HEIGHT, int) and GRID_HEIGHT > 0


def test_num_agents_fits_in_grid() -> None:
    assert isinstance(NUM_AGENTS, int) and NUM_AGENTS > 0
    assert NUM_AGENTS < GRID_WIDTH * GRID_HEIGHT


def test_num_states_is_at_least_two() -> None:
    assert isinstance(NUM_STATES, int) and NUM_STATES >= 2


def test_num_steps_is_positive() -> None:
    assert isinstance(NUM_STEPS, int) and NUM_STEPS > 0


def test_halt_window_less_than_num_steps() -> None:
    assert isinstance(HALT_WINDOW, int) and HALT_WINDOW > 0
    assert HALT_WINDOW < NUM_STEPS


def test_shuffle_null_n_is_positive() -> None:
    assert isinstance(SHUFFLE_NULL_N, int) and SHUFFLE_NULL_N > 0


def test_block_ncd_window_is_positive() -> None:
    assert isinstance(BLOCK_NCD_WINDOW, int) and BLOCK_NCD_WINDOW > 0


def test_clock_period_is_positive() -> None:
    assert isinstance(CLOCK_PERIOD, int) and CLOCK_PERIOD > 0


def test_action_space_size_matches_spec() -> None:
    # 4 movement + 4 state-change + 1 no-op = 9
    assert ACTION_SPACE_SIZE == 9


def test_flush_threshold_is_power_of_two_or_large() -> None:
    assert isinstance(FLUSH_THRESHOLD, int) and FLUSH_THRESHOLD >= 1024


def test_max_experiment_work_units_is_large() -> None:
    assert isinstance(MAX_EXPERIMENT_WORK_UNITS, int)
    assert MAX_EXPERIMENT_WORK_UNITS >= 1_000_000
