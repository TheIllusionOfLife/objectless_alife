import pytest

from src.rules import ObservationPhase, generate_rule_table
from src.world import Agent, World, WorldConfig


def test_torus_movement_wraps() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=1, steps=1)
    world = World.from_agents(config, [Agent(agent_id=0, x=0, y=0, state=0)], sim_seed=1)

    world.apply_action(agent_id=0, action=2)  # left
    assert world.get_agent(0).x == 4

    world.apply_action(agent_id=0, action=0)  # up
    assert world.get_agent(0).y == 4


def test_move_fails_when_target_occupied() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=2, steps=1)
    world = World.from_agents(
        config,
        [Agent(agent_id=0, x=0, y=0, state=0), Agent(agent_id=1, x=1, y=0, state=1)],
        sim_seed=1,
    )

    world.apply_action(agent_id=0, action=3)  # right into occupied cell
    assert world.get_agent(0).x == 0
    assert world.get_agent(0).y == 0


def test_step_is_deterministic_for_same_seed() -> None:
    config = WorldConfig(grid_width=10, grid_height=10, num_agents=4, steps=5)
    w1 = World(config=config, sim_seed=123)
    w2 = World(config=config, sim_seed=123)

    # no-op table for phase1
    rule_table = [8] * 20
    for _ in range(3):
        w1.step(rule_table, ObservationPhase.PHASE1_DENSITY)
        w2.step(rule_table, ObservationPhase.PHASE1_DENSITY)

    assert w1.snapshot() == w2.snapshot()


def test_sequential_update_changes_later_observation() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=2, steps=1)
    world = World.from_agents(
        config,
        [Agent(agent_id=0, x=0, y=0, state=0), Agent(agent_id=1, x=2, y=0, state=0)],
        sim_seed=42,
    )

    # both move right if neighbor_count=0, else no-op
    table = [3, 8, 8, 8, 8] * 4
    world.step(table, ObservationPhase.PHASE1_DENSITY)

    # deterministic order from seed should keep positions valid and non-overlapping
    agents = [world.get_agent(0), world.get_agent(1)]
    assert len({(a.x, a.y) for a in agents}) == 2


def test_from_agents_sorts_and_validates_agent_ids() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=3, steps=1)
    world = World.from_agents(
        config,
        [
            Agent(agent_id=2, x=2, y=0, state=0),
            Agent(agent_id=0, x=0, y=0, state=0),
            Agent(agent_id=1, x=1, y=0, state=0),
        ],
        sim_seed=1,
    )

    assert [agent.agent_id for agent in world.agents] == [0, 1, 2]


def test_from_agents_rejects_non_contiguous_ids() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=3, steps=1)

    with pytest.raises(ValueError):
        World.from_agents(
            config,
            [
                Agent(agent_id=0, x=0, y=0, state=0),
                Agent(agent_id=2, x=1, y=0, state=0),
                Agent(agent_id=3, x=2, y=0, state=0),
            ],
            sim_seed=1,
        )


def test_apply_action_rejects_invalid_action() -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=1, steps=1)
    world = World.from_agents(config, [Agent(agent_id=0, x=0, y=0, state=0)], sim_seed=1)

    with pytest.raises(ValueError):
        world.apply_action(agent_id=0, action=9)


def test_step_control_phase_produces_valid_actions() -> None:
    config = WorldConfig(grid_width=10, grid_height=10, num_agents=4, steps=5)
    world = World(config=config, sim_seed=99)
    rule_table = [8] * 100  # all no-ops
    actions = world.step(rule_table, ObservationPhase.CONTROL_DENSITY_CLOCK, step_number=7)
    assert len(actions) == 4
    assert all(a == 8 for a in actions)


def test_step_control_phase_deterministic() -> None:
    config = WorldConfig(grid_width=10, grid_height=10, num_agents=4, steps=5)
    w1 = World(config=config, sim_seed=42)
    w2 = World(config=config, sim_seed=42)
    rule_table = generate_rule_table(ObservationPhase.CONTROL_DENSITY_CLOCK, seed=10)
    for step in range(3):
        w1.step(rule_table, ObservationPhase.CONTROL_DENSITY_CLOCK, step_number=step)
        w2.step(rule_table, ObservationPhase.CONTROL_DENSITY_CLOCK, step_number=step)
    assert w1.snapshot() == w2.snapshot()


def test_step_existing_phases_ignore_step_number() -> None:
    """Existing phases must produce identical results regardless of step_number."""
    config = WorldConfig(grid_width=10, grid_height=10, num_agents=4, steps=5)
    w1 = World(config=config, sim_seed=42)
    w2 = World(config=config, sim_seed=42)
    rule_table = [8] * 20
    w1.step(rule_table, ObservationPhase.PHASE1_DENSITY, step_number=0)
    w2.step(rule_table, ObservationPhase.PHASE1_DENSITY, step_number=99)
    assert w1.snapshot() == w2.snapshot()


def test_snapshot_and_state_vector_do_not_depend_on_sorted_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WorldConfig(grid_width=5, grid_height=5, num_agents=2, steps=1)
    world = World.from_agents(
        config,
        [Agent(agent_id=0, x=1, y=1, state=2), Agent(agent_id=1, x=2, y=2, state=3)],
        sim_seed=1,
    )

    def _raise_sorted(*args: object, **kwargs: object) -> object:
        raise AssertionError("sorted must not be called")

    monkeypatch.setattr("builtins.sorted", _raise_sorted)

    assert world.snapshot() == ((0, 1, 1, 2), (1, 2, 2, 3))
    assert world.state_vector() == [2, 3]
