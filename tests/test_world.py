from src.rules import ObservationPhase
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
