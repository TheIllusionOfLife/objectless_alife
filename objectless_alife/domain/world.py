"""Toroidal grid world with collision-constrained agent updates."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from objectless_alife.config.constants import (
    CLOCK_PERIOD,
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_AGENTS,
    NUM_STATES,
    NUM_STEPS,
)
from objectless_alife.domain.rules import (
    ObservationPhase,
    compute_capacity_matched_index,
    compute_control_index,
    compute_phase1_index,
    compute_phase2_index,
    dominant_neighbor_state,
)

NUM_ACTIONS = 9
"""Total action count: 4 movement + 4 state-change + 1 no-op."""

NOOP_ACTION = NUM_ACTIONS - 1
"""Explicit no-op action ID."""


@dataclass
class Agent:
    """Mutable agent state tracked in the simulation world."""

    agent_id: int
    x: int
    y: int
    state: int


@dataclass(frozen=True)
class WorldConfig:
    """Simulation parameters for the grid world."""

    grid_width: int = GRID_WIDTH
    grid_height: int = GRID_HEIGHT
    num_agents: int = NUM_AGENTS
    num_states: int = NUM_STATES
    steps: int = NUM_STEPS


class World:
    """Toroidal grid world with collision-constrained agent updates."""

    def __init__(self, config: WorldConfig, sim_seed: int) -> None:
        """Create a seeded world with randomly initialized non-overlapping agents."""
        self.config = config
        self.rng = Random(sim_seed)
        self.agents = self._initialize_agents()
        self._occupancy = {(agent.x, agent.y): agent.agent_id for agent in self.agents}
        self._neighbor_cache = self._build_neighbor_cache()

    def _build_neighbor_cache(self) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
        """Precompute neighbor coordinates for all grid cells."""
        cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        w, h = self.config.grid_width, self.config.grid_height
        for x in range(w):
            for y in range(h):
                cache[(x, y)] = (
                    (x, (y - 1) % h),
                    (x, (y + 1) % h),
                    ((x - 1) % w, y),
                    ((x + 1) % w, y),
                )
        return cache

    @classmethod
    def from_agents(cls, config: WorldConfig, agents: list[Agent], sim_seed: int) -> "World":
        """Construct world from explicit agents, requiring contiguous IDs 0..n-1."""
        if len(agents) != config.num_agents:
            raise ValueError("len(agents) must equal config.num_agents")
        world = cls.__new__(cls)
        world.config = config
        world.rng = Random(sim_seed)

        sorted_agents = sorted(agents, key=lambda a: a.agent_id)
        expected_ids = list(range(config.num_agents))
        actual_ids = [a.agent_id for a in sorted_agents]
        if actual_ids != expected_ids:
            raise ValueError("agent_ids must be contiguous 0..num_agents-1")

        for a in sorted_agents:
            if not (0 <= a.x < config.grid_width and 0 <= a.y < config.grid_height):
                raise ValueError(
                    f"Agent {a.agent_id} position ({a.x}, {a.y}) is outside the grid "
                    f"({config.grid_width}x{config.grid_height})"
                )
        world.agents = [Agent(a.agent_id, a.x, a.y, a.state) for a in sorted_agents]
        if len({(a.x, a.y) for a in world.agents}) != len(world.agents):
            raise ValueError("Agents cannot overlap")
        world._occupancy = {(agent.x, agent.y): agent.agent_id for agent in world.agents}
        world._neighbor_cache = world._build_neighbor_cache()
        return world

    def _initialize_agents(self) -> list[Agent]:
        """Initialize unique agent positions and random internal states."""
        max_cells = self.config.grid_width * self.config.grid_height
        if self.config.num_agents > max_cells:
            raise ValueError("num_agents cannot exceed number of grid cells")

        agents: list[Agent] = []

        # Use dense-mode sampling to avoid retry-heavy rejection loops.
        if self.config.num_agents * 2 >= max_cells:
            indices = self.rng.sample(range(max_cells), k=self.config.num_agents)
            coords = [
                (idx % self.config.grid_width, idx // self.config.grid_width) for idx in indices
            ]
        else:
            positions: set[tuple[int, int]] = set()
            coords = []
            for _ in range(self.config.num_agents):
                while True:
                    x = self.rng.randrange(self.config.grid_width)
                    y = self.rng.randrange(self.config.grid_height)
                    if (x, y) not in positions:
                        positions.add((x, y))
                        coords.append((x, y))
                        break

        for agent_id, (x, y) in enumerate(coords):
            state = self.rng.randrange(self.config.num_states)
            agents.append(Agent(agent_id=agent_id, x=x, y=y, state=state))

        return agents

    def get_agent(self, agent_id: int) -> Agent:
        """Return agent by ID (IDs must be contiguous and index-aligned)."""
        return self.agents[agent_id]

    def snapshot(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return immutable snapshot of (agent_id, x, y, state) in agent-id order."""
        return tuple((a.agent_id, a.x, a.y, a.state) for a in self.agents)

    def state_vector(self) -> list[int]:
        """Return agent states ordered by ascending agent_id."""
        return [a.state for a in self.agents]

    def apply_action(self, agent_id: int, action: int) -> None:
        """Apply one action for one agent.

        Action 8 is an explicit no-op. Values outside [0, 8] are invalid.
        """
        if not 0 <= action <= NOOP_ACTION:
            raise ValueError(f"action must be in [0, {NOOP_ACTION}]")

        agent = self.agents[agent_id]
        if action in {0, 1, 2, 3}:
            dx, dy = self._delta_for_action(action)
            new_x = (agent.x + dx) % self.config.grid_width
            new_y = (agent.y + dy) % self.config.grid_height
            occ = self._occupancy.get((new_x, new_y))
            if occ is None or occ == agent_id:
                del self._occupancy[(agent.x, agent.y)]
                agent.x = new_x
                agent.y = new_y
                self._occupancy[(agent.x, agent.y)] = agent_id
            return

        if 4 <= action <= 7:
            agent.state = action - 4

    def step(
        self,
        rule_table: list[int],
        phase: ObservationPhase,
        step_number: int = 0,
        update_mode: object = None,
    ) -> list[int]:
        """Advance one simulation step and return intended actions."""
        from objectless_alife.config.types import UpdateMode

        if update_mode is None:
            update_mode = UpdateMode.SEQUENTIAL
        if update_mode == UpdateMode.SEQUENTIAL:
            return self._step_sequential(rule_table, phase, step_number)
        if update_mode == UpdateMode.SYNCHRONOUS:
            return self._step_synchronous(rule_table, phase, step_number)
        raise ValueError(f"Unhandled update mode: {update_mode}")

    def _step_sequential(
        self,
        rule_table: list[int],
        phase: ObservationPhase,
        step_number: int,
    ) -> list[int]:
        """Advance one random-sequential simulation step."""
        order = list(range(len(self.agents)))
        self.rng.shuffle(order)
        actions = [NOOP_ACTION] * len(self.agents)
        state_by_agent_id = {a.agent_id: a.state for a in self.agents}

        for agent_id in order:
            agent = self.agents[agent_id]
            action = self._select_action_for_agent(
                agent=agent,
                phase=phase,
                step_number=step_number,
                rule_table=rule_table,
                occupancy=self._occupancy,
                state_by_agent_id=state_by_agent_id,
            )
            actions[agent_id] = action
            self.apply_action(agent_id=agent_id, action=action)
            state_by_agent_id[agent_id] = self.agents[agent_id].state

        return actions

    def _step_synchronous(
        self,
        rule_table: list[int],
        phase: ObservationPhase,
        step_number: int,
    ) -> list[int]:
        """Advance one synchronous simulation step.

        Decisions are computed from a frozen step-start snapshot, then applied.
        """
        order = list(range(len(self.agents)))
        self.rng.shuffle(order)
        frozen_occupancy = dict(self._occupancy)
        frozen_states = {a.agent_id: a.state for a in self.agents}
        actions = [NOOP_ACTION] * len(self.agents)

        for agent_id in order:
            agent = self.agents[agent_id]
            actions[agent_id] = self._select_action_for_agent(
                agent=agent,
                phase=phase,
                step_number=step_number,
                rule_table=rule_table,
                occupancy=frozen_occupancy,
                state_by_agent_id=frozen_states,
            )

        for agent_id in order:
            self.apply_action(agent_id=agent_id, action=actions[agent_id])
        return actions

    def _select_action_for_agent(
        self,
        *,
        agent: Agent,
        phase: ObservationPhase,
        step_number: int,
        rule_table: list[int],
        occupancy: dict[tuple[int, int], int],
        state_by_agent_id: dict[int, int],
    ) -> int:
        """Select an action for one agent from the provided observation snapshot."""
        if phase == ObservationPhase.RANDOM_WALK:
            return self.rng.randrange(NUM_ACTIONS)

        neighbor_states = self._neighbor_states_from_observation(
            x=agent.x,
            y=agent.y,
            occupancy=occupancy,
            state_by_agent_id=state_by_agent_id,
        )
        neighbor_count = len(neighbor_states)
        self_state = state_by_agent_id[agent.agent_id]

        if phase == ObservationPhase.PHASE1_DENSITY:
            index = compute_phase1_index(self_state, neighbor_count)
        elif phase == ObservationPhase.CONTROL_DENSITY_CLOCK:
            step_mod = step_number % CLOCK_PERIOD
            index = compute_control_index(self_state, neighbor_count, step_mod)
        elif phase == ObservationPhase.PHASE2_PROFILE:
            dom = dominant_neighbor_state(neighbor_states)
            index = compute_phase2_index(self_state, neighbor_count, dom)
        elif phase == ObservationPhase.PHASE1_CAPACITY_MATCHED:
            dom = dominant_neighbor_state(neighbor_states)
            index = compute_capacity_matched_index(self_state, neighbor_count, dom)
        elif phase == ObservationPhase.PHASE2_RANDOM_ENCODING:
            dom = dominant_neighbor_state(neighbor_states)
            index = compute_phase2_index(self_state, neighbor_count, dom)
        else:
            raise ValueError(f"Unhandled observation phase: {phase}")
        return rule_table[index]

    def _neighbor_states(self, x: int, y: int) -> list[int]:
        """Collect occupied-neighbor states from 4-neighborhood on torus grid."""
        return self._neighbor_states_from_observation(
            x=x,
            y=y,
            occupancy=self._occupancy,
            state_by_agent_id={a.agent_id: a.state for a in self.agents},
        )

    def _neighbor_states_from_observation(
        self,
        *,
        x: int,
        y: int,
        occupancy: dict[tuple[int, int], int],
        state_by_agent_id: dict[int, int],
    ) -> list[int]:
        """Collect occupied-neighbor states from an explicit snapshot."""
        neighbors = self._neighbor_cache[(x, y)]
        return [
            state_by_agent_id[agent_id]
            for nx, ny in neighbors
            if (agent_id := occupancy.get((nx, ny))) is not None
        ]

    @staticmethod
    def _delta_for_action(action: int) -> tuple[int, int]:
        """Map movement action ID to coordinate delta."""
        deltas = {
            0: (0, -1),  # up
            1: (0, 1),  # down
            2: (-1, 0),  # left
            3: (1, 0),  # right
        }
        return deltas.get(action, (0, 0))
