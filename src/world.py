from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.rules import (
    ObservationPhase,
    compute_phase1_index,
    compute_phase2_index,
    dominant_neighbor_state,
)


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

    grid_width: int = 20
    grid_height: int = 20
    num_agents: int = 30
    num_states: int = 4
    steps: int = 200


class World:
    """Toroidal grid world with collision-constrained agent updates."""

    def __init__(self, config: WorldConfig, sim_seed: int) -> None:
        """Create a seeded world with randomly initialized non-overlapping agents."""
        self.config = config
        self.rng = Random(sim_seed)
        self.agents = self._initialize_agents()
        self._occupancy = {(agent.x, agent.y): agent.agent_id for agent in self.agents}

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

        world.agents = [Agent(a.agent_id, a.x, a.y, a.state) for a in sorted_agents]
        if len({(a.x, a.y) for a in world.agents}) != len(world.agents):
            raise ValueError("Agents cannot overlap")
        world._occupancy = {(agent.x, agent.y): agent.agent_id for agent in world.agents}
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
        if not 0 <= action <= 8:
            raise ValueError("action must be in [0, 8]")

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

    def step(self, rule_table: list[int], phase: ObservationPhase) -> list[int]:
        """Advance one random-sequential simulation step and return intended actions."""
        order = list(range(len(self.agents)))
        self.rng.shuffle(order)
        actions = [8] * len(self.agents)

        for agent_id in order:
            agent = self.agents[agent_id]
            neighbor_states = self._neighbor_states(agent.x, agent.y)
            neighbor_count = len(neighbor_states)

            if phase == ObservationPhase.PHASE1_DENSITY:
                index = compute_phase1_index(agent.state, neighbor_count)
            else:
                dom = dominant_neighbor_state(neighbor_states)
                index = compute_phase2_index(agent.state, neighbor_count, dom)

            action = rule_table[index]
            actions[agent_id] = action
            self.apply_action(agent_id=agent_id, action=action)

        return actions

    def _neighbor_states(self, x: int, y: int) -> list[int]:
        """Collect occupied-neighbor states from 4-neighborhood on torus grid."""
        neighbors = [
            (x, (y - 1) % self.config.grid_height),
            (x, (y + 1) % self.config.grid_height),
            ((x - 1) % self.config.grid_width, y),
            ((x + 1) % self.config.grid_width, y),
        ]
        return [
            self.agents[agent_id].state
            for nx, ny in neighbors
            if (agent_id := self._occupancy.get((nx, ny))) is not None
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
