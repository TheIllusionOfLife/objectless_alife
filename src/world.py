from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.rules import (
    ObservationPhase,
    compute_phase1_index,
    compute_phase2_index,
    dominant_neighbor_state,
)


@dataclass(frozen=False)
class Agent:
    agent_id: int
    x: int
    y: int
    state: int


@dataclass(frozen=True)
class WorldConfig:
    grid_width: int = 20
    grid_height: int = 20
    num_agents: int = 30
    num_states: int = 4
    steps: int = 200


class World:
    def __init__(self, config: WorldConfig, sim_seed: int) -> None:
        self.config = config
        self.rng = Random(sim_seed)
        self.agents = self._initialize_agents()
        self._occupancy = {(agent.x, agent.y): agent.agent_id for agent in self.agents}

    @classmethod
    def from_agents(cls, config: WorldConfig, agents: list[Agent], sim_seed: int) -> "World":
        if len(agents) != config.num_agents:
            raise ValueError("len(agents) must equal config.num_agents")
        world = cls.__new__(cls)
        world.config = config
        world.rng = Random(sim_seed)
        world.agents = [Agent(a.agent_id, a.x, a.y, a.state) for a in agents]
        if len({(a.x, a.y) for a in world.agents}) != len(world.agents):
            raise ValueError("Agents cannot overlap")
        world._occupancy = {(agent.x, agent.y): agent.agent_id for agent in world.agents}
        return world

    def _initialize_agents(self) -> list[Agent]:
        positions = set()
        agents: list[Agent] = []
        max_cells = self.config.grid_width * self.config.grid_height
        if self.config.num_agents > max_cells:
            raise ValueError("num_agents cannot exceed number of grid cells")

        for agent_id in range(self.config.num_agents):
            while True:
                x = self.rng.randrange(self.config.grid_width)
                y = self.rng.randrange(self.config.grid_height)
                if (x, y) not in positions:
                    positions.add((x, y))
                    break
            state = self.rng.randrange(self.config.num_states)
            agents.append(Agent(agent_id=agent_id, x=x, y=y, state=state))
        return agents

    def get_agent(self, agent_id: int) -> Agent:
        return self.agents[agent_id]

    def snapshot(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (a.agent_id, a.x, a.y, a.state) for a in sorted(self.agents, key=lambda a: a.agent_id)
        )

    def state_vector(self) -> list[int]:
        return [a.state for a in sorted(self.agents, key=lambda a: a.agent_id)]

    def apply_action(self, agent_id: int, action: int) -> None:
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
        neighbors = [
            (x, (y - 1) % self.config.grid_height),
            (x, (y + 1) % self.config.grid_height),
            ((x - 1) % self.config.grid_width, y),
            ((x + 1) % self.config.grid_width, y),
        ]
        states: list[int] = []
        for nx, ny in neighbors:
            agent_id = self._occupancy.get((nx, ny))
            if agent_id is not None:
                states.append(self.agents[agent_id].state)
        return states

    @staticmethod
    def _delta_for_action(action: int) -> tuple[int, int]:
        if action == 0:  # up
            return (0, -1)
        if action == 1:  # down
            return (0, 1)
        if action == 2:  # left
            return (-1, 0)
        if action == 3:  # right
            return (1, 0)
        return (0, 0)
