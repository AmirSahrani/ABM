import mesa as ms
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class Tribe():
    id: int
    total_spice: int


class Nomad(ms.Agent):
    def __init__(self, id: int, model: ms.Model, pos: tuple, spice: int, vision: int, tribe: Tribe, lamb: float):
        super().__init__(id, model)
        self.pos = pos
        self.spice = spice
        self.vision = vision
        self.tribe = tribe
        self.lamb = lamb
        self.hardship = self.calculate_hardship()
        self.legitimacy = self.calculate_legitimacy()

    def calculate_hardship(self):
        return 1 / math.exp(self.spice * self.lamb)

    def calculate_legitimacy(self):
        legitimacy = {}
        for other_tribe in self.model.tribes:
            if other_tribe.id != self.tribe.id:
                legitimacy[other_tribe.id] = 1 / math.exp((self.tribe.total_spice - other_tribe.total_spice) * self.lamb)
        return legitimacy

    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, Nomad) for agent in this_cell)

    def get_spice(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if isinstance(agent, Spice):
                return agent

    def move(self):
        visible_positions = [
            i
            for i in self.model.grid.get_neighborhood(
                self.pos, False, False, self.vision
            )
        ]
        visible_positions = [i for i in visible_positions if not self.is_occupied(i)]

        if not visible_positions:
            return

        visible_positions.append(self.pos)

        spice_levels = [self.get_spice(p).spice if self.get_spice(p) else 0 for p in visible_positions]
        interaction_scores = []

        for p in visible_positions:
            cellmates = self.model.grid.get_cell_list_contents([p])
            other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != self]
            interaction_score = 0
            if not other_nomads:
                interaction_score += spice_levels[visible_positions.index(p)] * 10
            for other in other_nomads:
                if self.tribe != other.tribe:
                    interaction_score -= other.spice if other.spice > self.spice else 0
                else:
                    if other.spice > self.spice:
                        interaction_score += other.spice
                    else:
                        interaction_score -= other.spice
            interaction_scores.append(interaction_score)

        preferences = [spice + interaction for spice, interaction in zip(spice_levels, interaction_scores)]
        total_preference = sum(preferences)

        if total_preference > 0:
            probabilities = [pref / total_preference for pref in preferences]
        else:
            probabilities = [1 / len(preferences)] * len(preferences)

        chosen_pos = self.random.choices(visible_positions, probabilities)[0]

        immediate_neighbors = [
            (self.pos[0] + dx, self.pos[1] + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx, dy) != (0, 0)
        ]

        immediate_neighbors = [
            pos for pos in immediate_neighbors
            if self.model.grid.out_of_bounds(pos) == False and not self.is_occupied(pos)
        ]

        if not immediate_neighbors:
            return

        best_move = min(immediate_neighbors, key=lambda pos: (pos[0] - chosen_pos[0])**2 + (pos[1] - chosen_pos[1])**2)

        self.model.grid.move_agent(self, best_move)

    def sniff(self):
        spice_patch = self.get_spice(self.pos)
        if spice_patch is not None:
            self.spice += 1
            spice_patch.spice -= 1
            if spice_patch.spice <= 0:
                self.model.remove_agent(spice_patch)
        else:
            pass

    def trade(self, trade_percentage=0.1):
        neighbor_agents = self.model.grid.get_neighbors(self.pos, True, True, self.vision)
        tribe_neighbors = [a for a in neighbor_agents if isinstance(a, Nomad) and a.tribe == self.tribe]

        for other_nomad in tribe_neighbors:
            if other_nomad != self:
                trade_amount_self = int(self.spice * trade_percentage)
                trade_amount_other = int(other_nomad.spice * trade_percentage)

                self.spice = self.spice - trade_amount_self + trade_amount_other
                other_nomad.spice = other_nomad.spice - trade_amount_other + trade_amount_self

                self.model.record_trade()

    def step(self):

        self.move()
        self.sniff()
        self.trade()
        self.spice -= 1
        if self.spice < 0:
            self.model.remove_agent(self)
        # TODO split agent

class Spice(ms.Agent):
    def __init__(self, id: int, pos: tuple, model: ms.Model, max_spice: int):
        super().__init__(id, model)
        self.pos = pos
        self.spice = max_spice
        self.max_spice = max_spice

    def step(self):
        if self.spice == 0:
            self.model.remove_agent(self)