import mesa as ms
import nashpy as nash
import numpy as np
from dataclasses import dataclass
import random
import math


@dataclass
class Tribe():
    id: int
    total_spice: int


class Nomad(ms.Agent):
    """
    A Nomad is an agent foraging for spice, it has the following attributes:

    [pos (int, int)]: and x,y coordinate representing its location on a finite grid
    [spice int]: Their spice level, if it gets below a threshold it dies
    [vision int]: Their range of vision, they can see spice and enemy Nomads within their vision
    [tribe Tribe]: a tribe they are associated with, they only attack enemy tribes
    [hardship float]: their spice level, defined as 1/exp(spice*lamb), 1 represents maximum hardship
    [legitimacy dict(Tribe, float)]: a dict containing the legitimacy felt toward each other tribe 1/exp((self.tribe.total_spice - other.tribe.total_spice)*lamb).
    [id int]: Unique id to represent them in the model
    [model ms.Model]: The model they are associated with
    """

    def __init__(self, id: int, model: ms.Model, pos: tuple, spice: int, vision: int, tribe: Tribe, metabolism: float):
        super().__init__(id, model)
        self.pos = pos
        self.spice = spice
        self.vision = vision
        self.tribe = tribe
        self.metabolism = metabolism
        # self.hardship = self.calculate_hardship()
        # self.legitimacy = self.calculate_legitimacy()

    # def calculate_hardship(self):
    #     return 1 / math.exp(self.spice * self.lamb)

    # def calculate_legitimacy(self):
    #     legitimacy = {}
    #     for other_tribe in self.model.tribes:
    #         if other_tribe.id != self.tribe.id:
    #             legitimacy[other_tribe] = 1 / math.exp((self.tribe.total_spice - other_tribe.total_spice) * self.lamb)
    #     return legitimacy

    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, Nomad) for agent in this_cell)

    def get_spice(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if isinstance(agent, Spice):
                return agent
        return None

    def move(self):
        """
        move towards spice, if no spice is visible, move randomly
        """
        visible_positions = [
            i for i in self.model.grid.get_neighborhood(
                self.pos, False, False, self.vision
            )
        ]

        visible_positions = [i for i in visible_positions if not self.is_occupied(i)]

        if not visible_positions:
            return

        spice_levels = [self.get_spice(p).spice if self.get_spice(p) else 0 for p in visible_positions]

        if max(spice_levels) > 0:
            max_spice_positions = [pos for pos, spice in zip(visible_positions, spice_levels) if spice == max(spice_levels)]
            chosen_pos = self.random.choice(max_spice_positions)
        else:
            chosen_pos = self.random.choice(visible_positions)

        immediate_neighbors = [
            (self.pos[0] + dx, self.pos[1] + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx, dy) != (0, 0)
        ]

        immediate_neighbors = [
            pos for pos in immediate_neighbors
            if not self.model.grid.out_of_bounds(pos) and not self.is_occupied(pos)
        ]

        if not immediate_neighbors:
            return

        best_move = min(immediate_neighbors, key=lambda pos: (pos[0] - chosen_pos[0])**2 + (pos[1] - chosen_pos[1])**2)

        self.model.grid.move_agent(self, best_move)
        self.check_interactions()

        # neighbors.append(self.pos)
        # max_spice = [self.get_spice(p) for p in neighbors]
        # max_spice = list(filter(lambda x: x is not None, max_spice))
        # new_pos = np.random.choice(max_spice)

        # self.model.grid.move_agent(self, new_pos.pos)

    def sniff(self):
        spice_patch = self.get_spice(self.pos)
        if spice_patch is not None:
            self.spice += 1
            spice_patch.spice -= 1
            if spice_patch.spice <= 0:
                self.model.remove_agent(spice_patch)
        else:
            pass

    def check_interactions(self):
        """
        Check for interactions (fight or trade) with other nomads in the immediate neighborhood.
        """
        immediate_neighbors = [
            (self.pos[0] + dx, self.pos[1] + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx, dy) != (0, 0)
        ]

        immediate_neighbors = [
            pos for pos in immediate_neighbors
            if not self.model.grid.out_of_bounds(pos)
        ]

        for pos in immediate_neighbors:
            cellmates = self.model.grid.get_cell_list_contents([pos])
            other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != self]

            for other in other_nomads:
                if other.tribe != self.tribe:
                    fighting_game(self, other, alpha=0.2)
                elif other.tribe == self.tribe:
                    trade(self, agent1=self, agent2=other, trade_percentage=0.5)

    # def fight(self):
    #     visible_positions = [i for i in self.model.grid.get_neighborhood(self.pos, False, False, self.vision)]
    #     for p in visible_positions:
    #         cellmates = self.model.grid.get_cell_list_contents([p])
    #         other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != self and agent.tribe != self.tribe]
    #     if other_nomads:
    #         opponent = random.choice(other_nomads)
    #         fighting_game(self, opponent, alpha=0.5)

    def step(self):
        swimming_pentaly = 5 ** any(isinstance(x, Water) for x in self.model.grid.get_cell_list_contents(self.pos))
        self.move()
        self.sniff()
        self.spice -= self.metabolism * swimming_pentaly

        if self.spice <= 0:
            self.model.remove_agent(self)
        elif self.spice >= 20:  # Not sure how much they should have to reproduce yet. This is a placeholder.
            self.model.add_agent(self)


def trade(self, agent1: Nomad, agent2: Nomad, trade_percentage: float):
    trade_amount_self = int(agent1.spice * trade_percentage)
    trade_amount_other = int(agent2.spice * trade_percentage)

    agent1.spice = agent1.spice - trade_amount_self + trade_amount_other
    agent2.spice = agent2.spice - trade_amount_other + trade_amount_self
    
    self.model.record_trade(agent1.tribe.id)


def fighting_game(self, agent1: Nomad, agent2: Nomad, alpha: float):
    if agent1.spice >= agent2.spice:
        weak_agent = agent2
        strong_agent = agent1
    else:
        weak_agent = agent1
        strong_agent = agent2

    if weak_agent.spice > alpha * strong_agent.spice:
        strong_agent.spice += alpha * weak_agent.spice - alpha * strong_agent.spice
        weak_agent.spice -= alpha * weak_agent.spice
    elif weak_agent.spice <= alpha * strong_agent.spice:
        strong_agent.spice += 0
        weak_agent.spice -= 0
        self.model.record_cooperation()
    


class Spice(ms.Agent):
    def __init__(self, id: int, pos: tuple, model: ms.Model, max_spice: int):
        super().__init__(id, model)
        self.pos = pos
        self.spice = max_spice
        self.max_spice = max_spice

    def step(self):
        if self.spice == 0:
            self.model.remove_agent(self)
        elif self.spice > 20:
            self.spice += 1 * np.random.binomial(1, 0.01, 1)[0]


class Water(ms.Agent):
    def __init__(self, id: int, pos: tuple, model: ms.Model):
        super().__init__(id, model)
        self.pos = pos
