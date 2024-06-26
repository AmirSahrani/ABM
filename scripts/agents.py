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


    def __init__(self, id: int, model: ms.Model, pos: tuple, spice: int, vision: int, tribe: Tribe, metabolism: float, alpha: float, trade_percentage: float, spice_movement_bias: float, tribe_movement_bias: float):
        super().__init__(id, model)
        self.pos = pos
        self.spice = spice
        self.tribe = tribe

        self.vision = np.random.randint(0, 10)
        self.metabolism = np.random.uniform(0, 0.4)
        self.spice_movement_bias = np.random.uniform(0, 1)
        self.tribe_movement_bias = np.random.uniform(0, 1)
        self.alpha = np.random.uniform(0, 1)
        self.trade_percentage = np.random.uniform(0, 1)
        self.reproduction_threshold = np.random.randint(10, 100)
        self. visible_positions = []


    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, Nomad) for agent in this_cell)


    def get_spice(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if isinstance(agent, Spice):
                return agent
        return None


    def is_tribe_member(self, pos):
        """
        Check if the agent at the given position is a member of the same tribe.
        """
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if isinstance(agent, Nomad):
                # print(f"Nomad {agent.unique_id} found at {pos} with tribe {agent.tribe.id}")
                if agent.tribe == self.tribe:
                    # print(f"Nomad {agent.unique_id} is a member of the same tribe {self.tribe.id}")
                    return True
        return False


    def move(self):
        """
        Move towards spice if visible and the spice movement bias is met.
        If no spice is visible, move towards the center of mass of the spice level of visible tribal members if the tribe movement bias is met.
        Otherwise, move randomly.
        """
        self.visible_positions = [
            i for i in self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False, radius=self.vision
            )
        ]

        if not self.visible_positions:
            return

        # Check for spice in visible positions
        spice_levels = [self.get_spice(p).spice if self.get_spice(p) else 0 for p in self.visible_positions]
        moved_towards = ""

        if max(spice_levels) > 0 and self.random.random() < self.spice_movement_bias:
            # Move towards position with max spicyness
            max_spice_positions = [pos for pos, spice in zip(self.visible_positions, spice_levels) if spice == max(spice_levels)]
            chosen_pos = self.random.choice(max_spice_positions)
            moved_towards = "spice"
        else:
            # Get visible tribal members and their spice levels
            tribe_members = [(pos, agent.spice) for pos in self.visible_positions
                             for agent in self.model.grid.get_cell_list_contents([pos])
                             if isinstance(agent, Nomad) and agent.tribe == self.tribe]

            if tribe_members and self.random.random() < self.tribe_movement_bias:
                # Calculate center of mass of spice levels
                total_spice = sum(spice for _, spice in tribe_members)
                if total_spice > 0:
                    center_of_mass = (
                        sum(pos[0] * spice for pos, spice in tribe_members) / total_spice,
                        sum(pos[1] * spice for pos, spice in tribe_members) / total_spice
                    )
                    # Closest pos to center of mass
                    chosen_pos = min(self.visible_positions, key=lambda pos: (pos[0] - center_of_mass[0])**2 + (pos[1] - center_of_mass[1])**2)
                    moved_towards = "tribe member center of mass"
                else:
                    chosen_pos = self.non_random_walking()
                    moved_towards = "random"
            else:
                # No visible tribe members, move randomly
                chosen_pos = self.non_random_walking()
                moved_towards = "direction"

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
        # print(f"Nomad {self.unique_id} moved towards {moved_towards} to {best_move}")
        # print(f"Nomad {self.unique_id} moved towards {moved_towards} to {best_move}")
        self.model.grid.move_agent(self, best_move)
        self.check_interactions()
        
        
    def non_random_walking(self):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        if not hasattr(self, 'current_direction') or self.random.random() < 0.1:
            self.current_direction = self.random.choice(directions)

        new_pos = (self.pos[0] + self.current_direction[0], self.pos[1] + self.current_direction[1])

        if self.model.grid.out_of_bounds(new_pos) or self.is_occupied(new_pos):
            self.current_direction = self.random.choice(directions)
            new_pos = (self.pos[0] + self.current_direction[0], self.pos[1] + self.current_direction[1])

        return new_pos


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
                    fighting_game(self, other, alpha=self.alpha, model=self.model)
                elif other.tribe == self.tribe:
                    trade(agent1=self, agent2=other, trade_percentage=self.trade_percentage, model=self.model)


    def step(self):
        swimming_pentaly = 5 ** any(isinstance(x, Water) for x in self.model.grid.get_cell_list_contents(self.pos))
        self.move()
        self.sniff()
        self.spice -= self.metabolism * swimming_pentaly

        if self.spice <= 0:
            self.model.remove_agent(self)
        elif self.spice >= self.reproduction_threshold:
            self.model.add_agent(self)


def trade(agent1: Nomad, agent2: Nomad, trade_percentage: float, model: ms.Model):
    trade_amount_self = int(agent1.spice * trade_percentage)
    trade_amount_other = int(agent2.spice * trade_percentage)

    agent1.spice = agent1.spice - trade_amount_self + trade_amount_other
    agent2.spice = agent2.spice - trade_amount_other + trade_amount_self

    model.record_trade(agent1.tribe.id)


def fighting_game(agent1: Nomad, agent2: Nomad, alpha: float, model: ms.Model):
    if agent1.spice >= agent2.spice:
        weak_agent = agent2
        strong_agent = agent1
    else:
        weak_agent = agent1
        strong_agent = agent2

    visible_positions = [i for i in model.grid.get_neighborhood(strong_agent.pos, moore=True, include_center=False, radius=strong_agent.vision)]

    other_nomads = []
    same_tribe = []

    for p in visible_positions:
        cellmates = model.grid.get_cell_list_contents([p])
        other_nomads += [agent for agent in cellmates if isinstance(agent, Nomad) and agent != strong_agent and agent.tribe != strong_agent.tribe]
        same_tribe += [agent for agent in cellmates if isinstance(agent, Nomad) and agent != strong_agent and agent.tribe == strong_agent.tribe]

    cost = (len(other_nomads)) / (len(other_nomads) + len(same_tribe) + 1)

    if (1 - cost) * weak_agent.spice > cost * strong_agent.spice:
        strong_agent.spice += (1 - cost) * weak_agent.spice - cost * strong_agent.spice
        weak_agent.spice -= (1 - cost) * weak_agent.spice
        model.record_fight()
    elif (1 - cost) * weak_agent.spice <= cost * strong_agent.spice:
        strong_agent.spice += 0
        weak_agent.spice -= 0
        model.record_cooperation()



class Spice(ms.Agent):
    def __init__(self, id: int, pos: tuple, model: ms.Model, max_spice: int, grow_threshold: int):
        super().__init__(id, model)
        self.pos = pos
        self.spice = max_spice
        self.max_spice = max_spice
        self.grow_threshold = grow_threshold

    def step(self):
        if self.spice == 0:
            self.model.remove_agent(self)
        elif self.spice > self.grow_threshold:
            self.spice += 0 * np.random.binomial(1, .99, 1)[0]


class Water(ms.Agent):
    def __init__(self, id: int, pos: tuple, model: ms.Model):
        super().__init__(id, model)
        self.pos = pos
