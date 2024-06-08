import mesa as ms
import numpy as np
from dataclasses import dataclass


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

    def __init__(self, id: int, model: ms.Model, pos: tuple, spice: int, vision: int, tribe: Tribe):
        super().__init__(id, model)
        self.pos = pos
        self.spice = spice
        self.vision = vision
        self.tribe = tribe

    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, Nomad) for agent in this_cell)

    def get_spice(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if isinstance(agent, Spice):
                return agent

    def move(self):
        """
        !! vision is currently the step size, we probably do not want that
        """
        neighbors = [
            i
            for i in self.model.grid.get_neighborhood(
                self.pos, False, False, self.vision
            )
            if not self.is_occupied(i)
        ]

        # TODO this is hacky and in accurate, now we just randomly move
        neighbors.append(self.pos)
        max_spice = [self.get_spice(p) for p in neighbors]
        max_spice = list(filter(lambda x: x is not None, max_spice))
        new_pos = np.random.choice(max_spice)
        self.model.grid.move_agent(self, new_pos.pos)

    def sniff(self):
        spice_patch = self.get_spice(self.pos)
        self.spice += 1
        spice_patch.spice -= 1
        if spice_patch.spice < 0:
            self.model.remove_agent(spice_patch)

    def step(self):
        self.move()
        self.sniff()

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
        elif self.spice > 20:
            self.spice += 1
