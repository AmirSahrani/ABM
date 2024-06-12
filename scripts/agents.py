import mesa as ms
import nashpy as nash
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

        # TODO this is hacky and inaccurate, now we just randomly move
        neighbors.append(self.pos)
        max_spice = [self.get_spice(p) for p in neighbors]
        max_spice = list(filter(lambda x: x is not None, max_spice))
        if max_spice:
            new_pos = np.random.choice(max_spice)
            new_pos = new_pos.pos
        else:
            move = np.random.binomial(1, 0.5, 2)
            new_pos = ((self.pos[0] + move[0]) % 100, (self.pos[1] + move[1]) % 100)

        self.model.grid.move_agent(self, new_pos)

    def sniff(self):
        spice_patch = self.get_spice(self.pos)
        if spice_patch:
            self.spice += 1
            spice_patch.spice -= 1
            if spice_patch.spice < 0:
                self.model.remove_agent(spice_patch)

    def fight(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != self]
        for other in other_nomads:
            fighting_game(self, other, alpha=0.5)

    def step(self):
        self.move()
        self.sniff()
        self.fight()

        if self.spice < 0:
            self.model.remove_agent(self)
        # TODO split agent


def fighting_game(agent1: Nomad, agent2: Nomad, alpha):
    if agent1.spice >= agent2.spice:
        weak_agent = agent2
        strong_agent = agent1
    else:
        weak_agent = agent1
        strong_agent = agent2

    if agent1.tribe != agent2.tribe:
        strong_agent_payoffs = np.array([[0, weak_agent.spice - alpha * strong_agent.spice], [weak_agent.spice, weak_agent.spice - (alpha / 2) * strong_agent.spice]])
        weak_agent_payoffs = np.array([[0, -weak_agent.spice], [-weak_agent.spice, -weak_agent.spice]])

        fight = nash.Game(strong_agent_payoffs, weak_agent_payoffs)

        equilibria = list(fight.support_enumeration())
        print(equilibria)

        strong_agent_strategy = np.argmax(equilibria[0])
        weak_agent_strategy = np.argmax(equilibria[1])

        strong_agent.spice += strong_agent_payoffs[strong_agent_strategy, weak_agent_strategy]
        weak_agent.spice += weak_agent_payoffs[weak_agent_strategy, strong_agent_strategy]

    else:
        payoff = (strong_agent.spice - weak_agent.spice) // 2
        weak_agent.spice += payoff
        strong_agent.spice -= payoff

    # print(f"After the game, the stronger agent has {strong_agent.spice} spice.")
    # print(f"After the game, the weaker agent has {weak_agent.spice} spice.")


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
