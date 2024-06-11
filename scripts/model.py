import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe

MONITOR = True


class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, width: int, height: int, n_tribes: int, n_agents: int):
        super().__init__()
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents

        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = ms.DataCollector({
            "Nomad": lambda m: m.schedule.get_type_count(Nomad)
        })

        spice_dist = np.random.binomial(5, 0.2, (self.width, self.height))
        id = 0
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            spice = Spice(id, (x, y), self, max_spice)
            id += 1
            self.grid.place_agent(spice, (x, y))
            self.schedule.add(spice)

        for t in range(self.n_tribes):
            tribe = Tribe(t, 0)
            for a in range(self.n_agents):
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                spice = 3
                vision = 3
                nom = Nomad(id, self, (x, y), spice, vision, tribe)
                id += 1
                self.grid.place_agent(nom, (x, y))
                self.schedule.add(nom)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time, self.schedule.get_type_count(Nomad)])

    def remove_agent(self, agent):
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)
    
    def add_agent(self, parent_agent):
        x, y = parent_agent.pos
        spice = parent_agent.spice//2
        vision = parent_agent.vision
        tribe = parent_agent.tribe

        new_agent_id = max(agent.unique_id for agent in self.schedule.agents) + 1
        new_agent = Nomad(new_agent_id, self, (x, y), spice, vision, tribe)
        self.grid.place_agent(new_agent, (x, y))
        self.schedule.add(new_agent)

        parent_agent.spice -= parent_agent.spice//2

    def run_model(self, step_count=200):
        if self.verbose:
            print(
                "Initial number Sugarscape Agent: ",
                self.schedule.get_type_count(Nomad),
            )

        for i in range(step_count):
            self.step()

        if self.verbose:
            print("")
            print(
                "Final number Sugarscape Agent: ",
                self.schedule.get_type_count(Nomad),
            )
