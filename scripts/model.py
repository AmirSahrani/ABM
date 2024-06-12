import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe

MONITOR = True

class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, width: int, height: int, n_tribes: int, n_agents: int, sigma: float):
        super().__init__()
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents
        self.dist_sigma = sigma
        self.tribes = []
        self.total_trades = 0

        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = ms.DataCollector({
            "Nomads": lambda m: m.schedule.get_type_count(Nomad),
            "Trades": lambda m: m.total_trades,
            "Tribe_0_Nomads": lambda m: m.count_tribe_nomads(0),
            "Tribe_1_Nomads": lambda m: m.count_tribe_nomads(1),
            "Tribe_0_Spice": lambda m: m.total_spice(0),
            "Tribe_1_Spice": lambda m: m.total_spice(1),
        })

        self.lamb = 0.1

        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        spice_dist = np.exp(-dist / self.dist_sigma)
        spice_dist = (spice_dist / spice_dist.max() * 20).astype(int)
        id = 0
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            spice = Spice(id, (x, y), self, max_spice)
            id += 1
            self.grid.place_agent(spice, (x, y))
            self.schedule.add(spice)

        for t in range(self.n_tribes):
            tribe = Tribe(t, 0)
            self.tribes.append(tribe)
            for a in range(self.n_agents):
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                spice = 3
                vision = 3
                nom = Nomad(id, self, (x, y), spice, vision, tribe, self.lamb)
                id += 1
                self.grid.place_agent(nom, (x, y))
                self.schedule.add(nom)

        self.running = True
        self.datacollector.collect(self)

    def count_tribe_nomads(self, tribe_id):
        return sum(1 for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)

    def total_spice(self, tribe_id):
        return sum(a.spice for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)

    def record_trade(self):
        self.total_trades += 1

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time, self.schedule.get_type_count(Nomad)])

    def remove_agent(self, agent):
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)

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