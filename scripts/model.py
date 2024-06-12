import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe

MONITOR = True


def gen_spice_map(width: int, height: int, n_heaps: int, total_spice: int):
    # Initialize an empty map
    spice_map = np.zeros((width, height))
    heap_pos_x = np.random.randint(0, width, n_heaps)
    heap_pos_y = np.random.randint(0, height, n_heaps)

    for (heap_x, heap_y) in zip(heap_pos_x, heap_pos_y):
        cov = np.random.uniform(3, 9, (2, 2))
        cov = cov @ cov.T
        heap = np.random.multivariate_normal([heap_x, heap_y], cov, size=total_spice).astype(int)

        for (x, y) in zip(heap[:, 0], heap[:, 1]):
            if 0 < x < width and 0 < y < height:
                spice_map[x, y] += 1

    return (spice_map / np.max(spice_map) * 20).astype(int)


class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, width: int, height: int, n_tribes: int, n_agents: int, n_heaps: int, vision_radius: int):
        super().__init__()
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents
        self.n_heaps = n_heaps
        self.vision_radius = vision_radius
        self.tribes = []

        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = ms.DataCollector({
            "Nomad": lambda m: m.schedule.get_type_count(Nomad)
        })

        spice_dist = gen_spice_map(self.width, self.height, self.n_heaps, 1000)
        id = 0
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            if max_spice > 0:
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
                vision = vision_radius
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

    def run_model(self, step_count=10000):
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
