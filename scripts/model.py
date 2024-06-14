import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe, Water
import random

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


def gen_river(width, height):
    river = np.zeros((width, height))
    river[width // 2, :] = 1
    river[:, height // 3: -height // 3] = 0
    return river


class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, width: int, height: int, n_tribes: int, n_agents: int, n_heaps: int, vision_radius: int):
        super().__init__()
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents
        self.n_heaps = n_heaps
        self.total_fights = 0
        self.total_cooperation = 0
        self.tribes = []
        self.vision_radius = vision_radius

        self.trades_per_tribe = {tribe_id: 0 for tribe_id in range(n_tribes)}
        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = ms.DataCollector({
            "Nomads": lambda m: m.schedule.get_type_count(Nomad),
            "Fights_per_step": lambda m: m.total_fights / m.schedule.time if m.schedule.time > 0 else 0,
            "Cooperation_per_step": lambda m: m.total_cooperation / m.schedule.time if m.schedule.time > 0 else 0,
            "Tribe_0_Nomads": lambda m: m.count_tribe_nomads(0),
            "Tribe_1_Nomads": lambda m: m.count_tribe_nomads(1),
            "Tribe_0_Spice": lambda m: m.total_spice(0),
            "Tribe_1_Spice": lambda m: m.total_spice(1),
            "Tribe_0_Clustering": lambda m: m.clustering(0),
            "Tribe_1_Clustering": lambda m: m.clustering(1),
            "Tribe_0_Trades": lambda m: m.trades_per_tribe[0]/ m.schedule.time if m.schedule.time > 0 else 0,
            "Tribe_1_Trades": lambda m: m.trades_per_tribe[1]/ m.schedule.time if m.schedule.time > 0 else 0,
        })

        spice_dist = gen_spice_map(self.width, self.height, self.n_heaps, 1000)
        river = gen_river(self.width, self.height)
        id = 0
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            if river[x, y]:
                pass
                # water = Water(id, (x, y), self)
                # id += 1
                # self.grid.place_agent(water, (x, y))
            elif max_spice > 0:
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
                metabolism = .1
                nom = Nomad(id, self, (x, y), spice, vision, tribe, metabolism)
                id += 1
                self.grid.place_agent(nom, (x, y))
                self.schedule.add(nom)

        self.running = True
        self.datacollector.collect(self)

    def count_tribe_nomads(self, tribe_id):
        return sum(1 for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)

    def total_spice(self, tribe_id):
        return sum(a.spice for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)
    
    def clustering(self, tribe_id):
        clustering = 0
        i=0
        for a in self.schedule.agents:
            if isinstance(a, Nomad) and a.tribe.id == tribe_id:
                x, y = a.pos
                neighbors = self.grid.get_neighborhood((x, y), moore=False, include_center=False)
                for pos in neighbors:
                    cellmates = self.grid.get_cell_list_contents([pos])
                    other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != a and agent.tribe == a.tribe]
                    clustering += len(other_nomads)/len(neighbors)
            i +=1
        return clustering/i
        

    def record_trade(self, tribe_id):
        self.trades_per_tribe[tribe_id] += 1/2

    
    def record_fight(self):
        self.total_fights += 1/2
        
    def record_cooperation(self):
        self.total_cooperation += 1/2

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time, self.schedule.get_type_count(Nomad)])

    def remove_agent(self, agent):
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)

    def add_agent(self, parent_agent):
        x, y = parent_agent.pos
        neighbors = self.grid.get_neighborhood((x, y), moore=False, include_center=False)
        empty_cells = [cell for cell in neighbors if self.grid.is_cell_empty(cell)]
        if empty_cells:
            new_pos = random.choice(empty_cells)
            spice = parent_agent.spice // 2
            vision = parent_agent.vision
            tribe = parent_agent.tribe

            new_agent_id = max(agent.unique_id for agent in self.schedule.agents) + 1
            new_agent = Nomad(new_agent_id, self, new_pos, spice, vision, tribe, metabolism=parent_agent.metabolism)
            self.grid.place_agent(new_agent, new_pos)
            self.schedule.add(new_agent)

            parent_agent.spice -= parent_agent.spice // 2

    def run_model(self, step_count=10000):
        if self.verbose:
            print(
                "Initial number Agent: ",
                self.schedule.get_type_count(Nomad),
            )

        for i in range(step_count):
            self.step()

        if self.verbose:
            print("")
            print(
                "Final number Agent: ",
                self.schedule.get_type_count(Nomad),
            )