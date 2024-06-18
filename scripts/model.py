import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe, Water
import random
from typing import Callable
import os
from matplotlib import pyplot as plt

MONITOR = False


class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, experiment_name: str, width: int, height: int,
                 n_tribes: int, n_agents: int, n_heaps: int,
                 vision_radius: int, step_count: int, alpha: float,
                 trade_percentage: float, spice_movement_bias: float, tribe_movement_bias: float, spice_generator: Callable,
                 river_generator: Callable, location_generator: Callable,
                 spice_kwargs: dict, river_kwargs: dict = {}, location_kwargs: dict = {}):
        super().__init__()
        self.experiment_name = experiment_name
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents
        self.n_heaps = n_heaps
        self.total_fights = 0
        self.total_cooperation = 0
        self.tribes = []
        self.vision_radius = vision_radius
        self.step_count = step_count
        self.current_step = 0
        self.alpha = alpha
        self.trade_percentage = trade_percentage
        self.spice_movement_bias = spice_movement_bias
        self.tribe_movement_bias = tribe_movement_bias
        self.spice_kwargs = spice_kwargs
        self.river_kwargs = spice_kwargs
        self.location_kwargs = spice_kwargs

        self.trades_per_tribe = {tribe_id: 0 for tribe_id in range(n_tribes)}
        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)

        self.datacollector = ms.DataCollector({
            "Nomads": lambda m: m.schedule.get_type_count(Nomad),
            "Fights_per_step": lambda m: m.total_fights / m.schedule.time if m.schedule.time > 0 else 0,
            "Cooperation_per_step": lambda m: m.total_cooperation / m.schedule.time if m.schedule.time > 0 else 0,
            **{f"Tribe_{i}_Nomads": (lambda m, i=i: m.count_tribe_nomads(i)) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Spice": (lambda m, i=i: m.total_spice(i)) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Clustering": (lambda m, i=i: m.clustering(i)) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Trades": (lambda m, i=i: m.trades_per_tribe[i] / m.schedule.time if m.schedule.time > 0 else 0) for i in range(self.n_tribes)}
        })

        spice_dist = spice_generator(self)
        river = river_generator(self)
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
            for x, y in location_generator(self):
                spice = 3
                vision = vision_radius
                metabolism = .1
                nom = Nomad(id, self, (x, y), spice, vision, tribe, metabolism, alpha, trade_percentage, spice_movement_bias, tribe_movement_bias)
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
        i = 0
        for a in self.schedule.agents:
            if isinstance(a, Nomad) and a.tribe.id == tribe_id:
                x, y = a.pos
                neighbors = self.grid.get_neighborhood((x, y), moore=False, include_center=False, radius=self.vision_radius)
                for pos in neighbors:
                    cellmates = self.grid.get_cell_list_contents([pos])
                    other_nomads = [agent for agent in cellmates if isinstance(agent, Nomad) and agent != a and agent.tribe == a.tribe]
                    clustering += len(other_nomads) / len(neighbors)
            i += 1
        return clustering / i if i != 0 else 0

    def record_trade(self, tribe_id):
        self.trades_per_tribe[tribe_id] += 1 / 2

    def record_fight(self):
        self.total_fights += 1 / 2

    def record_cooperation(self):
        self.total_cooperation += 1 / 2

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time, self.schedule.get_type_count(Nomad)])
        self.current_step += 1
        if self.current_step >= self.step_count:
            self.running = False
            # self.save_results(self.experiment_name)

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
            new_agent = Nomad(new_agent_id, self, new_pos, spice, vision, tribe, metabolism=parent_agent.metabolism, alpha=parent_agent.alpha, trade_percentage=parent_agent.trade_percentage, spice_movement_bias=parent_agent.spice_movement_bias, tribe_movement_bias=parent_agent.tribe_movement_bias)
            self.grid.place_agent(new_agent, new_pos)
            self.schedule.add(new_agent)

            parent_agent.spice -= parent_agent.spice // 2

    def save_results(self, experiment_name):
        experiment_dir = os.path.join("Experiments", experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        figures_dir = os.path.join(experiment_dir, "Figures")
        records_dir = os.path.join(experiment_dir, "Records")
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(records_dir, exist_ok=True)

        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv(os.path.join(records_dir, "simulation_data.csv"))

        self.save_plot(data, [f"Tribe_{i}_Nomads" for i in range(self.n_tribes)], os.path.join(figures_dir, "nomads_plot.png"))
        self.save_plot(data, [f"Tribe_{i}_Spice" for i in range(self.n_tribes)], os.path.join(figures_dir, "spice_plot.png"))
        self.save_plot(data, [f"Tribe_{i}_Clustering" for i in range(self.n_tribes)], os.path.join(figures_dir, "clustering_plot.png"))
        self.save_plot(data, "Fights_per_step", os.path.join(figures_dir, "fights_plot.png"))
        self.save_plot(data, "Cooperation_per_step", os.path.join(figures_dir, "cooperation_plot.png"))
        self.save_plot(data, [f"Tribe_{i}_Trades" for i in range(self.n_tribes)], os.path.join(figures_dir, "trades_plot.png"))

    def save_plot(self, data, columns, filename):
        plt.figure()
        if isinstance(columns, list):
            for column in columns:
                plt.plot(data[column], label=column)
            plt.legend()
            ylabel = "Values"
            title = "Values over Time"
        else:
            plt.plot(data[columns])
            ylabel = columns
            title = f"{columns} over Time"
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def run_model(self, step_count=200, save=False):
        self.current_step = 0
        self.running = True

        if self.verbose:
            print("Initial number Sugarscape Agent: ", self.schedule.get_type_count(Nomad))

        while self.running and self.current_step < self.step_count:
            self.step()

        if self.verbose:
            print("")
            print("Final number Sugarscape Agent: ", self.schedule.get_type_count(Nomad))

        if save:
            self.save_results(self.experiment_name)

        return self.datacollector.get_model_vars_dataframe()

