import mesa as ms
import numpy as np
from agents import Nomad, Spice, Tribe, Water
import random
from typing import Callable
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import warnings

MONITOR = False


class DuneModel(ms.Model):
    verbose = MONITOR

    def __init__(self, experiment_name: str, width: int, height: int,
                 n_tribes: int, n_agents: int, n_heaps: int,
                 vision_radius: int, step_count: int, alpha: float,
                 trade_percentage: float, spice_movement_bias: float, tribe_movement_bias: float, spice_threshold: int, spice_generator: Callable,
                 river_generator: Callable, location_generator: Callable,
                 spice_kwargs: dict, river_kwargs: dict = {}, location_kwargs: dict = {}, frequency=10):
        super().__init__()
        self.experiment_name = experiment_name
        self.width = width
        self.height = height
        self.n_tribes = n_tribes
        self.n_agents = n_agents // n_tribes
        self.n_heaps = n_heaps
        self.total_fights = 0
        self.total_cooperation = 0
        self.tribes = []
        self.spice_threshold = spice_threshold
        self.step_count = step_count
        self.current_step = 0
        self.vision_radius = vision_radius
        self.alpha = alpha
        self.trade_percentage = trade_percentage
        self.spice_movement_bias = spice_movement_bias
        self.tribe_movement_bias = tribe_movement_bias
        self.frequency = frequency

        self.spice_generator = spice_generator
        self.river_generator = river_generator
        self.location_generator = location_generator

        self.spice_kwargs = spice_kwargs
        self.spice_kwargs["total_spice"] = self.spice_kwargs["total_spice"] // self.n_heaps
        self.river_kwargs = river_kwargs
        self.location_kwargs = location_kwargs

        self.trades_per_tribe = {tribe_id: 0 for tribe_id in range(n_tribes)}
        self.schedule = ms.time.RandomActivationByType(self)
        self.grid = ms.space.MultiGrid(self.width, self.height, torus=False)
        self.id = 0

        self.datacollector = ms.DataCollector({
            "Nomads": lambda m: m.schedule.get_type_count(Nomad),
            "total_Clustering": lambda m: m.total_clustering(self.n_tribes),
            "Fights_per_step": lambda m: m.total_fights / (m.total_fights + m.total_cooperation) if (m.total_fights + m.total_cooperation) > 0 else 0,
            "Cooperation_per_step": lambda m: m.total_cooperation / (m.total_fights + m.total_cooperation) if (m.total_fights + m.total_cooperation) > 0 else 0,
            **{f"Tribe_{i}_Nomads": (lambda m, i=i: m.count_tribe_nomads(i)) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Spice": (lambda m, i=i: m.total_spice(i)) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Clustering": (lambda m, i=i: m.clustering_for_tribe(i)[0]) for i in range(self.n_tribes)},
            **{f"Tribe_{i}_Trades": (lambda m, i=i: m.trades_per_tribe[i] / m.schedule.time if m.schedule.time > 0 else 0) for i in range(self.n_tribes)}
        })

        warnings.filterwarnings("ignore")
        spice_dist = self.spice_generator(self)
        river = self.river_generator(self)
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            if river[x, y]:
                self.id += 1
                water = Water(self.id, (x, y), self)
                self.grid.place_agent(water, (x, y))
                self.schedule.add(water)
            elif max_spice > 0:
                spice = Spice(self.id, (x, y), self, max_spice)
                self.id += 1
                self.grid.place_agent(spice, (x, y))
                self.schedule.add(spice)

        for t in range(self.n_tribes):
            tribe = Tribe(t, 0)
            self.tribes.append(tribe)
            for x, y in self.location_generator(self):
                spice = 3
                vision = vision_radius
                metabolism = .1
                self.id += 1
                nom = Nomad(self.id, self, (x, y), spice, vision, tribe, metabolism, alpha, trade_percentage, spice_movement_bias, tribe_movement_bias)
                self.grid.place_agent(nom, (x, y))
                self.schedule.add(nom)

        self.running = True
        self.datacollector.collect(self)

    def count_tribe_nomads(self, tribe_id):
        return sum(1 for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)

    def total_spice(self, tribe_id):
        return sum(a.spice for a in self.schedule.agents if isinstance(a, Nomad) and a.tribe.id == tribe_id)

    # def determine_optimal_k(self, points, max_k):
    #     silhouette_scores = []
    #     k_range = range(2, min(max_k, len(points)))

    #     if len(points) < 2:
    #         return len(points)

    #     for k in k_range:
    #         if len(points) <= k:
    #             silhouette_scores.append(-1)
    #         else:
    #             kmeans = KMeans(n_clusters=k, random_state=0)
    #             kmeans.fit(points)
    #             labels = kmeans.labels_
    #             if len(set(labels)) > 1:
    #                 silhouette_scores.append(silhouette_score(points, labels))
    #             else:
    #                 silhouette_scores.append(-1)

    #     if all(score == -1 for score in silhouette_scores):
    #         return len(points)

    #     valid_scores = [(score, k) for score, k in zip(silhouette_scores, k_range) if score != -1]
    #     optimal_k = max(valid_scores, key=lambda x: x[0])[1]

    #     return optimal_k

    # def clustering_K_means(self, tribe_id, k_max=20):
    #     positions = []
    #     for a in self.schedule.agents:
    #         if isinstance(a, Nomad) and a.tribe.id == tribe_id:
    #             positions.append(a.pos)

    #     positions = np.array(positions)
    #     total_individuals = len(positions)
    #     if total_individuals == 0:
    #         return 0, []
    #     k = self.determine_optimal_k(positions, k_max)

    #     if k > 0:
    #         kmeans = KMeans(n_clusters=k, random_state=0).fit(positions)
    #         labels = kmeans.labels_
    #         unique_labels, counts = np.unique(labels, return_counts=True)
    #         average_cluster_size = np.mean(counts) / total_individuals
    #     else:
    #         average_cluster_size = 0
    #         counts = []

    #     return average_cluster_size, counts

    # def total_clutering(self, n_tribes):
    #     total_clustering = 0
    #     for i in range(n_tribes):
    #         clustering, _ = self.clustering_K_means(i)
    #         if self.schedule.get_type_count(Nomad) > 0:
    #             total_clustering += self.count_tribe_nomads(i) * clustering / self.schedule.get_type_count(Nomad)
    #         else:
    #             total_clustering += 0
    #     return total_clustering

    def clustering_DBSCAN(self, points, eps, min_samples=3):
        if len(points) < 2:
            return 0, []

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print(n_clusters)

        if n_clusters > 0:
            counts = np.bincount(labels[labels >= 0])
            average_cluster_size = np.mean(counts) 
        else:
            average_cluster_size = 0
            counts = []
        return average_cluster_size, counts



    def clustering_for_tribe(self, tribe_id, min_samples=3):
        positions = []
        for a in self.schedule.agents:
            if isinstance(a, Nomad) and a.tribe.id == tribe_id:
                positions.append(a.pos)

        positions = np.array(positions)
        total_individuals = len(positions)
        if total_individuals == 0:
            return 0, []

        average_cluster_size, counts = self.clustering_DBSCAN(positions, eps=self.vision_radius, min_samples=min_samples)

        return average_cluster_size, counts
    
    def total_clustering(self, n_tribes, min_samples=3):
        total_clustering = 0
        for i in range(n_tribes):
            clustering, _ = self.clustering_for_tribe(i, min_samples=min_samples)
            nomad_count = self.schedule.get_type_count(Nomad)
            if nomad_count > 0:
                total_clustering += self.count_tribe_nomads(i) * clustering 
            else:
                total_clustering += 0
        return total_clustering

    def record_trade(self, tribe_id):
        self.trades_per_tribe[tribe_id] += 1 / 2

    def record_fight(self):
        self.total_fights += 1 / 2

    def record_cooperation(self):
        self.total_cooperation += 1 / 2

    def total_spice_in_system(self):
        total_spice = 0
        for agent in self.schedule.agents:
            if isinstance(agent, Spice):
                total_spice += agent.spice
        return total_spice

    def regenerate_spice(self):
        self.n_heaps = 1
        spice_dist = self.spice_generator(self)
        for _, (x, y) in self.grid.coord_iter():
            max_spice = spice_dist[x, y]
            if max_spice > 0:
                for agent in self.grid.get_cell_list_contents([x, y]):
                    if isinstance(agent, Water):
                        continue
                    elif isinstance(agent, Spice) and agent.spice < 20:
                        agent.spice += max_spice
                        agent.spice %= 21
                        break
                    else:
                        self.id += 1
                        new_spice = Spice(self.id, (x, y), self, max_spice)
                        self.grid.place_agent(new_spice, (x, y))
                        self.schedule.add(new_spice)
                        break

    def step(self):
        self.schedule.step()
        if self.schedule.time % self.frequency == 0:
            self.datacollector.collect(self)
            total_spice = self.total_spice_in_system()
            if total_spice < self.spice_threshold:
                self.regenerate_spice()
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
        neighbors = self.grid.get_neighborhood((x, y), moore=True, include_center=False, radius=1)
        empty_cells = [cell for cell in neighbors if self.grid.is_cell_empty(cell)]
        if empty_cells:
            new_pos = random.choice(empty_cells)
            spice = parent_agent.spice // 2
            vision = parent_agent.vision
            tribe = parent_agent.tribe

            self.id += 1
            new_agent = Nomad(self.id, self, new_pos, spice, vision, tribe, metabolism=parent_agent.metabolism, alpha=parent_agent.alpha, trade_percentage=parent_agent.trade_percentage, spice_movement_bias=parent_agent.spice_movement_bias, tribe_movement_bias=parent_agent.tribe_movement_bias)
            self.grid.place_agent(new_agent, new_pos)
            self.schedule.add(new_agent)
            parent_agent.spice = spice

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
        self.save_plot(data, "total_Clustering", os.path.join(figures_dir, "total_clustering_plot.png"))
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

        # print("Data collection successful.")

        return self.datacollector.get_model_vars_dataframe()

        # return self.datacollector.get_model_vars_dataframe()

        # data = self.datacollector.get_model_vars_dataframe()
        # if data is None or data.empty:
        #     print("Data collection returned None or empty DataFrame.")
        # else:
        #     print("Data collection successful.")

        # return data
