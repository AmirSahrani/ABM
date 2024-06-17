import numpy as np
from model import DuneModel
from experiment_utils import *
import pandas as pd

parameter_ranges = {
    "n_tribes": [2, 3, 4, 5, 6],
    "n_agents": [100, 250, 500, 750, 1000],
    "n_heaps": [2, 4, 6, 8, 10],
    "vision_radius": [2, 3, 4, 5, 6],
    "alpha": [0.1, 0.3, 0.5, 0.7],
    "trade_percentage": [0.1, 0.3, 0.5, 0.7],
    "spice_threshold": [0, 100, 300, 500, 700]
}

def run_simulation(params):
    model = DuneModel(
        experiment_name=params["experiment_name"],
        width=params["width"],
        height=params["height"],
        n_tribes=params["n_tribes"],
        n_agents=params["n_agents"],
        n_heaps=params["n_heaps"],
        vision_radius=params["vision_radius"],
        step_count=params["step_count"],
        alpha=params["alpha"],
        trade_percentage=params["trade_percentage"],
        spice_threshold=params["spice_threshold"],
        spice_generator=gen_spice_map,
        river_generator=gen_river_random,
        location_generator=random_locations,
        spice_kwargs={"cov_range": [3, 9], "total_spice": 1000}
    )
    for _ in range(params["step_count"]):
        model.step()
    results = model.datacollector.get_model_vars_dataframe()
    return results

def ofat_analysis(base_params, parameter_ranges, n_trials):
    results = []
    for param, values in parameter_ranges.items():
        for value in values:
            for trial in range(n_trials):
                params = base_params.copy()
                params[param] = value
                result = run_simulation(params)
                result["param"] = param
                result["value"] = value
                result["trial"] = trial + 1
                results.append(result)
    return pd.concat(results, ignore_index=True)

base_params = {
    "experiment_name": "ofat_results",
    "width": 100,
    "height": 100,
    "n_tribes": 3,
    "n_agents": 200,
    "n_heaps": 8,
    "vision_radius": 5,
    "step_count": 100,
    "alpha": 0.5,
    "trade_percentage": 0.5,
    "spice_threshold": 500
}

n_trials = 10
results = ofat_analysis(base_params, parameter_ranges, n_trials)
results.to_csv("../data/ofat_results.csv", index=False)
