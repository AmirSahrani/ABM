import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append("../scripts")
from model import DuneModel
from experiment_utils import *
sys.path.remove("../scripts")

def salib_wrapper(target: str, **kwargs):
    def out_fun(inp):
        try:
            m = DuneModel(width=int(inp[0]),
                          height=int(inp[1]),
                          n_tribes=int(inp[2]),
                          n_agents=int(inp[3]),
                          n_heaps=int(inp[4]),
                          vision_radius=int(inp[5]),
                          step_count=int(inp[6]),
                          alpha=float(inp[7]),
                          trade_percentage=float(inp[8]),
                          spice_movement_bias=float(inp[9]),
                          tribe_movement_bias=float(inp[10]),
                          spice_threshold=float(inp[11]),
                          **kwargs)
            out = m.run_model()
            return out[target].to_numpy()[-1]
        except Exception as e:
            print(f"Error processing input: {inp}, Error: {e}")
            raise e

    return out_fun

def parallel_evaluation(fun, samples, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(fun)(sample) for sample in samples)
    return np.array(results)

def save_phase_plot_data(problem, samples, results, filename="phase_plot_data.csv"):
    data = pd.DataFrame(samples, columns=problem['names'])
    data['result'] = results
    data.to_csv(filename, index=False)

def main():
    num_tribes = 1
    problem = {
        'num_vars': 12,
        'names': [
            "width",
            "height",
            "n_tribes",
            "n_agents",
            "n_heaps",
            "vision_radius",
            "step_count",
            "alpha",
            "trade_percentage",
            "spice_movement_bias",
            "tribe_movement_bias",
            "spice_threshold"
        ],
        'bounds': [
            (10, 200),  # width
            (10, 200),  # height
            (2, 10),    # n_tribes
            (10, 200),  # n_agents
            (1, 50),    # n_heaps
            (2, 20),    # vision_radius
            # (50, 500),  # step_count run every 50
            (0.0, 1.0), # alpha
            (0.0, 1.0), # trade_percentage
            (0.0, 1.0), # spice_movement_bias
            (0.0, 1.0), # tribe_movement_bias
            (0.0, 1.0)  # spice_threshold
        ]
    }

    kwargs = {
        "experiment_name": "test",
        "spice_generator": gen_spice_map,
        "river_generator": gen_river_random,
        "location_generator": random_locations,
        "spice_kwargs": {"cov_range": (3, 9), "total_spice": 1000}
    }

    output_params = [
        # "Nomads", 
        "Fights_per_step", 
        "Cooperation_per_step"
    ] + [f"Tribe_{i}_Nomads" for i in range(num_tribes)] + [f"Tribe_{i}_Spice" for i in range(num_tribes)] + [f"Tribe_{i}_Clustering" for i in range(num_tribes)] + [f"Tribe_{i}_Trades" for i in range(num_tribes)
        ]

    results_dict = {}

    samples = sobol_sample.sample(problem, nr_sobol_samples)

    for param in output_params:
        print(f"Running sensitivity analysis for {param}")
        sensitivity_target = salib_wrapper(param, **kwargs)
        results = parallel_evaluation(sensitivity_target, samples)
        Si = sobol_analyze.analyze(problem, results.flatten())
        results_dict[param] = Si

        save_phase_plot_data(problem, samples, results, filename=f"phase_plot_data_{param}_{nr_sobol_samples}.csv")

        sobol_indices_data = {
            'Parameter': problem['names'],
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        }

        num_vars = problem['num_vars']
        S2 = Si['S2']
        S2_conf = Si['S2_conf']
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                sobol_indices_data[f'S2_{problem["names"][i]}_{problem["names"][j]}'] = S2[i, j]
                sobol_indices_data[f'S2_conf_{problem["names"][i]}_{problem["names"][j]}'] = S2_conf[i, j]

        sobol_indices_df = pd.DataFrame(sobol_indices_data)
        sobol_indices_df.to_csv(f'sobol_results_{param}_{nr_sobol_samples}.csv', index=False)


if __name__ == "__main__":
    nr_sobol_samples = 512
    main()
