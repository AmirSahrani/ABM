from experiment_utils import *
from model import DuneModel
import os
import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import sys
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

sys.path.append("../scripts")
sys.path.remove("../scripts")


def sensitivity_target(inp):
    kwargs = {
        "experiment_name": "test",
        "spice_generator": gen_spice_map,
        "river_generator": gen_river_random,
        "location_generator": random_locations,
        "spice_kwargs": {"cov_range": (3, 9), "total_spice": 1000},

    }
    try:
        m = DuneModel(
            width=width,
            height=height,
            n_tribes=int(inp[0]),
            n_agents=int(inp[1]),
            n_heaps=int(inp[2]),
            vision_radius=int(inp[3]), step_count=200, alpha=float(inp[4]),
            trade_percentage=float(inp[5]),
            spice_movement_bias=float(inp[6]),
            tribe_movement_bias=float(inp[7]),
            spice_threshold=float(inp[8]),
            frequency=50,
            **kwargs
        )
        out = m.run_model()
        return out
    except Exception as e:
        print(f"Error processing input: {inp}, Error: {e}")
        raise e


def parallel_evaluation(fun, samples, n_jobs=1):
    output_params = [
        "total_Clustering",
        "Fights_per_step",
        "Cooperation_per_step"
    ]
    out = np.empty((len(samples), 3, 4))
    with multiprocessing.Pool(n_jobs) as pool:
        results = list(tqdm(pool.map(fun, samples), total=len(samples)))

    for i, result in enumerate(results):
        for j, param in enumerate(output_params):
            print(result[param][-4:])
            out[i, j, :] = result[param][-4:]

    return out


def save_phase_plot_data(problem, samples, results, filename="phase_plot_data.csv"):
    data = pd.DataFrame(samples, columns=problem['names'])
    data['result'] = results
    data.to_csv(filename, index=False)


def main():
    num_tribes = 1
    problem = {
        'num_vars': 9,
        'names': [
            "n_tribes",
            "n_agents",
            "n_heaps",
            "vision_radius",
            "alpha",
            "trade_percentage",
            "spice_movement_bias",
            "tribe_movement_bias",
            "spice_threshold"
        ],
        'bounds': [
            (2, 4),     # n_tribes
            (100, 2000),# n_agents
            (1, 10),    # n_heaps
            (2, 50),    # vision_radius
            (0.0, 1.0),  # alpha
            (0.0, 1.0),  # trade_percentage
            (0.0, 1.0),  # spice_movement_bias
            (0.0, 1.0),  # tribe_movement_bias
            (0.0, 1.0)  # spice_threshold
        ]
    }

    [f"Tribe_{i}_Clustering" for i in range(num_tribes)]

    results_dict = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples = sobol_sample.sample(problem, nr_sobol_samples)
    samples_csv = pd.DataFrame(samples, columns=problem['names'])
    samples_csv.to_csv(os.path.join(output_dir, f'sobol_samples_{nr_sobol_samples}.csv'), index=False)

    all_results = parallel_evaluation(sensitivity_target, samples, n_jobs=32)

    output_params = [
        "total_Clustering",
        "Fights_per_step",
        "Cooperation_per_step"
    ]
    for step_count in range(all_results.shape[2]):
        print(f"Running sensitivity analysis for step_count: {50*step_count}")

        for param in range(all_results.shape[1]):
            results = all_results[:, param, step_count]
            Si = sobol_analyze.analyze(problem, results.flatten())
            results_dict[param] = Si

            save_phase_plot_data(problem, samples, results, filename=os.path.join(output_dir, f"phase_plot_data_{output_dir[param]}_sample_{nr_sobol_samples}_step_{(step_count + 1 )*50}.csv"))

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
            sobol_indices_df.to_csv(os.path.join(output_dir, f'sobol_results_{output_params[param]}_sample_{nr_sobol_samples}_step_{(step_count+1)*50}.csv'), index=False)


if __name__ == "__main__":

    height = 500
    width = 500
    nr_sobol_samples = 1024

    output_dir = "GSA"

    main()
