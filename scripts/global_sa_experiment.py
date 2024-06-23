import os
import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
import logging

sys.path.append("../scripts")
from model import DuneModel
from experiment_utils import *
sys.path.remove("../scripts")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def salib_wrapper(target: str, step_count, **kwargs):
    def out_fun(inp):
        try:
            m = DuneModel(
                width=width,
                height=height,
                n_tribes=int(inp[0]),
                n_agents=int(inp[1]),
                n_heaps=int(inp[2]),
                vision_radius=int(inp[3]),
                step_count=step_count,
                alpha=float(inp[4]),
                trade_percentage=float(inp[5]),
                spice_movement_bias=float(inp[6]),
                tribe_movement_bias=float(inp[7]),
                spice_threshold=float(inp[8]),
                **kwargs
            )
            out = m.run_model()
            return out[target].to_numpy()[-1]
        except Exception as e:
            logging.error(f"Error processing input: {inp}, Error: {e}")
            raise e

    return out_fun

def parallel_evaluation(fun, samples, n_jobs=-1):
    with parallel_backend('loky', inner_max_num_threads=1):
        try:
            results = Parallel(n_jobs=n_jobs, timeout=600)(
                delayed(fun)(sample) for sample in tqdm(samples, desc="Evaluating samples")
            )
            return np.array(results)
        except Exception as e:
            logging.error(f"Error during parallel evaluation: {e}")
            raise e

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
            (400, 2000),# n_agents
            (1, 10),    # n_heaps
            (2, 50),    # vision_radius
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
        "total_Clustering",
        "Fights_per_step",
        "Cooperation_per_step"
    ]
    [f"Tribe_{i}_Clustering" for i in range(num_tribes)]

    results_dict = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples = sobol_sample.sample(problem, nr_sobol_samples)
    samples_csv = pd.DataFrame(samples, columns=problem['names'])
    samples_csv.to_csv(os.path.join(output_dir, f'sobol_samples_{nr_sobol_samples}.csv'), index=False)

    for step_count in step_counts:
        logging.info(f"Running sensitivity analysis for step_count: {step_count}")

        for param in output_params:
            logging.info(f"Running sensitivity analysis for {param} with step_count {step_count}")
            sensitivity_target = salib_wrapper(param, step_count, **kwargs)
            results = parallel_evaluation(sensitivity_target, samples)
            Si = sobol_analyze.analyze(problem, results.flatten())
            results_dict[param] = Si

            save_phase_plot_data(problem, samples, results, filename=os.path.join(output_dir, f"phase_plot_data_{param}_sample_{nr_sobol_samples}_step_{step_count}.csv"))

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
            sobol_indices_df.to_csv(os.path.join(output_dir, f'sobol_results_{param}_sample_{nr_sobol_samples}_step_{step_count}.csv'), index=False)

if __name__ == "__main__":
    
    height = 300
    width = 300
    nr_sobol_samples = 16
    
    step_counts = [50, 100, 150, 200]
    
    output_dir = "GSA"
    
    main()
