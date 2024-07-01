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
from joblib import Parallel, delayed

sys.path.append("../scripts")
sys.path.remove("../scripts")


def sensitivity_target(inp, i):
    '''
    Wrapper for the model, averages over all the possible maps we have defined, cov_range, total_spice,spice_grow_threshold, and step_count are hard coded. total_spice is 0.2*width*height of the grid. 
    '''
    random_spice_map_picker = lambda *args: np.random.choice([gen_spice_map, gen_central_spice_heap])( *args)
    random_river_picker = lambda *args: np.random.choice([gen_river_random, gen_river_line, no_river])( *args)
    random_location_picker = lambda *args: np.random.choice([random_locations, split_tribes_locations, tribe_locations_naturally_distributed, tribe_locations_single_cluster_per_tribe])( *args)
    kwargs = {
        "experiment_name": "test",
        "spice_generator": random_spice_map_picker,
        "river_generator": random_river_picker,
        "location_generator": random_location_picker,
        "spice_kwargs": {"cov_range": (8, 20), "total_spice": 10000},

    }
    try:
        m = DuneModel(
            width=width,
            height=height,
            n_tribes=int(inp[0]),
            n_agents=int(width*height/10),
            n_heaps=int(inp[1]),
            vision_radius=int(inp[2]), step_count=300, 
            trade_percentage=float(inp[3]),
            spice_movement_bias=float(inp[4]),
            tribe_movement_bias=float(inp[5]),
            spice_threshold=float(inp[6]*kwargs["spice_kwargs"]["total_spice"]),
            spice_grow_threshold=20,
            frequency=50,
            **kwargs
        )
        out = m.run_model()
        print(f'Experiment {i}')
        return out
    except Exception as e:
        print(f"Error processing input: {inp}, Error: {e}")
        raise e



def parallel_evaluation(fun, samples, n_jobs=None):
    '''
    Evaluate the model on the samples in parallel.
    
    Parameters:
    fun (Callable): The function to be evaluated.
    samples (iterable): An iterable of samples to be passed to the function.
    n_jobs (int, optional): The number of jobs to run in parallel. Defaults to the number of CPU cores.
    
    Returns:
    np.ndarray: A 3D numpy array with evaluation results.
    '''
    if n_jobs is None:
        n_jobs = os.cpu_count()  # Automatically set to the number of available CPU cores

    output_params = [
        "total_Clustering",
        "Fights_per_step",
        "Cooperation_per_step"
    ]
    
    out = np.empty((len(samples), len(output_params), 10))

    # Use joblib's Parallel and delayed for multiprocessing
    print(f'Starting experiments, total: {len(samples)}')
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(fun)(sample, i) for i, sample in enumerate(samples))

    for i, result in enumerate(results):
        for j, param in enumerate(output_params):
            print(result[param][-10:])
            out[i, j, :] = result[param][-10:]

    return out


def save_phase_plot_data(problem, samples, results, filename="phase_plot_data.csv"):
    '''
    Helper function for generating data needed for the generation of phase-diagrams.
    '''
    data = pd.DataFrame(samples, columns=problem['names'])
    data['result'] = results
    data.to_csv(filename, index=False)


def main():
    num_tribes = 1
    problem = {
        'num_vars': 7,
        'names': [
            "n_tribes",
            "n_heaps",
            "vision_radius",
            "trade_percentage",
            "spice_movement_bias",
            "tribe_movement_bias",
            "spice_threshold"
        ],
        'bounds': [
            (2, 4),     # n_tribes
            (1, 10),    # n_heaps
            (2, 13),    # vision_radius
            (0.0, 1.0),  # trade_percentage
            (0.3, 1.0),  # spice_movement_bias
            (0.0, 0.5),  # tribe_movement_bias
            (0.0, 0.8)  # spice_threshold
        ]
    }

    # [f"Tribe_{i}_Clustering" for i in range(num_tribes)]

    results_dict = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    samples = sobol_sample.sample(problem, nr_sobol_samples, calc_second_order=False)
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

            save_phase_plot_data(problem, samples, results, filename=os.path.join(output_dir, f"phase_plot_data_{output_params[param]}_sample_{nr_sobol_samples}_step_{(step_count + 1 )*50}.csv"))

            sobol_indices_data = {
                'Parameter': problem['names'],
                'S1': Si['S1'],
                'S1_conf': Si['S1_conf'],
                'ST': Si['ST'],
                'ST_conf': Si['ST_conf']
            }

            # num_vars = problem['num_vars']
            # S2 = Si['S2']
            # S2_conf = Si['S2_conf']
            # for i in range(num_vars):
            #     for j in range(i + 1, num_vars):
            #         sobol_indices_data[f'S2_{problem["names"][i]}_{problem["names"][j]}'] = S2[i, j]
            #         sobol_indices_data[f'S2_conf_{problem["names"][i]}_{problem["names"][j]}'] = S2_conf[i, j]

            sobol_indices_df = pd.DataFrame(sobol_indices_data)
            sobol_indices_df.to_csv(os.path.join(output_dir, f'sobol_results_{output_params[param]}_sample_{nr_sobol_samples}_step_{(step_count+1)*50}.csv'), index=False)


if __name__ == "__main__":

    height = 100
    width = 100
    nr_sobol_samples = 1024

    output_dir = "GSA"

    main()
