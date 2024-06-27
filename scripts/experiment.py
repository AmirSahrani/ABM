import numpy as np
from model import DuneModel
from experiment_utils import *
import toml as toml
from sys import argv
from tqdm import tqdm
import pandas as pd
import multiprocessing
import warnings

warnings.filterwarnings("ignore")


def generate_experiments(kwargs, variables, trials):
    experiments = []

    if variables:
        for param, val in variables.items():
            for iter_val in np.arange(val[0], val[1], val[2]):
                new_kwargs = kwargs.copy()
                new_kwargs[param] = iter_val
                experiments.extend([new_kwargs] * trials)
    return experiments


def main(kwargs: dict):
    id = np.random.randint(2e7)
    model = DuneModel(**kwargs)
    print(f"Running an experiment {id}")
    return kwargs, model.run_model()


if __name__ == "__main__":
    config_file = argv[1]
    with open(config_file, 'r') as f:
        config = toml.load(f)

    function_names = config.pop('functions')
    kwargs = config.pop('parameters')
    trials = config.pop('trials')
    try:
        variables = config.pop('ranges')
    except KeyError:
        variables = None

    experiments = []

# Convert function names to actual function references
    for key, func_name in function_names.items():
        kwargs[key] = eval(func_name) if isinstance(func_name, str) else func_name

    if variables:
        experiments = generate_experiments(kwargs, variables, trials)
    else:
        experiments = [kwargs] * trials
        
    resulting_dfs = []
    print(f"Running {len(experiments)} experiments")
    with multiprocessing.Pool(10) as pool:
        for kwargs, result in pool.imap_unordered(main, experiments):
            for key, val in kwargs.items():
                result[key] = val
            resulting_dfs.append(result)


    df = pd.concat(resulting_dfs)
    df.to_csv(f'data/{kwargs["experiment_name"]}.csv')
