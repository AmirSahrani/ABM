import numpy as np
from experiment_utils import *
from model import DuneModel
import toml as toml
from sys import argv
import multiprocessing


def main(kwargs):
    model = DuneModel(**kwargs)
    model.run_model()


if __name__ == "__main__":
    config_file = argv[1]
    # Read configuration from the TOML file
    with open(config_file, 'r') as f:
        config = toml.load(f)

    function_names = config.pop('functions')
    variables = config.pop('ranges')
    kwargs = config.pop('parameters')

    all_experiments = []

    # Convert function names to actual function references
    for key, func_name in function_names.items():
        kwargs[key] = eval(func_name) if isinstance(func_name, str) else func_name

    for param, val in variables.items():
        for iter_val in np.arange(val[0], val[1], val[2]):
            kwargs["experiment_name"] += f'_{param}_{iter_val}'
            kwargs[param] = iter_val
            all_experiments.append(kwargs)

    with multiprocessing.Pool() as pool:
        pool.map(main, all_experiments)
