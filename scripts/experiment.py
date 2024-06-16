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


def main(kwargs: dict):
    model = DuneModel(**kwargs)
    return model.run_model()


if __name__ == "__main__":
    config_file = argv[1]
    with open(config_file, 'r') as f:
        config = toml.load(f)

    function_names = config.pop('functions')
    kwargs = config.pop('parameters')
    trails = config.pop('trails')
    try:
        variables = config.pop('ranges')
    except KeyError:
        variables = None
    experiments = []

    # Convert function names to actual function references
    for key, func_name in function_names.items():
        kwargs[key] = eval(func_name) if isinstance(func_name, str) else func_name

    if variables:
        for param, val in variables.items():
            for iter_val in np.arange(val[0], val[1], val[2]):
                for _ in range(trails):  # Add each configuration multiple times
                    new_kwargs = kwargs.copy()
                    new_kwargs["experiment_name"] += f'_{param}_{iter_val}'
                    new_kwargs[param] = iter_val
                    experiments.append(new_kwargs)

        resulting_dfs = []
        with multiprocessing.Pool() as pool:
            for result in tqdm(pool.map(main, experiments)):
                result['experiment_id'] = np.random.randint(1000000)
                resulting_dfs.append(result)

    else:
        resulting_dfs = []
        for _ in range(trails):
            result = main(kwargs)
            result['experiment_id'] = np.random.randint(1000000)
            resulting_dfs.append(result)

    df = pd.concat(resulting_dfs)
    df.to_csv(f'data/{kwargs["experiment_name"]}.csv')
