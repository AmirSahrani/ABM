import numpy as np
from model import DuneModel
from experiment_utils import *
import toml
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
                resulting_dfs.append(pd.DataFrame([result]))

    else:
        resulting_dfs = []
        for _ in range(trails):
            new_kwargs = kwargs.copy()
            new_kwargs["experiment_id"] = np.random.randint(1000000)
            new_kwargs["n_tribes"] = new_kwargs.get("n_tribes", kwargs["n_tribes"])
            new_kwargs["n_agents"] = new_kwargs.get("n_agents", kwargs["n_agents"])
            new_kwargs["n_heaps"] = new_kwargs.get("n_heaps", kwargs["n_heaps"])
            new_kwargs["vision_radius"] = new_kwargs.get("vision_radius", kwargs["vision_radius"])
            new_kwargs["alpha"] = new_kwargs.get("alpha", kwargs["alpha"])
            new_kwargs["trade_percentage"] = new_kwargs.get("trade_percentage", kwargs["trade_percentage"])
            result = main(new_kwargs)
            resulting_dfs.append(pd.DataFrame([result]))

    df = pd.concat(resulting_dfs, ignore_index=True)
    df.to_csv(f'data/{kwargs["experiment_name"]}.csv', index=False)
