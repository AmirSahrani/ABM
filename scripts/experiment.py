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

def generate_values(start, end, num_values, as_int=False):
    values = np.linspace(start, end, num_values)
    return values.astype(int) if as_int else values

def main(kwargs: dict):
    model = DuneModel(**kwargs)
    results = model.run_model()
    for key, value in kwargs.items():
        results[key] = value
    return results

if __name__ == "__main__":
    config_file = argv[1]
    with open(config_file, 'r') as f:
        config = toml.load(f)

    function_names = config.pop('functions')
    kwargs = config.pop('parameters')
    trails = config.pop('trails')
    try:
        int_ranges = config.pop('integer_ranges')
    except KeyError:
        int_ranges = None
    try:
        float_ranges = config.pop('float_ranges')
    except KeyError:
        float_ranges = None

    experiments = []

    for key, func_name in function_names.items():
        kwargs[key] = eval(func_name) if isinstance(func_name, str) else func_name

    #Integer ranges
    if int_ranges:
        for param, val in int_ranges.items():
            values = generate_values(val[0], val[1], 10, as_int=True)
            for iter_val in values:
                for _ in range(trails):  
                    new_kwargs = kwargs.copy()
                    new_kwargs["experiment_name"] += f'_{param}_{iter_val}'
                    new_kwargs[param] = iter_val
                    experiments.append(new_kwargs)

    #Floating-point ranges
    if float_ranges:
        for param, val in float_ranges.items():
            values = generate_values(val[0], val[1], 10, as_int=False)
            for iter_val in values:
                for _ in range(trails):
                    new_kwargs = kwargs.copy()
                    new_kwargs["experiment_name"] += f'_{param}_{iter_val}'
                    new_kwargs[param] = iter_val
                    experiments.append(new_kwargs)

    resulting_dfs = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.map(main, experiments)):
            result['experiment_id'] = np.random.randint(1000000)
            resulting_dfs.append(pd.DataFrame(result, index=[0]))

    df = pd.concat(resulting_dfs, ignore_index=True)
    df.to_csv(f'data/{kwargs["experiment_name"]}.csv', index=False)
