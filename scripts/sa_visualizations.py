import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ofat_results_joana_finer_grain.csv')

# This produces a plot similar to the one in the paper.


def plot_single_ofat_result(df, param_name, output_name):
    grouped = df.groupby(param_name)[output_name].agg(['mean', 'max', 'min']).reset_index()

    plt.figure(figsize=(10, 6))
    for idx, row in grouped.iterrows():
        plt.scatter(row[param_name], row['mean'], label='Mean' if idx == 0 else "", marker='o', color='blue')
        plt.scatter(row[param_name], row['max'], label='Max' if idx == 0 else "", marker='^', color='green')
        plt.scatter(row[param_name], row['min'], label='Min' if idx == 0 else "", marker='s', color='red')

    plt.xlabel(f'{param_name}')
    plt.ylabel(f'{output_name}')
    plt.title(f'Mean, Max, and Min of {output_name} for {param_name}')
    plt.legend()
    plt.show()


# Give the parameter you're looking at and the particular output of interest.
# plot_single_ofat_result(df, 'vision_radius', 'Fights_per_step')


def plot_all_ofat_results(df, param_name):
    metrics = ['Nomads', 'Cooperation_per_step', 'Fights_per_step', 'total_Clustering']
    fig, axs = plt.subplots(1, 4, figsize=(30, 5))

    for i, metric in enumerate(metrics):
        ax = axs[i]
        grouped = df.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
        replicates = df.groupby(param_name)[metric].count().reindex(grouped[param_name]).values
        err = (1.96 * grouped['std']) / np.sqrt(replicates)
        ax.plot(grouped[param_name], grouped['mean'], label='Mean', color='blue')
        ax.fill_between(grouped[param_name], grouped['mean'] - err, grouped['mean'] + err, color='blue', alpha=0.2)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs {param_name}')
        ax.grid(True)  # For Sandor lol, I like em too

    plt.subplots_adjust(left=0.042,
                        bottom=0.1,
                        right=0.994,
                        top=0.9,
                        wspace=0.326,
                        hspace=0.2)
    plt.show()


plot_all_ofat_results(df, 'n_tribes')
plot_all_ofat_results(df, 'n_agents')
plot_all_ofat_results(df, 'n_heaps')
plot_all_ofat_results(df, 'vision_radius')
plot_all_ofat_results(df, 'alpha')
plot_all_ofat_results(df, 'trade_percentage')
plot_all_ofat_results(df, 'spice_movement_bias')
plot_all_ofat_results(df, 'tribe_movement_bias')
plot_all_ofat_results(df, 'spice_threshold')
