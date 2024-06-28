import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from scipy.interpolate import make_interp_spline

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
mpl.rcParams['text.usetex'] = False

#Pay no attention to the man behind the screen.
warnings.filterwarnings("ignore", category=DeprecationWarning, message='DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.')

df_random = pd.read_csv('data/ofat_500_just_heaps2.csv')
#df_clustered = pd.read_csv('data/sandor2_clustered_old_vision.csv')
#df_center_random = pd.read_csv('data/sandor2_center_heap_random_old_vision.csv')
#df_center_clustered = pd.read_csv('data/sandor2_center_heap_clustered_old_vision.csv')

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
#plot_single_ofat_result(df_random, 'n_heaps', 'Tribe_0_Nomads')

def plot_normal(ax, x, y, yerr, label, color):
        ax.plot(x, y, label=label, color=color)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

def plot_all_ofat_results(dfs, param_name, use_interpolation=True):
    print(f'Parameter of Interest: {param_name}')
    metrics = ['total_Clustering']
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax1, ax2 = axs[0], axs[1]

    def get_last_n_indices(group, n=5):
        max_value = group["Unnamed: 0"].max()
        filtered = group.loc[group["Unnamed: 0"] > (max_value - n)]
        return filtered

    def plot_with_interpolation(ax, x, y, yerr, label, color):
        ax.plot(x, y, label=label, color=color, marker='o')
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline_mean = make_interp_spline(x, y, k=3)
        y_smooth = spline_mean(x_smooth)

        spline_err = make_interp_spline(x, yerr, k=3)
        yerr_smooth = spline_err(x_smooth)

        ax.plot(x_smooth, y_smooth, label=f'{label}', color=color, alpha=0.5)
        ax.fill_between(x_smooth, y_smooth - yerr_smooth, y_smooth + yerr_smooth, color=color, alpha=0.1)

    dataset_labels = ['', '']
    colors = ['blue', 'red']
    
    # Plotting total_Clustering
    for i, metric in enumerate(metrics):
        ax = ax1
        for df, label, color in zip(dfs, dataset_labels, colors):
            grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
            grouped_metric = grouped.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
            replicates = grouped.groupby(param_name)[metric].count().reindex(grouped_metric[param_name]).values
            err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)

            if use_interpolation:
                plot_with_interpolation(ax, grouped_metric[param_name], grouped_metric['mean'], err, f'{label}', color)
            else:
                plot_normal(ax, grouped_metric[param_name], grouped_metric['mean'], err, f'{label}', color)
            
            max_mean_value = grouped_metric['mean'].max()
            max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
            print(f'{label} - Max Parameter Value for {metric}: {max_param_value}   Max Mean Value: {max_mean_value}')
        
        ax.set_xlabel('Number of Spice heaps')
        ax.set_ylabel('Clustering')
        ax.set_title(f'Clustering vs Number of Spice heaps')
        ax.grid(True)

    # Plotting Fights_per_step and Cooperation_per_step
    metrics_to_plot = ['Cooperation_per_step', 'Fights_per_step']
    cooperation_colors = ['green', 'orange']
    fighting_colors = ['purple', 'brown']
    dataset_labels = ['Fights', 'Cooperation', 'Random Fights', 'Clustered Fights']
    
    handles = []
    for metric, color_set in zip(metrics_to_plot, [cooperation_colors, fighting_colors]):
        for df, label, color in zip(dfs, dataset_labels, color_set):
            grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
            grouped_metric = grouped.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
            replicates = grouped.groupby(param_name)[metric].count().reindex(grouped_metric[param_name]).values
            err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)

            if use_interpolation:
                handle, = ax2.plot(grouped_metric[param_name], grouped_metric['mean'], label=label, color=color, marker='o')
                ax2.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color=color, alpha=0.2)
            
                x_smooth = np.linspace(grouped_metric[param_name].min(), grouped_metric[param_name].max(), 300)
                spline_mean = make_interp_spline(grouped_metric[param_name], grouped_metric['mean'], k=3)
                y_smooth = spline_mean(x_smooth)
                
                spline_err = make_interp_spline(grouped_metric[param_name], err, k=3)
                yerr_smooth = spline_err(x_smooth)
                ax2.plot(x_smooth, y_smooth, label=f'{label}', color=color, alpha=0.5)
                ax2.fill_between(x_smooth, y_smooth - yerr_smooth, y_smooth + yerr_smooth, color=color, alpha=0.1)
                
            else:
                handle, = ax2.plot(grouped_metric[param_name], grouped_metric['mean'], label=label, color=color)
                ax2.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color=color, alpha=0.2)

            handles.append(handle)
            
            max_mean_value = grouped_metric['mean'].max()
            max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
            print(f'{label} - Max Parameter Value for {metric}: {max_param_value}   Max Mean Value: {max_mean_value}')

    ax2.set_xlabel('Number of Spice heaps')
    ax2.set_ylabel('Interactions per Step')
    ax2.set_title(f'Interactions vs Number of Spice heaps')
    ax2.grid(True)
    ax2.legend(handles, dataset_labels, loc='upper left')
    
    plt.subplots_adjust(left=0.058, bottom=0.087, right=0.985, top=0.932, wspace=0.2, hspace=0.32)
    plt.tight_layout()
    plt.show()


# plot_all_ofat_results([df_random, df_clustered], 'n_agents', use_interpolation=False)
#plot_all_ofat_results([df_random], 'n_heaps', use_interpolation=False)
# plot_all_ofat_results([df_random, df_clustered], 'vision_radius', use_interpolation=False)

#plot_all_ofat_results([df_random, df_random], 'spice_threshold', use_interpolation=True)
#plot_all_ofat_results([df_random, df_random], 'n_heaps', use_interpolation=True)
# plot_all_ofat_results([df_center_random, df_center_clustered], 'vision_radius', use_interpolation=False)

def plot_averaged_results2(df, param_values):
    filtered_df = df[
        (df['n_tribes'] == param_values['n_tribes']) &
        (df['n_agents'] == param_values['n_agents']) &
        (df['n_heaps'] == param_values['n_heaps']) &
        (df['vision_radius'] == param_values['vision_radius']) &
        (df['step_count'] == param_values['step_count']) &
        (df['alpha'] == param_values['alpha']) &
        (df['trade_percentage'] == param_values['trade_percentage']) &
        (df['spice_movement_bias'] == param_values['spice_movement_bias']) &
        (df['tribe_movement_bias'] == param_values['tribe_movement_bias']) &
        (df['spice_grow_threshold'] == param_values['spice_grow_threshold']) &
        (df['spice_threshold'] == param_values['spice_threshold'])
    ]

    metrics = ['Tribe_1_Nomads']
    interaction_metrics = ['Fights_per_step', 'Cooperation_per_step']
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    grouped = filtered_df.groupby(filtered_df.iloc[:, 0]).agg({metric: ['mean', 'std'] for metric in metrics + interaction_metrics}).reset_index()

    colors = ['blue', 'red', 'green']

    # Plot total_Clustering
    ax = axs[0]
    for i, metric in enumerate(metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[i])

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Clustering')
        ax.set_title('Clustering over Time')
        ax.grid(True)

    # Plot Fights_per_step and Cooperation_per_step
    ax = axs[1]
    for i, metric in enumerate(interaction_metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[len(metrics) + i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Interactions per Step')
    ax.set_title('Fights and Cooperation over Time')
    ax.grid(True)
    ax.legend()
    
    plt.subplots_adjust(left=0.058, bottom=0.087, right=0.985, top=0.932, wspace=0.2, hspace=0.32)
    plt.tight_layout()
    plt.show()

#Parameter Values for Filtering
param_values = {
    'n_tribes': 2,
    'n_agents': 390, 
    'n_heaps': 1,
    'vision_radius': 10,
    'step_count': 500,
    'alpha': 0.5,
    'trade_percentage': 0.5,
    'spice_movement_bias': 1.0,
    'tribe_movement_bias': 0.0,
    'spice_grow_threshold': 20,
    'spice_threshold': 1290
}

#plot_averaged_results(df_random, param_values=param_values)
#plot_averaged_results(df_clustered, param_values=param_values)

# plot_averaged_results(df_center_random, param_values=param_values)
# plot_averaged_results(df_center_clustered, param_values=param_values)

def plot_averaged_results(df, param_values, steps_per_run=500):
    # Filtering based on param_values
    filtered_df = df[
        (df['n_tribes'] == param_values['n_tribes']) &
        (df['n_agents'] == param_values['n_agents']) &
        (df['n_heaps'] == param_values['n_heaps']) &
        (df['vision_radius'] == param_values['vision_radius']) &
        (df['step_count'] == param_values['step_count']) &
        (df['alpha'] == param_values['alpha']) &
        (df['trade_percentage'] == param_values['trade_percentage']) &
        (df['spice_movement_bias'] == param_values['spice_movement_bias']) &
        (df['tribe_movement_bias'] == param_values['tribe_movement_bias']) &
        (df['spice_grow_threshold'] == param_values['spice_grow_threshold']) &
        (df['spice_threshold'] == param_values['spice_threshold'])
    ]

    # Determine the number of runs
    num_runs = len(filtered_df) // steps_per_run
    runs_to_include = []

    for run in range(num_runs):
        run_start = run * steps_per_run
        run_end = run_start + steps_per_run
        run_df = filtered_df.iloc[run_start:run_end]
        if run_df.iloc[-1]['Tribe_0_Nomads'] > run_df.iloc[-1]['Tribe_1_Nomads']:
            runs_to_include.append(run_df)

    # Combine the included runs
    if runs_to_include:
        filtered_df = pd.concat(runs_to_include)
    else:
        filtered_df = pd.DataFrame(columns=filtered_df.columns)  # Empty DataFrame if no runs are included

    metrics = ['Tribe_0_Nomads']
    interaction_metrics = ['Fights_per_step', 'Cooperation_per_step']
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    grouped = filtered_df.groupby(filtered_df.index % steps_per_run).agg({metric: ['mean', 'std'] for metric in metrics + interaction_metrics}).reset_index()

    colors = ['blue', 'red', 'green']

    # Plot Tribe_0_Nomads
    ax = axs[0]
    for i, metric in enumerate(metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[i])

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of Nomads')
        ax.set_title('Tribe 0 Nomads over Time')
        ax.grid(True)

    # Plot Fights_per_step and Cooperation_per_step
    ax = axs[1]
    for i, metric in enumerate(interaction_metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[len(metrics) + i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Interactions per Step')
    ax.set_title('Fights and Cooperation over Time')
    ax.grid(True)
    ax.legend()
    
    plt.subplots_adjust(left=0.058, bottom=0.087, right=0.985, top=0.932, wspace=0.2, hspace=0.32)
    plt.tight_layout()
    plt.show()

# Read your CSV file into a DataFrame
# df = pd.read_csv('your_file.csv')

# Parameter Values for Filtering
param_values = {
    'n_tribes': 2,
    'n_agents': 390,
    'n_heaps': 1,
    'vision_radius': 10,
    'step_count': 500,
    'alpha': 0.5,
    'trade_percentage': 0.5,
    'spice_movement_bias': 1.0,
    'tribe_movement_bias': 0.0,
    'spice_grow_threshold': 20,
    'spice_threshold': 1290
}

# Assuming df_random is your DataFrame read from the CSV
#plot_averaged_results(df_random, param_values=param_values)


def plot_normal(ax, x, y, yerr, label, color):
    ax.errorbar(x, y, yerr=yerr, label=label, color=color)

def plot_averaged_results_for_run(df, param_values, run_index=0, steps_per_run=500):
    # Filtering based on param_values
    filtered_df = df[
        (df['n_tribes'] == param_values['n_tribes']) &
        (df['n_agents'] == param_values['n_agents']) &
        (df['n_heaps'] == param_values['n_heaps']) &
        (df['vision_radius'] == param_values['vision_radius']) &
        (df['step_count'] == param_values['step_count']) &
        (df['alpha'] == param_values['alpha']) &
        (df['trade_percentage'] == param_values['trade_percentage']) &
        (df['spice_movement_bias'] == param_values['spice_movement_bias']) &
        (df['tribe_movement_bias'] == param_values['tribe_movement_bias']) &
        (df['spice_grow_threshold'] == param_values['spice_grow_threshold']) &
        (df['spice_threshold'] == param_values['spice_threshold'])
    ]

    # Determine the number of runs
    num_runs = len(filtered_df) // steps_per_run

    if run_index >= num_runs:
        raise ValueError(f"Run index {run_index} is out of bounds for the number of runs {num_runs}")

    # Extract the specific run
    run_start = run_index * steps_per_run
    run_end = run_start + steps_per_run
    run_df = filtered_df.iloc[run_start:run_end]

    metrics = ['Tribe_0_Nomads', 'Tribe_1_Nomads']
    interaction_metrics = ['Tribe_0_Clustering', 'Tribe_1_Clustering']
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    grouped = run_df.groupby(run_df.index % steps_per_run).agg({metric: ['mean', 'std'] for metric in metrics + interaction_metrics}).reset_index()

    colors = ['blue', 'green', 'red', 'purple']

    # Plot Tribe_0_Nomads
    ax = axs[0]
    for i, metric in enumerate(metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[i])

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of Nomads')
        ax.set_title('Nomads over Time')
        ax.legend()
        ax.grid(True)

    # Plot Fights_per_step and Cooperation_per_step
    ax = axs[1]
    for i, metric in enumerate(interaction_metrics):
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[len(metrics) + i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Clustering per Step')
    ax.set_title('Clustering over Time')
    ax.grid(True)
    ax.legend()
    
    plt.subplots_adjust(left=0.058, bottom=0.087, right=0.985, top=0.932, wspace=0.2, hspace=0.32)
    plt.tight_layout()
    plt.show()

# Read your CSV file into a DataFrame
# df = pd.read_csv('your_file.csv')

# Parameter Values for Filtering
param_values = {
    'n_tribes': 2,
    'n_agents': 390,
    'n_heaps': 1,
    'vision_radius': 10,
    'step_count': 500,
    'alpha': 0.5,
    'trade_percentage': 0.5,
    'spice_movement_bias': 1.0,
    'tribe_movement_bias': 0.0,
    'spice_grow_threshold': 20,
    'spice_threshold': 1290
}

# Assuming df_random is your DataFrame read from the CSV
# Plot the first run (index 0)
plot_averaged_results_for_run(df_random, param_values=param_values, run_index=1)

# Plot the second run (index 1)
#plot_averaged_results_for_run(df_random, param_values=param_values, run_index=1)
