import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import make_interp_spline

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

#Pay no attention to the man behind the screen.
warnings.filterwarnings("ignore", category=DeprecationWarning, message='DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.')

df_random = pd.read_csv('data/sandor_random_old_vision.csv')
df_clustered = pd.read_csv('data/sandor_clustered_old_vision.csv')
df_center_random = pd.read_csv('data/sandor_center_heap_random_old_vision.csv')
df_center_clustered = pd.read_csv('data/sandor_center_heap_clustered_old_vision.csv')

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

def plot_normal(ax, x, y, yerr, label, color):
        ax.plot(x, y, label=label, color=color)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

def plot_all_ofat_results(dfs, param_name, use_interpolation=True):
    print(f'Parameter of Interest: {param_name}')
    metrics = ['Nomads', 'total_Clustering']
    
    fig, axs = plt.subplots(2, 2, figsize=(30, 10))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    def get_last_n_indices(group, n=5):
        # Ensure sorting by 'Unnamed: 0' before processing
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

    dataset_labels = ['Random', 'Clustered']
    colors = ['blue', 'red']
    
    # Plotting Nomads and total_Clustering
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
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
        
        ax.set_xlabel(param_name)
        axs[0,0].set_ylabel('Population')
        axs[0,1].set_ylabel('Clustering')
        axs[0,0].set_title(f'Population vs {param_name}')
        axs[0,1].set_title(f'Clustering vs {param_name}')
        ax.grid(True)
        ax.legend()

    # Plotting Fights_per_step and Cooperation_per_step
    ax5 = axs[1, 1]
    metrics_to_plot = ['Cooperation_per_step', 'Fights_per_step']
    cooperation_colors = ['green', 'orange']
    fighting_colors = ['purple', 'brown']
    dataset_labels = ['Random Cooperation', 'Clustered Cooperation', 'Random Fights', 'Clustered Fights']
    
    handles = []
    for metric, color_set in zip(metrics_to_plot, [cooperation_colors, fighting_colors]):
        for df, label, color in zip(dfs, dataset_labels, color_set):
            grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
            grouped_metric = grouped.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
            replicates = grouped.groupby(param_name)[metric].count().reindex(grouped_metric[param_name]).values
            err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)

            if use_interpolation:
                handle, = ax5.plot(grouped_metric[param_name], grouped_metric['mean'], label=label, color=color, marker='o')
                ax5.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color=color, alpha=0.2)
            
                x_smooth = np.linspace(grouped_metric[param_name].min(), grouped_metric[param_name].max(), 300)
                spline_mean = make_interp_spline(grouped_metric[param_name], grouped_metric['mean'], k=3)
                y_smooth = spline_mean(x_smooth)
                
                spline_err = make_interp_spline(grouped_metric[param_name], err, k=3)
                yerr_smooth = spline_err(x_smooth)
                ax5.plot(x_smooth, y_smooth, label=f'{label}', color=color, alpha=0.5)
                ax5.fill_between(x_smooth, y_smooth - yerr_smooth, y_smooth + yerr_smooth, color=color, alpha=0.1)
                
            else:
                handle, = ax5.plot(grouped_metric[param_name], grouped_metric['mean'], label=label, color=color)
                ax5.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color=color, alpha=0.2)

            handles.append(handle)
            
            max_mean_value = grouped_metric['mean'].max()
            max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
            print(f'{label} - Max Parameter Value for {metric}: {max_param_value}   Max Mean Value: {max_mean_value}')

    ax5.set_xlabel(param_name)
    ax5.set_ylabel('Interactions per Step')
    ax5.set_title(f'Interactions vs {param_name}')
    ax5.grid(True)
    ax5.legend(handles, dataset_labels, loc='upper left')


    # Plotting average trades across all tribes
    dataset_labels = ['Random', 'Clustered']
    trades_columns = ['Tribe_{}_Trades'.format(i) for i in range(2)]
    for df, label, color in zip(dfs, dataset_labels, colors):
        df['average_trades'] = df[trades_columns].mean(axis=1)
        grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
        grouped_metric = grouped.groupby(param_name)['average_trades'].agg(['mean', 'max', 'min', 'std']).reset_index()
        replicates = grouped.groupby(param_name)['average_trades'].count().reindex(grouped_metric[param_name]).values
        err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)

        if use_interpolation:
            plot_with_interpolation(ax3, grouped_metric[param_name], grouped_metric['mean'], err, f'{label}', color)
        else:
            plot_normal(ax3, grouped_metric[param_name], grouped_metric['mean'], err, f'{label}', color)
        
        max_mean_value = grouped_metric['mean'].max()
        max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
        print(f'{label} - Max Parameter Value for Average Trades: {max_param_value}   Max Mean Value: {max_mean_value}')

    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Average Trades')
    ax3.set_title(f'Average Trades vs {param_name}')
    ax3.grid(True)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(left=0.055, bottom=0.079, right=0.992, top=0.95, wspace=0.179, hspace=0.31)
    plt.show()


plot_all_ofat_results([df_random, df_clustered], 'n_agents', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'n_heaps', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'vision_radius', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'trade_percentage', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'spice_movement_bias', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'tribe_movement_bias', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'spice_threshold', use_interpolation=False)
plot_all_ofat_results([df_random, df_clustered], 'spice_grow_threshold', use_interpolation=False)

plot_all_ofat_results([df_center_random, df_center_clustered], 'n_agents', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'n_heaps', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'vision_radius', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'trade_percentage', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'spice_movement_bias', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'tribe_movement_bias', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'spice_threshold', use_interpolation=False)
plot_all_ofat_results([df_center_random, df_center_clustered], 'spice_grow_threshold', use_interpolation=False)

def plot_averaged_results(df, param_values):
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

    metrics = ['Nomads', 'total_Clustering', 'average_trades']
    interaction_metrics = ['Fights_per_step', 'Cooperation_per_step']
    fig, axs = plt.subplots(2, 2)
    trades_columns = ['Tribe_0_Trades', 'Tribe_1_Trades', 'Tribe_2_Trades']
    filtered_df['average_trades'] = filtered_df[trades_columns].mean(axis=1)
    grouped = filtered_df.groupby(filtered_df.iloc[:, 0]).agg({metric: ['mean', 'std'] for metric in metrics + interaction_metrics}).reset_index()

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Plot Nomads, total_Clustering, and average_trades
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        x = grouped.iloc[:, 0]
        y = grouped[metric]['mean']
        yerr = grouped[metric]['std']

        plot_normal(ax, x, y, yerr, metric, colors[i])

        ax.set_xlabel('Time Step')
        axs[0,0].set_ylabel('Population')
        axs[0,0].set_title('Population over Time')
        axs[0,1].set_ylabel('Clustering')
        axs[0,1].set_title('Clustering over Time')
        axs[1,0].set_ylabel('Average Trades')
        axs[1,0].set_title('Average Trades over Time')
        ax.grid(True)

    # Plot Fights_per_step and Cooperation_per_step
    ax = axs[1, 1]
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
    'n_tribes': 3,
    'n_agents': 900,
    'n_heaps': 3,
    'vision_radius': 10,
    'step_count': 100,
    'alpha': 0.5,
    'trade_percentage': 0.8,
    'spice_movement_bias': 0.07,
    'tribe_movement_bias': 0.15,
    'spice_grow_threshold': 20,
    'spice_threshold': 9150
}

# plot_averaged_results(df_random, param_values=param_values)
# plot_averaged_results(df_clustered, param_values=param_values)

plot_averaged_results(df_center_random, param_values=param_values)
plot_averaged_results(df_center_clustered, param_values=param_values)
