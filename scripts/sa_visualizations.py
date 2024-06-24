import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/ofat_new_nominal_values.csv')

#This produces a plot similar to the one in the paper. 
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
    

#Give the parameter you're looking at and the particular output of interest. 
#plot_single_ofat_result(df, 'vision_radius', 'Fights_per_step')

def plot_all_ofat_results(df, param_name):
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

        


    # Plotting Nomads and total_Clustering
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
        grouped_metric = grouped.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
        replicates = grouped.groupby(param_name)[metric].count().reindex(grouped_metric[param_name]).values
        err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)
        ax.plot(grouped_metric[param_name], grouped_metric['mean'], label='Mean', color='blue', marker='o')
        ax.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color='blue', alpha=0.2)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs {param_name}')
        ax.grid(True)  # For Sandor lol, I like em too

        max_mean_value = grouped_metric['mean'].max()
        max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
        print(f'Max Parameter Value: {max_param_value}   Max Mean Value: {max_mean_value}')
    
    # Plotting Fights_per_step and Cooperation_per_step
    ax5 = axs[1, 1]
    metrics_to_plot = ['Cooperation_per_step', 'Fights_per_step']
    for metric in metrics_to_plot:
        grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
        grouped_metric = grouped.groupby(param_name)[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
        replicates = grouped.groupby(param_name)[metric].count().reindex(grouped_metric[param_name]).values
        err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)
        ax5.plot(grouped_metric[param_name], grouped_metric['mean'], label=metric, linestyle='-', marker='o')
        ax5.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, alpha=0.2)
        
        max_mean_value = grouped_metric['mean'].max()
        max_param_value = grouped_metric.loc[grouped_metric['mean'].idxmax(), param_name]
        print(f'Max Parameter Value: {max_param_value}   Max Mean Value: {max_mean_value}')

    ax5.set_xlabel(param_name)
    ax5.set_ylabel('Cooperation_per_step')
    ax5.set_title(f'{metric} vs {param_name}')
    ax5.grid(True)
    ax5.legend(loc='upper left')

    # Plotting average trades across all tribes
    trades_columns = ['Tribe_{}_Trades'.format(i) for i in range(3)]
    df['average_trades'] = df[trades_columns].mean(axis=1)
    grouped = df.groupby(param_name).apply(lambda x: get_last_n_indices(x)).reset_index(drop=True)
    grouped_metric = grouped.groupby(param_name)['average_trades'].agg(['mean', 'max', 'min', 'std']).reset_index()
    replicates = grouped.groupby(param_name)['average_trades'].count().reindex(grouped_metric[param_name]).values
    err = (1.96 * grouped_metric['std']) / np.sqrt(replicates)
    ax3.plot(grouped_metric[param_name], grouped_metric['mean'], label='Average Trades', color='purple', marker='o')
    ax3.fill_between(grouped_metric[param_name], grouped_metric['mean'] - err, grouped_metric['mean'] + err, color='purple', alpha=0.2)
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Average Trades')
    ax3.set_title(f'Average Trades vs {param_name}')
    ax3.grid(True)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(left=0.046, bottom=0.268, right=0.988, top=0.964, wspace=0.129, hspace=0.23)
    plt.show()
    

#plot_all_ofat_results(df, 'n_tribes')
plot_all_ofat_results(df, 'n_agents')
plot_all_ofat_results(df, 'n_heaps')
plot_all_ofat_results(df, 'vision_radius')
plot_all_ofat_results(df, 'trade_percentage')
plot_all_ofat_results(df, 'spice_movement_bias')
plot_all_ofat_results(df, 'tribe_movement_bias')
plot_all_ofat_results(df, 'spice_threshold')
