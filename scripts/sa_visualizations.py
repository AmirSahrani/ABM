import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ofat_results_toml.csv')

def plot_ofat_results(df, param_name, output_name):
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
    
def plot_all_values(df, param_name, output_name):
    df_param = df[[param_name, output_name]]
    plt.figure(figsize=(10, 6))
    plt.scatter(df_param[param_name], df_param[output_name], marker='o', color='blue')

    plt.xlabel(f'{param_name}')
    plt.ylabel(f'{output_name}')
    #plt.title(f'Number of Nomads for each {param_name}')
    plt.legend()
    plt.show()
    

#plot_all_values(df, 'n_heaps', 'Cooperation_per_step')
#plot_ofat_results(df, 'vision_radius', 'Fights_per_step')
plot_all_values(df, 'vision_radius', 'Fights_per_step')
#plot_ofat_results(df, 'vision_radius', 'Nomads')
#plot_ofat_results(df, 'alpha', 'Nomads')
#plot_ofat_results(df, 'trade_percentage', 'Nomads')
