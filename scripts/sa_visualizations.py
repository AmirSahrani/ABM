import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ofat_results.csv')

def plot_ofat_results(df, param_name):
    df_param = df[df['param'] == param_name]
    grouped = df_param.groupby('value')['Nomads'].agg(['mean', 'max', 'min']).reset_index()
    
    plt.figure(figsize=(10, 6))
    for idx, row in grouped.iterrows():
        plt.scatter(row['value'], row['mean'], label='Mean' if idx == 0 else "", marker='o', color='blue')
        plt.scatter(row['value'], row['max'], label='Max' if idx == 0 else "", marker='^', color='green')
        plt.scatter(row['value'], row['min'], label='Min' if idx == 0 else "", marker='s', color='red')

    plt.xlabel(f'{param_name}')
    plt.ylabel('Number of Nomads')
    plt.title(f'Mean, Max, and Min Number of Nomads for {param_name}')
    plt.legend()
    plt.show()


# plot_ofat_results(df, 'n_heaps')
# plot_ofat_results(df, 'vision_radius')
# plot_ofat_results(df, 'step_count')
# plot_ofat_results(df, 'alpha')
# plot_ofat_results(df, 'trade_percentage')
