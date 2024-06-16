import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ofat_analysis.csv')

def plot_ofat_results(df, param_name):
    grouped = df.groupby(param_name)['Nomads'].agg(['mean', 'max', 'min']).reset_index()
    print(grouped)
    plt.figure(figsize=(10, 6))
    for idx, row in grouped.iterrows():
        plt.scatter(row[param_name], row['mean'], label='Mean' if idx == 0 else "", marker='o', color='blue')
        plt.scatter(row[param_name], row['max'], label='Max' if idx == 0 else "", marker='^', color='green')
        plt.scatter(row[param_name], row['min'], label='Min' if idx == 0 else "", marker='s', color='red')

    plt.xlabel(f'{param_name}')
    plt.ylabel('Number of Nomads')
    plt.title(f'Mean, Max, and Min Number of Nomads for {param_name}')
    plt.legend()
    plt.show()


plot_ofat_results(df, 'n_tribes')
