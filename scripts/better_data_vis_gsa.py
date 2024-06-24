import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

def plot_sobol_indices(csv_file, phase_plot_data_file, output_dir):
    sobol_indices_df = pd.read_csv(csv_file)

    problem_names = sobol_indices_df['Parameter'].tolist()
    S1 = sobol_indices_df['S1'].values
    S1_conf = sobol_indices_df['S1_conf'].values
    ST = sobol_indices_df['ST'].values
    ST_conf = sobol_indices_df['ST_conf'].values

    S2_matrix = np.zeros((len(problem_names), len(problem_names)))
    S2_conf_matrix = np.zeros((len(problem_names), len(problem_names)))
    num_vars = len(problem_names)
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            S2_label = f'S2_{problem_names[i]}_{problem_names[j]}'
            S2_conf_label = f'S2_conf_{problem_names[i]}_{problem_names[j]}'
            if S2_label in sobol_indices_df.columns and S2_conf_label in sobol_indices_df.columns:
                S2_matrix[i, j] = sobol_indices_df[S2_label].values[0]
                S2_matrix[j, i] = sobol_indices_df[S2_label].values[0]
                S2_conf_matrix[i, j] = sobol_indices_df[S2_conf_label].values[0]
                S2_conf_matrix[j, i] = sobol_indices_df[S2_conf_label].values[0]
            else:
                print(f"Warning: {S2_label} or {S2_conf_label} not found in the CSV columns")

    fig, ax = plt.subplots(figsize=(12, 8))
    indices = range(len(problem_names))
    ax.bar(indices, S1, yerr=S1_conf, align='center', capsize=5)
    ax.set_xticks(indices)
    ax.set_xticklabels(problem_names, rotation=90)
    ax.set_ylabel(r'$S_1$')
    ax.set_title(f'First-order Sobol Indices ($S_1$) for {os.path.basename(csv_file)}')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_S1_results_{os.path.splitext(os.path.basename(csv_file))[0]}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(indices, ST, yerr=ST_conf, align='center', capsize=5, alpha=0.7)
    ax.set_xticks(indices)
    ax.set_xticklabels(problem_names, rotation=90)
    ax.set_ylabel(r'$S_T$')
    ax.set_title(f'Total-order Sobol Indices ($S_T$) for {os.path.basename(csv_file)}')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_ST_results_{os.path.splitext(os.path.basename(csv_file))[0]}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(len(S2_matrix[np.triu_indices(len(problem_names), 1)])), 
           S2_matrix[np.triu_indices(len(problem_names), 1)], 
           yerr=S2_conf_matrix[np.triu_indices(len(problem_names), 1)], 
           align='center', capsize=5)
    ax.set_xticks(range(len(S2_matrix[np.triu_indices(len(problem_names), 1)])))
    S2_labels = [f'{problem_names[i]}-{problem_names[j]}' for i in range(num_vars) for j in range(i + 1, num_vars)]
    ax.set_xticklabels(S2_labels, rotation=90)
    ax.set_ylabel(r'$S_2$')
    ax.set_title(f'Second-order Sobol Indices ($S_2$) for {os.path.basename(csv_file)}')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_S2_results_{os.path.splitext(os.path.basename(csv_file))[0]}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(S2_matrix, xticklabels=problem_names, yticklabels=problem_names, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title(f'Second-order Sobol Indices Heatmap for {os.path.basename(csv_file)}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_S2_heatmap_{os.path.splitext(os.path.basename(csv_file))[0]}.png'))
    plt.close()

    phase_data = pd.read_csv(phase_plot_data_file)

    def generate_phase_plots(data, param1, param2, result_col='result'):
        x = data[param1]
        y = data[param2]
        z = data[result_col]

        plt.figure(figsize=(10, 8))
        plt.tricontourf(x, y, z, levels=14, cmap='RdBu_r')
        plt.colorbar(label=result_col)
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(rf'Phase plot of {result_col} in {param1}-{param2} space')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'phase_plot_{param1}_{param2}_{os.path.splitext(os.path.basename(phase_plot_data_file))[0]}.png'))
        plt.close()

    parameter_pairs = [
        ("n_tribes", "n_agents"),
        ("alpha", "vision_radius"),
        ("trade_percentage","spice_threshold"),
        ("tribe_movement_bias", "spice_movement_bias")
    ]

    for param1, param2 in parameter_pairs:
        generate_phase_plots(phase_data, param1, param2)

if __name__ == "__main__":
    input_dir = "GSA"
    output_dir = "GSA"

    csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.startswith('sobol_results_') and file.endswith('.csv')]
    phase_plot_data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.startswith('phase_plot_data_') and file.endswith('.csv')]

    for csv_file, phase_plot_data_file in zip(csv_files, phase_plot_data_files):
        plot_sobol_indices(csv_file, phase_plot_data_file, output_dir)
