import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=24)


def plot_sobol_indices(csv_files, phase_plot_data_files, output_dir):
    all_sobol_indices = []

    for csv_file in csv_files:
        sobol_indices_df = pd.read_csv(csv_file)
        sobol_indices_df['Output'] = os.path.basename(csv_file).split('sobol_results_')[1].split('.csv')[0]
        all_sobol_indices.append(sobol_indices_df)

    all_sobol_indices_df = pd.concat(all_sobol_indices, ignore_index=True)

    problem_names = all_sobol_indices_df['Parameter'].unique()
    problem_names = [x.replace("_", " ").capitalize() for x in problem_names]

    output_parameters = all_sobol_indices_df['Output'].unique()
    clean_out = ["Cooperation", "Fights", "Clusteing"]

    num_params = len(problem_names)
    num_outputs = len(output_parameters)
    bar_width = 1 / (num_outputs + 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, output_param in enumerate(output_parameters):
        data = all_sobol_indices_df[all_sobol_indices_df['Output'] == output_param]
        indices = np.arange(num_params)
        ax.bar(indices + i * bar_width, data['S1'], yerr=data['S1_conf'], width=bar_width, align='center', alpha=0.7, capsize=5, label=clean_out[i])

    ax.set_xticks(indices + bar_width * (num_outputs - 1) / 2)
    ax.set_xticklabels(problem_names, rotation=90)
    ax.set_ylabel(r'$S_1$')
    # ax.set_title(f'First-order Sobol Indices ($S_1$) at Time Step {timestep} across Output Parameters')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_S1_combined_timestep_{timestep}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, output_param in enumerate(output_parameters):
        data = all_sobol_indices_df[all_sobol_indices_df['Output'] == output_param]
        indices = np.arange(num_params)
        ax.bar(indices + i * bar_width, data['ST'], yerr=data['ST_conf'], width=bar_width, align='center', capsize=5, alpha=0.7, label=clean_out[i])

    ax.set_xticks(indices + bar_width * (num_outputs - 1) / 2)
    ax.set_xticklabels(problem_names, rotation=90)
    ax.set_ylabel(r'$S_T$')
    # ax.set_title(f'Total-order Sobol Indices ($S_T$) at Time Step {timestep} across Output Parameters')
    # ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sobol_ST_combined_timestep_{timestep}.png'))
    plt.close()

    for phase_plot_data_file in phase_plot_data_files:
        phase_data = pd.read_csv(phase_plot_data_file)

        result_col_raw = os.path.basename(phase_plot_data_file).split('_sample_')[0].replace('phase_plot_data_', '')
        result_col = result_col_raw.replace('_', ' ').title()
        print(f"result_col: {result_col}")
        print(f"Columns: {phase_data.columns.tolist()}")

        def generate_phase_plots(data, param1, param2, result_col):
            x = data[param1]
            y = data[param2]
            z = data['result']

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
            # ("n_tribes", "n_agents"),
            # ("alpha", "vision_radius"),
            ("trade_percentage", "spice_threshold"),
            ("tribe_movement_bias", "spice_movement_bias")
        ]

        # for param1, param2 in parameter_pairs:
        #     generate_phase_plots(phase_data, param1, param2, result_col)
        #

if __name__ == "__main__":
    input_dir = "GSA"
    output_dir = "GSA"
    timestep = 300

    csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.startswith('sobol_results_') and file.endswith('.csv') and str(timestep) in file]
    phase_plot_data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.startswith('phase_plot_data_') and file.endswith('.csv')]

    print("Processing Sobol indices...")
    plot_sobol_indices(csv_files, phase_plot_data_files, output_dir)
