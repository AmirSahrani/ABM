# Project: Agent-Based Modeling

## Description

This project implements an Agent-Based Model (ABM) where different agents, known as nomads, belong to distinct tribes. The nomads can hunt for a valuable resource called spice, which is essential for their survival and prosperity. Each tribe can trade resources internally, either fight with or cooperate with other tribes, and adapt their strategies based on the environment and interactions.

The main objectives of this model are to study the dynamics of resource acquisition, trade, and conflict among different tribes, and to explore the emergent behaviors that arise from these interactions.

## Usage

### Server
To run the server, you need to follow these steps:

1. **Install Dependencies**: Ensure that you have Python and required packages installed. You can install the necessary packages using `pip`:

    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Server**: Start the server by executing:
    ```sh
    python server.py
    ```
### Experimenting

To run experiments with different configurations, you can use a TOML file to specify the parameters. Here’s how you can do it:

1. **Create a Configuration File**: Create a TOML file (e.g., `config.toml`) with the desired parameters. Below is an example configuration file:

    ```toml
    [parameters]
    experiment_name = "test"
    width = 100
    height = 100
    n_tribes = 3
    n_agents = 100
    vision_radius = 4
    step_count = 200
    alpha = 0.2
    trade_percentage = 0.3

    [functions]
    spice_generator = "gen_spice_map"
    river_generator = "gen_river"
    location_generator = "random_locations"

    [ranges]
    n_heaps = [1, 10, 1]

    ```

2. **Run the Experiment**: Execute the script with the configuration file to run the model with different parameter settings in parallel. Use the following command:
    ```sh
    python scripts/experiment.py configs/your_config.toml
    ```
