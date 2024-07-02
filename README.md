# Project: Agent-Based Modeling

## Description

This project implements an Agent-Based Model (ABM) where different agents, known as nomads, belong to distinct tribes. The nomads can hunt for a valuable resource called spice, which is essential for their survival and prosperity. Each tribe can trade resources internally, either fight with or cooperate with other tribes, and adapt their strategies based on the environment and interactions.

The main objectives of this model are to study the dynamics of resource acquisition, trade, and conflict among different tribes, and to explore the emergent behaviors that arise from these interactions.

## Usage
**Install Dependencies**: Ensure that you have Python and required packages installed. You can install the necessary packages using `pip`:

    ```sh
    pip install -r requirements.txt
    ```

*All* files assume you are running them from the main directory, do not try to run e.g. the scripts from inside the scripts directory. 

### Server

**Run the Server**: Start the server by executing:
    ```sh
    python scripts/server.py
    ```

### Experimenting

To run experiments with different configurations, you can use a TOML file to specify the parameters. Here’s how you can do it:

1. **Create a Configuration File**: Create a TOML file (e.g., `configs/config.toml`) with the desired parameters. Below is an example configuration file. In case you want to create your own function for the generation of a map, do so by defining a function in `scripts/experiment_utils.py`. 

    ```toml
    trials = 10 # The number of trials for each parameter set

    [parameters]
    experiment_name = "example"
    width = 100
    height = 100
    n_tribes = 3
    n_agents = 740
    n_heaps = 3
    vision_radius = 10
    step_count = 500
    alpha = 0.5
    trade_percentage = 0.8
    spice_movement_bias = 0.07
    tribe_movement_bias = 0.15
    spice_grow_threshold = 20
    spice_threshold = 9150

    # parameters to be passed to the generation functions.  
    [parameters.spice_kwargs]
    total_spice = 10000
    cov_range = [8, 20]

    [parameters.river_kwargs]
    cov_range = [10, 100]
    total_spice = 1000

    [parameters.location_kwargs]
    cov_range = [8, 20]
    total_spice = 20000

    [functions] # functions defined in 'scripts/experiment_utils.py'
    spice_generator = "gen_central_spice_heap"
    river_generator = "no_river" 
    location_generator = "tribe_locations_naturally_distributed"

    [ranges]
    n_heaps = [1, 10, 1]

    ```

2. **Run the Experiment**: Execute the script with the configuration file to run the model with different parameter settings in parallel. Use the following command:
    ```sh
    python scripts/experiment.py configs/your_config.toml
    ```

### Sensitivity analysis 

You can run `scripts/global_sa_experiment.py`, this will rerun with our exact set-up. It wil take about a day to finish on a computer with 16-cores and 32 threads.
