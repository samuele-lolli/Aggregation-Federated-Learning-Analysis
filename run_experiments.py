import subprocess
import json
from typing import Dict, Any
import time
from run_configuration import scenarios
import os

# Ensure Ray logs are not deduplicated to avoid losing logs between runs
#os.environ["RAY_DEDUP_LOGS"] = "0"

# Profile to use for the Flower simulation (defined in pyproject.toml)
FEDERATION_PROFILE = "local-sim-gpu"  # or "local-sim" to use only CPU

# Number of times each scenario is repeated
NUM_REPETITIONS = 15 

# Base configuration common to all runs
BASE_CONFIG: Dict[str, Any] = {
    # FL parameters
    "total-nodes": 10,
    "num-server-rounds": 40,
    "fraction-train": 1.0,
    "fraction-evaluate": 1.0,
    "client-timeout": 3600,
    "server-device": "cpu",

    # Local training parameters
    "local-epochs": 1,
    "batch-size": 32,
    "lr": 0.01,
    "momentum": 0.9,

    # Data partitioning parameters
    "partitioner-name": "iid",  # Options: 'iid', 'dirichlet'
    "dirichlet-alpha": 0.0,     # Only used if partitioner-name is 'dirichlet'
    "dataset-name": "zalando-datasets/fashion_mnist",
    "val-split-percentage": 0.2,
    "seed": 42, # This will be the starting seed for the first repetition

    # Strategy parameters
    "strategy-name": "fedavg", 
    "personalization": False, # If True, clients will retain their classification head between rounds

    # FedProx 
    "proximal-mu": 0.0, # If 0.0, equivalent to FedAvg

    # FedTrimmedAvg
    "beta": 0.2,

    # MultiKrum
    "num_nodes_to_select": 1, # If 1, equivalent to Krum
    "num-malicious-nodes": 0,

    # FedAdam / FedYogi
    "eta": 1e-1,
    "eta_l": 1e-1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "tau": 1e-3,

    # FedAvgM
    "server_learning_rate": 1.0,
    "server_momentum": 0.0,

    # Backdoor Attack parameters
    "attack_target_class": 1,          
    "attack_injection_rate": 0.1, # Percentage of training data to poison per malicious client
    "attack_trigger_value": 3.0, # Pixel intensity for the trigger
}

# Runs a single Flower simulation with the given configuration
def run_experiment(config: Dict[str, Any], run_name: str) -> bool:  
    print("-" * 50)
    print(f"RUNNING: {run_name}")
    print("-" * 50)

    # Ensure dirichlet-alpha is set if not explicitly present in the scenario (for IID consistency)
    if "dirichlet-alpha" not in config:
        config["dirichlet-alpha"] = 0.0 # A dummy value, overwritten if set in scenario

    # Write the complete configuration for this run to a temporary JSON file
    config_path = "current_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Construct the command to run the Flower simulation
    command = ["flwr", "run", ".", FEDERATION_PROFILE]
    try:
        # Execute the command as a subprocess
        process = subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        print(f"Error during experiment run '{run_name}': {e}")
        return False
    finally:
        # Force Ray shutdown to clean up resources
        print("Forcing Ray shutdown to clean up resources...")
        subprocess.run(["ray", "stop", "--force"])
        print("Ray shutdown complete.")
    return True

# Main execution loop for all scenarios and repetitions
if __name__ == "__main__":

    total_runs = len(scenarios) * NUM_REPETITIONS
    run_counter = 0
    print(f"Starting execution of {len(scenarios)} scenarios, {NUM_REPETITIONS} repetitions each. Total runs: {total_runs}")
    
    for scenario_definition in scenarios:
        scenario_name = scenario_definition.get("scenario_name", "Unnamed Scenario")
        print(f"\n===== SCENARIO: {scenario_name} =====")

        for i in range(NUM_REPETITIONS):
            run_counter += 1
            
            # Create a full copy of the base config for this specific run
            exp_config = BASE_CONFIG.copy()

            # Override with scenario-specific parameters
            exp_config.update(scenario_definition)
            
            # The seed is incremented for each repetition to ensure independent trials.
            # This affects data partitioning and data shuffling.
            current_seed = BASE_CONFIG["seed"] + i
            exp_config["seed"] = current_seed

            print(f"\n--- Running repetition {i+1}/{NUM_REPETITIONS} (Total: {run_counter}/{total_runs}) ---")
            
            if not run_experiment(exp_config, scenario_name):
                print("Aborting experiment script due to error or user interruption.")
                exit()
            
            print(f"Repetition {i+1} finished. Waiting 3 seconds before next run...")
            time.sleep(3)
        
    print("\nAll experiments finished successfully.")