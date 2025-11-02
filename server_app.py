import json
import time
from logging import INFO
from typing import Dict
import numpy as np
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Result
from torch.utils.data import DataLoader
from task import Net, create_run_dir, get_transforms, test
from strategy import (
    CustomFedAdam, CustomFedAvgM, CustomFedMedian, CustomFedProx,
    CustomFedTrimmedAvg, CustomFedYogi, CustomMultiKrum, CustomDifferentialPrivacyServerSideFixedClipping
)
 
app = ServerApp()

# Strategy Factory
def get_strategy(strategy_name: str, config: Dict, total_nodes: int):
    fraction_train = config["fraction-train"]
    fraction_eval = config["fraction-evaluate"]

    # Common parameters shared across all strategies
    common_params = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_eval,
        # Ensure min_train_nodes is at least 1, even if fraction_train is low
        "min_train_nodes": max(1, int(total_nodes * fraction_train)),
        "min_evaluate_nodes": max(1, int(total_nodes * fraction_eval)),
        "min_available_nodes": total_nodes,
    }

    # Mapping of strategy names to their classes and specific parameters
    strategy_mapping = {
        "fedprox": (CustomFedProx, {"proximal_mu": config["proximal-mu"]}),
        "multikrum": (
            CustomMultiKrum,
            {
                "num_malicious_nodes": config.get("num-malicious-nodes", 0),
                "num_nodes_to_select": config.get("num_nodes_to_select", 1),
            },
        ),
        "trimmed_avg": (CustomFedTrimmedAvg, {"beta": config.get("beta", 0.2)}),
        "median": (CustomFedMedian, {}),
        "fedadam": (
            CustomFedAdam,
            {
                "eta": config.get("eta", 1e-1),
                "eta_l": config.get("eta_l", 1e-1),
                "beta_1": config.get("beta_1", 0.9),
                "beta_2": config.get("beta_2", 0.999),
                "tau": config.get("tau", 1e-3),
            },
        ),
        "fedyogi": (
            CustomFedYogi,
            {
                "eta": config.get("eta", 1e-1),
                "eta_l": config.get("eta_l", 1e-1),
                "beta_1": config.get("beta_1", 0.9),
                "beta_2": config.get("beta_2", 0.999),
                "tau": config.get("tau", 1e-3),
            },
        ),
        "fedavgm": (
            CustomFedAvgM,
            {
                "server_learning_rate": config.get("server_learning_rate", 1.0),
                "server_momentum": config.get("server_momentum", 0.0),
            },
        ),
    }

    strategy_class, specific_params = strategy_mapping[strategy_name]

    return strategy_class(**common_params, **specific_params)

# Main Server Application
@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load configuration from the JSON file created by the experiment runner
    with open("current_run_config.json", "r") as f:
        config = json.load(f)

    total_nodes = config["total-nodes"]
    log(INFO, "Total nodes available for simulation: %s", total_nodes)

    # Initialize the global model
    global_model = Net()
    if config.get("personalization", False):
        # For FedPer, the global model consists only of the base layers
        arrays = ArrayRecord(
            {k: v for k, v in global_model.state_dict().items() if not k.startswith("fc2.")}
        )
    else:
        # For other strategies, the full model is used
        arrays = ArrayRecord(global_model.state_dict())

    # Instantiate the strategy for this run
    base_strategy = get_strategy(config["strategy-name"], config, total_nodes)

    # Wrap strategy with Differential Privacy if enabled
    if config.get("use_dp", False):
        num_sampled_clients = int(total_nodes * config["fraction-train"])
        
        noise_multiplier = config.get("dp_noise_multiplier")
        clipping_norm = config.get("dp_clipping_norm") 
        
        strategy = CustomDifferentialPrivacyServerSideFixedClipping(
            base_strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_sampled_clients
        )
    else:
        strategy = base_strategy

    # Setup the output directory for saving results
    save_path, run_dir = create_run_dir()
    strategy.set_save_path_and_run_dir(
        path=save_path,
        run_dir=run_dir,
        num_rounds_planned=config["num-server-rounds"],
        config=config,
    )

    log(INFO, "Strategy initialized: %s", strategy.__class__.__name__)

    # Setup centralized server-side evaluation if not a personalization scenario
    evaluate_fn = None
    if not config.get("personalization", False):
        log(INFO, "Setting up centralized server-side evaluation.")
        
        # Get the evaluation transforms from task.py
        eval_transforms = get_transforms(config["dataset-name"], is_train=False)

        # Define a function to apply transforms to a batch of data
        def apply_transforms(batch):
            batch["image"] = [eval_transforms(img) for img in batch["image"]]
            return batch

        # Load the central test set and apply the transform function
        test_set = load_dataset(config["dataset-name"])["test"]
        test_set = test_set.with_transform(apply_transforms)

        # Create the DataLoader
        test_loader = DataLoader(test_set, batch_size=config["batch-size"])

        # Get the server-side evaluation function
        evaluate_fn = get_global_evaluate_fn(
            device=config["server-device"], test_loader=test_loader
        )
    else:
        log(INFO, "Server-side evaluation is DISABLED for personalization (FedPer) scenario.")

    # FL Loop
    t_start = time.time()
    result = Result(arrays=arrays)

    # Initial server-side evaluation (Round 0)
    if evaluate_fn:
        res = evaluate_fn(0, arrays)
        if res:
            result.evaluate_metrics_serverapp[0] = res
            strategy.save_metrics_and_log(current_round=0, result=result)

    for current_round in range(1, config["num-server-rounds"] + 1):
        log(INFO, "\n[ROUND %s/%s]", current_round, config["num-server-rounds"])

        # Pass the full configuration to clients in each round
        client_config = ConfigRecord(config)

        # Train on clients
        train_replies = grid.send_and_receive(
            messages=strategy.configure_train(
                server_round=current_round,
                arrays=arrays,
                config=client_config,
                grid=grid,
            ),
            timeout=config["client-timeout"],
        )
        agg_arrays, agg_train_metrics = strategy.aggregate_train(
            current_round, train_replies
        )

        if agg_arrays:
            arrays = agg_arrays
            result.arrays = agg_arrays
        if agg_train_metrics:
            result.train_metrics_clientapp[current_round] = agg_train_metrics

        # 2. Evaluate on clients (Client-Side Evaluation)
        evaluate_replies = grid.send_and_receive(
            messages=strategy.configure_evaluate(
                server_round=current_round,
                arrays=arrays,
                config=client_config,
                grid=grid,
            ),
            timeout=config["client-timeout"],
        )
        agg_evaluate_metrics = strategy.aggregate_evaluate(
            current_round, evaluate_replies
        )

        if agg_evaluate_metrics:
            result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

        # Evaluate on server (Centralized Evaluation)
        if evaluate_fn:
            res = evaluate_fn(current_round, arrays)
            if res:
                result.evaluate_metrics_serverapp[current_round] = res

        # Log and save metrics for the current round
        strategy.save_metrics_and_log(current_round=current_round, result=result)

    log(INFO, "\nStrategy execution finished in %.2fs", time.time() - t_start)

# Server-Side Evaluation Function
def get_global_evaluate_fn(device: str, test_loader: DataLoader):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = Net()
        # Load the full model state for evaluation
        net.load_state_dict(arrays.to_torch_state_dict(), strict=False)
        net.to(device)

        loss, accuracy, cm = test(net, test_loader, device)

        # Calculate macro F1-score from the aggregated confusion matrix
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        # Suppress division by zero warnings for classes with no predictions/labels
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1)  # Replace NaN with 0
        macro_f1 = float(np.mean(f1))
        num_examples = len(test_loader.dataset)

        return MetricRecord(
            {
                "accuracy": accuracy,
                "loss": loss,
                "f1_score": macro_f1,
                "num_examples": num_examples,
            }
        )

    return global_evaluate