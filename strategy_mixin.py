import json
import time
from logging import INFO
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
import numpy as np
from flwr.common import log
from flwr.serverapp.strategy import Result
from flwr.app import ArrayRecord, Message, MetricRecord

# StrategyMixin provides common functionality for all strategies
# including saving metrics, logging, and custom aggregation logic.
class StrategyMixin:
    num_rounds_planned: Optional[int] = None
    save_path: Path
    run_dir: str
    run_config: Dict

    # Sets the output paths, total rounds, and run configuration
    def set_save_path_and_run_dir(
        self, path: Path, run_dir: str, num_rounds_planned: int, config: Dict
    ) -> None:
        
        self.save_path = path
        self.run_dir = run_dir
        self.num_rounds_planned = num_rounds_planned
        self.run_config = config

    # Saves metrics in JSON, handles summary calculation, and prints console logs     
    def save_metrics_and_log(self, current_round: int, result: Result) -> None:
        
        # Helper to round numeric values for clean output
        def round_metric(value: Any, decimals: int) -> Any:
            if isinstance(value, (int, float, np.floating)):
                return round(float(value), decimals)
            return value

        round_metrics: Dict[str, Any] = {
            "round": current_round,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training": {},
            "client_evaluation": {},
            "server_evaluation": {}
        }
        
        # Populate metrics
        if last_train_metrics := dict(result.train_metrics_clientapp.get(current_round, {})):
            round_metrics["training"] = {
                "loss": round_metric(last_train_metrics.get("train_loss", 0), 6),
                "num_examples": last_train_metrics.get("num-examples", 0),
            }
        
        if last_eval_client := dict(result.evaluate_metrics_clientapp.get(current_round, {})):
            round_metrics["client_evaluation"] = {
                "loss": round_metric(last_eval_client.get("eval_loss", 0), 6),
                "accuracy": round_metric(last_eval_client.get("eval_acc", 0), 4),
                "f1_score": round_metric(last_eval_client.get("f1_score", 0), 4), 
                "backdoor_asr": round_metric(last_eval_client.get("backdoor_asr", 0), 4),
                "num_examples": last_eval_client.get("num-examples", 0)
            }
        
        if last_eval_server := dict(result.evaluate_metrics_serverapp.get(current_round, {})):
            round_metrics["server_evaluation"] = {
                "loss": round_metric(last_eval_server.get("loss", 0), 6),
                "accuracy": round_metric(last_eval_server.get("accuracy", 0), 4),
                "f1_score": round_metric(last_eval_server.get("f1_score", 0), 4),
                "num_examples": last_eval_server.get("num_examples", 0)
            }
        
        # JSON Saving Logic
        results_file = self.save_path / "results.json"
        results_data = {"config": {}, "rounds": []}
        
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as fp:
                try:
                    results_data = json.load(fp)
                except json.JSONDecodeError:
                    log(INFO, "Warning: Could not decode existing results.json, creating a new one.")
                    
        if not results_data.get("config"):
            results_data["config"] = self.run_config
        
        rounds = results_data.get("rounds", [])
        # Find if metrics for the current round already exist
        existing_round_index = next((i for i, r in enumerate(rounds) if r.get("round") == current_round), -1)
        if existing_round_index != -1:
            # Overwrite existing round metrics
            rounds[existing_round_index] = round_metrics
        else:
            # Append new round metrics
            rounds.append(round_metrics)
        
        results_data["rounds"] = sorted(rounds, key=lambda x: x["round"])
        # Calculate and update summary statistics
        results_data["summary"] = self._calculate_summary_statistics(results_data["rounds"])
        
        try:
            with open(results_file, "w", encoding="utf-8") as fp:
                json.dump(results_data, fp, indent=2, ensure_ascii=False)
        except IOError as e:
            log(INFO, "Error saving results.json: %s", e)
        
        self._print_round_metrics(current_round, round_metrics)

    # Calculate summary statistics from all completed rounds
    def _calculate_summary_statistics(self, rounds: List[Dict]) -> Dict:      
        if not rounds: 
            return {}
        
        client_accuracies = [r.get("client_evaluation", {}).get("accuracy", 0) for r in rounds]
        server_accuracies = [r.get("server_evaluation", {}).get("accuracy", 0) for r in rounds]
        client_f1_scores = [r.get("client_evaluation", {}).get("f1_score", 0) for r in rounds]
        server_f1_scores = [r.get("server_evaluation", {}).get("f1_score", 0) for r in rounds]
        client_asr_scores = [r.get("client_evaluation", {}).get("backdoor_asr", 0) for r in rounds]

        return {
            "total_rounds_completed": len(rounds),
            "best_client_accuracy": max(client_accuracies) if client_accuracies else 0,
            "best_server_accuracy": max(server_accuracies) if server_accuracies else 0,
            "best_client_f1_score": max(client_f1_scores) if client_f1_scores else 0,
            "best_server_f1_score": max(server_f1_scores) if server_f1_scores else 0,
            # Final metrics are those from the last completed round
            "final_client_accuracy": client_accuracies[-1] if client_accuracies else 0,
            "final_server_accuracy": server_accuracies[-1] if server_accuracies else 0,
            "final_client_f1_score": client_f1_scores[-1] if client_f1_scores else 0,
            "final_server_f1_score": server_f1_scores[-1] if server_f1_scores else 0,
            "final_client_asr": client_asr_scores[-1] if client_asr_scores else 0,
        }

    # Prints formatted metrics for the current round to the console
    def _print_round_metrics(self, current_round: int, round_metrics: Dict) -> None: 
        log(INFO, f"--- ROUND {current_round} METRICS SUMMARY ---")
        
        if train_metrics := round_metrics.get("training"):
            log(INFO, "  [Training] Loss: %.4f, Examples: %d", 
                train_metrics.get("loss", 0), train_metrics.get("num_examples", 0))
        
        if client_eval := round_metrics.get("client_evaluation"):
            log(INFO, "  [Client Eval Agg] Acc: %.3f, F1: %.3f, Loss: %.4f, Examples: %d",
                client_eval.get("accuracy", 0), client_eval.get("f1_score", 0), 
                client_eval.get("loss", 0), client_eval.get("num_examples", 0))
            
            if "backdoor_asr" in client_eval and client_eval["backdoor_asr"] > 0:
                 log(INFO, "  [Client Eval Agg] Backdoor ASR: %.3f", client_eval.get("backdoor_asr"))
        
        if server_eval := round_metrics.get("server_evaluation"):
            log(INFO, "  [Server Eval Central] Acc: %.3f, F1: %.3f, Loss: %.4f, Examples: %d",
                server_eval.get("accuracy", 0), server_eval.get("f1_score", 0), 
                server_eval.get("loss", 0), server_eval.get("num_examples", 0))
        log(INFO, "--------------------------------------------")

    # Aggregate model parameters using the base strategy and then aggregate training metrics.
    # This approach is compatible with any base strategy (e.g., Krum, FedAvg) because
    # it delegates the core parameter aggregation to `super()` and only handles
    # the metric aggregation part itself.
    def aggregate_train(self, server_round: int, replies: List[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        
        if not replies:
            return None, None
        
        res = super().aggregate_train(server_round, replies)
        
        if res is None:
            log(INFO, "Warning: Parameter aggregation in super() failed for round %d", server_round)
            return None, None

        aggregated_arrays, _ = res 

        if aggregated_arrays is None:
            log(INFO, "Warning: Parameter aggregation failed (arrays are None) for round %d", server_round)
            return None, None 
        
        train_losses = []
        num_examples_total = 0
        for msg in replies:
            if "metrics" in msg.content:
                metrics = dict(msg.content["metrics"])
                num_examples = metrics.get("num-examples", 0)
                loss = metrics.get("train_loss", 0.0)
                train_losses.append((num_examples, loss))
                num_examples_total += num_examples
        
        # Calculate weighted average loss
        weighted_avg_loss = sum(
            num * loss for num, loss in train_losses
        ) / num_examples_total if num_examples_total > 0 else 0.0
        
        aggregated_metrics = MetricRecord({
            "train_loss": weighted_avg_loss,
            "num-examples": num_examples_total 
        })

        return aggregated_arrays, aggregated_metrics

    # Aggregate evaluation metrics, including average accuracy, loss, and macro F1-score from aggregated confusion matrices.
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: List[Message],
    ) -> Optional[MetricRecord]:
        
        if not replies:
            return None

        num_examples_total = 0
        weighted_losses = []
        weighted_accuracies = []
        confusion_matrices = []
        weighted_asr_scores = []

        for msg in replies:
            if "metrics" in msg.content:
                metrics = dict(msg.content["metrics"])
                num_examples = metrics.get("num-examples", 0)
                
                num_examples_total += num_examples
                weighted_losses.append((num_examples, metrics.get("eval_loss", 0.0)))
                weighted_accuracies.append((num_examples, metrics.get("eval_acc", 0.0)))
                weighted_asr_scores.append((num_examples, metrics.get("backdoor_asr", 0.0)))
                
                # Extract and collect confusion matrix for Macro F1 calculation
                if cm_record := msg.content.get("confusion_matrix"):
                    cm_array = cm_record.to_numpy_ndarrays()[0]
                    confusion_matrices.append(cm_array)

        # Calculate weighted averages for loss, accuracy and ASR
        avg_loss = (
            sum(num * loss for num, loss in weighted_losses) / num_examples_total
            if num_examples_total > 0 else 0.0
        )
        avg_accuracy = (
            sum(num * acc for num, acc in weighted_accuracies) / num_examples_total
            if num_examples_total > 0 else 0.0
        )

        avg_asr = (
            sum(num * asr for num, asr in weighted_asr_scores) / num_examples_total
            if num_examples_total > 0 else 0.0
        )

        # Calculate Macro F1-Score from the aggregated confusion matrix
        macro_f1 = 0.0
        if confusion_matrices and all(cm.shape == confusion_matrices[0].shape for cm in confusion_matrices):
            aggregated_cm = np.sum(confusion_matrices, axis=0)
            tp = np.diag(aggregated_cm)
            fp = np.sum(aggregated_cm, axis=0) - tp
            fn = np.sum(aggregated_cm, axis=1) - tp
            
            # Suppress division by zero warnings when calculating precision/recall
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                
            f1 = np.nan_to_num(f1) # Replace NaNs (from 0/0) with 0
            macro_f1 = float(np.mean(f1))
        
        return MetricRecord({
            "eval_loss": avg_loss,
            "eval_acc": avg_accuracy,
            "f1_score": macro_f1,
            "backdoor_asr": avg_asr,
            "num-examples": num_examples_total
        })