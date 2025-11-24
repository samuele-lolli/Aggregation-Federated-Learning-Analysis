import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback
from typing import List, Dict, Any

TARGET_SCENARIOS = [
    # Full Participation
    "FedAvg_IID_personalized",
    "FedAvg_nonIID_a0_5_personalized",
    "FedAvg_nonIID_a0_1_personalized",
    "FedAvg_nonIID_a0_03_personalized",
    
    # Partial Participation (80%)
    "FedAvg_IID_personalized_frac08",
    "FedAvg_nonIID_a0_5_personalized_frac08",
    "FedAvg_nonIID_a0_1_personalized_frac08",
    "FedAvg_nonIID_a0_03_personalized_frac08",

    # Partial participation (70%)
    "FedAvg_IID_personalized_frac07",
    "FedAvg_nonIID_a0_5_personalized_frac07",
    "FedAvg_nonIID_a0_1_personalized_frac07",
    "FedAvg_nonIID_a0_03_personalized_frac07"
]

OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR = Path("./plots/overfitting_checks")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_loss_data(outputs_dir: Path, target_scenario: str) -> pd.DataFrame:
    all_data: List[Dict[str, Any]] = []
    
    run_dirs = [d.parent for d in outputs_dir.rglob("results.json")]
    
    found_runs = 0
    for run_dir in run_dirs:
        json_path = run_dir / "results.json"
        run_id = run_dir.name
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            config = data.get("config", {})
            scenario = config.get("scenario_name")

            if scenario == target_scenario:
                found_runs += 1
                for round_data in data.get("rounds", []):
                    round_num = round_data.get("round")
                    train_loss = round_data.get("training", {}).get("loss")
                    val_loss = round_data.get("client_evaluation", {}).get("loss")

                    if round_num is not None and train_loss is not None and val_loss is not None:
                        all_data.append({
                            "run_id": run_id,
                            "round": int(round_num),
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss)
                        })
        except Exception as e:
            traceback.print_exc()

    return pd.DataFrame(all_data)

def plot_overfitting_check(df: pd.DataFrame, title: str, filename: str) -> None:
    if df.empty:
        return

    agg_df = df.groupby("round").agg(
        train_loss_mean=('train_loss', 'mean'),
        train_loss_std=('train_loss', 'std'),
        val_loss_mean=('val_loss', 'mean'),
        val_loss_std=('val_loss', 'std')
    ).fillna(0)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    ax.plot(agg_df.index, agg_df['train_loss_mean'], label="Training Loss (Media)", color="#1f77b4", linewidth=2)
    ax.fill_between(
        agg_df.index,
        (agg_df['train_loss_mean'] - agg_df['train_loss_std']).clip(lower=0),
        agg_df['train_loss_mean'] + agg_df['train_loss_std'],
        color="#1f77b4", alpha=0.1, label="Training Loss (Std Dev)"
    )

    ax.plot(agg_df.index, agg_df['val_loss_mean'], label="Validation Loss (Media)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.fill_between(
        agg_df.index,
        (agg_df['val_loss_mean'] - agg_df['val_loss_std']).clip(lower=0),
        agg_df['val_loss_mean'] + agg_df['val_loss_std'],
        color="#ff7f0e", alpha=0.1, label="Validation Loss (Std Dev)"
    )

    ax.set_title(f"Training vs Validation Loss\n{title}", fontsize=18)
    ax.set_xlabel("Federated Learning Round", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0) 
    
    max_round = int(agg_df.index.max())
    tick_step = 2 if max_round <= 40 else 5
    xticks = np.arange(0, max_round + 1, tick_step)
    ax.set_xticks(xticks)
    ax.set_xlim(-0.5, max_round + 0.5)

    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plots saved in: {plot_path}")

if __name__ == "__main__":
    for scenario_name in TARGET_SCENARIOS:
        data = load_loss_data(OUTPUTS_DIR, scenario_name)
        num_runs = data['run_id'].nunique()
        plot_filename = f"overfit_check_{scenario_name}.pdf"
        plot_overfitting_check(data, title=scenario_name, filename=plot_filename)
    
    print("\n✅ Process completed.")