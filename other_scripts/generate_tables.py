import json
from pathlib import Path
import pandas as pd
import numpy as np
import traceback 

# Constants
OUTPUTS_DIR = Path("./outputs")
TABLES_DIR = Path("./tables")
TABLES_DIR.mkdir(exist_ok=True)

TARGET_ROUND = 40

STRATEGY_ORDER = ["FedAvg", "FedProx", "FedAvgM", "FedAdam", "FedYogi", "FedMedian", "TrimmedAvg", "MultiKrum"]

DEFAULT_METRICS = [f"round{TARGET_ROUND}_server_accuracy", f"round{TARGET_ROUND}_server_f1_score", f"round{TARGET_ROUND}_client_accuracy", f"round{TARGET_ROUND}_client_f1_score"]
BACKDOOR_ATTACK_METRICS = [f"round{TARGET_ROUND}_server_accuracy", f"round{TARGET_ROUND}_server_f1_score", f"round{TARGET_ROUND}_client_accuracy", f"round{TARGET_ROUND}_client_f1_score", f"round{TARGET_ROUND}_client_asr"]
PERSONALIZATION_METRICS = [f"round{TARGET_ROUND}_client_accuracy", f"round{TARGET_ROUND}_client_f1_score"]

# Table Definitions
TABLE_DEFINITIONS = [
    # ==========================================================
    # --- Full Client Participation (fraction=1.0, No Attack) ---
    # ==========================================================
    {
        "title": f"Table 1: Performance at Round {TARGET_ROUND} in IID Scenarios (Full Participation)",
        "filename": "table_01_iid_full_r40.md",
        "filters": {"partitioner-name": "iid", "fraction-train": 1.0, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 2: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.5, Full Participation)",
        "filename": "table_02_noniid_a05_full_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 1.0, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 3: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.1, Full Participation)",
        "filename": "table_03_noniid_a01_full_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 1.0, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 4: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.03, Full Participation)",
        "filename": "table_04_noniid_a003_full_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 1.0, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ====================================================================
    # --- Partial Client Participation (fraction=0.8, No Attack) ---
    # ====================================================================
    {
        "title": f"Table 5: Performance at Round {TARGET_ROUND} in IID Scenarios (Partial Participation, fraction=0.8)",
        "filename": "table_05_iid_frac08_r40.md",
        "filters": {"partitioner-name": "iid", "fraction-train": 0.8, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 6: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.5, Partial Participation, fraction=0.8)",
        "filename": "table_06_noniid_a05_frac08_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.8, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 7: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.1, Partial Participation, fraction=0.8)",
        "filename": "table_07_noniid_a01_frac08_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.8, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 8: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.03, Partial Participation, fraction=0.8)",
        "filename": "table_08_noniid_a003_frac08_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.8, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ====================================================================
    # --- Partial Client Participation (fraction=0.7, No Attack) ---
    # ====================================================================
    {
        "title": f"Table 9: Performance at Round {TARGET_ROUND} in IID Scenarios (Partial Participation, fraction=0.7)",
        "filename": "table_09_iid_frac07_r40.md",
        "filters": {"partitioner-name": "iid", "fraction-train": 0.7, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
     {
        "title": f"Table 10: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.5, Partial Participation, fraction=0.7)",
        "filename": "table_10_noniid_a05_frac07_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.7, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 11: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.1, Partial Participation, fraction=0.7)",
        "filename": "table_11_noniid_a01_frac07_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.7, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 12: Performance at Round {TARGET_ROUND} in Non-IID Scenarios (α=0.03, Partial Participation, fraction=0.7)",
        "filename": "table_12_noniid_a003_frac07_r40.md",
        "filters": {"partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.7, "personalization": False, "attack_name": "none"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ==========================================================
    # --- Federated Personalization (FedPer, No Attack) ---
    # ==========================================================
    {
        "title": f"Table 13: Performance of Federated Personalization (FedPer) at Round {TARGET_ROUND}",
        "filename": "table_13_personalization_r40.md",
        "filters": {"personalization": True, "attack_name": "none"},
        "index_col": "scenario_name",
        "metrics": PERSONALIZATION_METRICS,
    },

    # ==========================================================
    # --- Backdoor Attack Scenarios (1 Malicious Client) ---
    # ==========================================================
    {
        "title": f"Table 14: Performance under Backdoor Attack at Round {TARGET_ROUND} (IID, 1 Malicious Client)",
        "filename": "table_14_attack_backdoor_iid_1_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "iid", "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },
    {
        "title": f"Table 15: Performance under Backdoor Attack at Round {TARGET_ROUND} (Non-IID α=0.5, 1 Malicious Client)",
        "filename": "table_15_attack_backdoor_noniid_a05_1_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },
    {
        "title": f"Table 16: Performance under Backdoor Attack at Round {TARGET_ROUND} (Non-IID α=0.1, 1 Malicious Client)",
        "filename": "table_16_attack_backdoor_noniid_a01_1_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },

    # ==========================================================
    # --- Backdoor Attack Scenarios (2 Malicious Clients) ---
    # ==========================================================
    {
        "title": f"Table 17: Performance under Backdoor Attack at Round {TARGET_ROUND} (IID, 2 Malicious Clients)",
        "filename": "table_17_attack_backdoor_iid_2_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "iid", "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },
    {
        "title": f"Table 18: Performance under Backdoor Attack at Round {TARGET_ROUND} (Non-IID α=0.5, 2 Malicious Clients)",
        "filename": "table_18_attack_backdoor_noniid_a05_2_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },
    {
        "title": f"Table 19: Performance under Backdoor Attack at Round {TARGET_ROUND} (Non-IID α=0.1, 2 Malicious Clients)",
        "filename": "table_19_attack_backdoor_noniid_a01_2_malicious_r40.md",
        "filters": {"attack_name": "backdoor", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": BACKDOOR_ATTACK_METRICS,
    },

    # ==========================================================
    # --- Byzantine Attack Scenarios ---
    # ==========================================================
    {
        "title": f"Table 20: Performance under Byzantine Attack at Round {TARGET_ROUND} (IID)",
        "filename": "table_20_attack_byzantine_iid_r40.md",
        "filters": {"attack_name": "byzantine", "partitioner-name": "iid"},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 21: Performance under Byzantine Attack at Round {TARGET_ROUND} (Non-IID α=0.5)",
        "filename": "table_21_attack_byzantine_noniid_a05_r40.md",
        "filters": {"attack_name": "byzantine", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 22: Performance under Byzantine Attack at Round {TARGET_ROUND} (Non-IID α=0.1)",
        "filename": "table_22_attack_byzantine_noniid_a01_r40.md",
        "filters": {"attack_name": "byzantine", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ==========================================================
    # --- Label Flipping Attack Scenarios (1 Malicious Client) ---
    # ==========================================================
     {
        "title": f"Table 23: Performance under Label Flipping Attack at Round {TARGET_ROUND} (IID, 1 Malicious Client)",
        "filename": "table_23_attack_labelflip_iid_1_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "iid", "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 24: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.5, 1 Malicious Client)",
        "filename": "table_24_attack_labelflip_noniid_a05_1_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 25: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.1, 1 Malicious Client)",
        "filename": "table_25_attack_labelflip_noniid_a01_1_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "num_malicious_clients": 1},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ==========================================================
    # --- Label Flipping Attack Scenarios (2 Malicious Clients) ---
    # ==========================================================
     {
        "title": f"Table 26: Performance under Label Flipping Attack at Round {TARGET_ROUND} (IID, 2 Malicious Clients)",
        "filename": "table_26_attack_labelflip_iid_2_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "iid", "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 27: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.5, 2 Malicious Clients)",
        "filename": "table_27_attack_labelflip_noniid_a05_2_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 28: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.1, 2 Malicious Clients)",
        "filename": "table_28_attack_labelflip_noniid_a01_2_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "num_malicious_clients": 2},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },

    # ==========================================================
    # --- Label Flipping Attack Scenarios (3 Malicious Clients) ---
    # ==========================================================
     {
        "title": f"Table 29: Performance under Label Flipping Attack at Round {TARGET_ROUND} (IID, 3 Malicious Clients)",
        "filename": "table_29_attack_labelflip_iid_3_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "iid", "num_malicious_clients": 3},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 30: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.5, 3 Malicious Clients)",
        "filename": "table_30_attack_labelflip_noniid_a05_3_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "num_malicious_clients": 3},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
    {
        "title": f"Table 31: Performance under Label Flipping Attack at Round {TARGET_ROUND} (Non-IID α=0.1, 3 Malicious Clients)",
        "filename": "table_31_attack_labelflip_noniid_a01_3_malicious_r40.md",
        "filters": {"attack_name": "label_flipping", "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "num_malicious_clients": 3},
        "index_col": "strategy-name",
        "metrics": DEFAULT_METRICS,
    },
]

# Load data from all results.json files and extract round 40 metrics
def load_round40_data(outputs_dir: Path, target_round: int) -> pd.DataFrame:
    all_round_data = []
    processed_runs = set()

    for json_path in outputs_dir.rglob("results.json"):
        run_id = json_path.parent.name
        if run_id in processed_runs:
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            config = data.get("config", {})
            rounds_list = data.get("rounds", [])

            if "scenario_name" not in config or not rounds_list:
                # print(f"Skipping {run_id}: Missing scenario_name or rounds data.")
                processed_runs.add(run_id)
                continue

            # Find data for the target round ---
            target_round_data = None
            for round_data in rounds_list:
                if round_data.get("round") == target_round:
                    target_round_data = round_data
                    break # Found the target round

            # Prepare the record for this run
            record = config.copy()
            record["run_id"] = run_id

            # Add defaults and clean config
            record.setdefault("attack_name", "none")
            if record["attack_name"] is None or record["attack_name"] == "": record["attack_name"] = "none"
            record.setdefault("personalization", False)
            if record["personalization"] is None: record["personalization"] = False
            record.setdefault("fraction-train", 1.0)
            if record["fraction-train"] is None: record["fraction-train"] = 1.0
            if record.get('partitioner-name') == 'iid':
                 record.setdefault("dirichlet-alpha", -1.0)
            elif 'dirichlet-alpha' not in record or record["dirichlet-alpha"] is None:
                 record["dirichlet-alpha"] = np.nan
            record.setdefault("strategy-name", "unknown")
            record.setdefault("proximal-mu", 0.0)

            malicious_ids = record.get("malicious-clients-ids", [])
            if isinstance(malicious_ids, list):
                record["num_malicious_clients"] = len(malicious_ids)
            else:
                record["num_malicious_clients"] = 0

            # Add metrics from the target round (or NaN if missing)
            metric_prefix = f"round{target_round}_"
            expected_metric_keys = {
                "server_accuracy": "server_evaluation_accuracy",
                "server_f1_score": "server_evaluation_f1_score",
                "client_accuracy": "client_evaluation_accuracy",
                "client_f1_score": "client_evaluation_f1_score",
                "client_asr": "client_evaluation_backdoor_asr"
            }

            if target_round_data:
                 server_metrics = target_round_data.get("server_evaluation", {})
                 client_metrics = target_round_data.get("client_evaluation", {})
                 record[metric_prefix + "server_accuracy"] = server_metrics.get("accuracy", np.nan)
                 record[metric_prefix + "server_f1_score"] = server_metrics.get("f1_score", np.nan)
                 record[metric_prefix + "client_accuracy"] = client_metrics.get("accuracy", np.nan)
                 record[metric_prefix + "client_f1_score"] = client_metrics.get("f1_score", np.nan)
                 record[metric_prefix + "client_asr"] = client_metrics.get("backdoor_asr", np.nan) 
            else:
                 print(f"   Warning: Round {target_round} data not found for run {run_id}. Metrics will be NaN.")
                 for key in expected_metric_keys:
                      record[metric_prefix + key] = np.nan

            all_round_data.append(record)
            processed_runs.add(run_id)

        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"Skipping corrupted/unreadable file: {json_path} ({type(e).__name__}: {e})")
            processed_runs.add(run_id)

    if not all_round_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_round_data)

    # Data Cleaning and Type Conversion
    print("Performing data cleaning and type conversion...")
    numeric_cols = ["fraction-train", "dirichlet-alpha", "proximal-mu", "num_malicious_clients"] + \
                   [col for col in df.columns if col.startswith(f'round{target_round}_')]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'personalization' in df.columns: df['personalization'] = df['personalization'].fillna(False).astype(bool)
    else: df['personalization'] = False

    string_cols = ["run_id", "scenario_name", "strategy-name", "partitioner-name", "attack_name"]
    for col in string_cols:
         if col in df.columns: df[col] = df[col].fillna('none').astype(str)
         else: df[col] = 'none'

    # Handle FedAvg/FedProx naming
    if 'strategy-name' in df.columns and 'proximal-mu' in df.columns:
        fedprox_mask = (df['strategy-name'] == 'fedprox') | (df['strategy-name'] == 'FedAvg') | (df['strategy-name'] == 'FedProx')
        avg_mask = (df['proximal-mu'] == 0.0) | df['proximal-mu'].isna()
        prox_mask = df['proximal-mu'] > 0.0
        df.loc[fedprox_mask & avg_mask, 'strategy-name'] = 'FedAvg'
        df.loc[fedprox_mask & prox_mask, 'strategy-name'] = 'FedProx'

    # Apply other name mappings
    name_map = {"fedavgm": "FedAvgM", "fedadam": "FedAdam", "fedyogi": "FedYogi",
                "median": "FedMedian", "trimmed_avg": "TrimmedAvg", "multikrum": "MultiKrum"}
    if 'strategy-name' in df.columns: df['strategy-name'] = df['strategy-name'].replace(name_map)

    # Re-apply placeholder for dirichlet-alpha IID AFTER potential NaN filling
    if 'dirichlet-alpha' in df.columns and 'partitioner-name' in df.columns:
         is_iid_mask = df['partitioner-name'] == 'iid'
         df.loc[is_iid_mask, 'dirichlet-alpha'] = df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)

    print(f"Data loading complete. DataFrame shape: {df.shape}")
    return df

def format_metric(mean_series: pd.Series, std_series: pd.Series, decimals=3) -> pd.Series:
    std_series_filled = std_series.fillna(0)
    # Format mean and std, handle NaN in mean_series after formatting
    # If mean is NaN, output 'N/A', otherwise format mean ± std
    formatted = []
    for mean_val, std_val in zip(mean_series, std_series_filled):
         if pd.isna(mean_val):
              formatted.append("N/A")
         else:
              formatted.append(f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}")
    return pd.Series(formatted, index=mean_series.index)


def generate_markdown_table(df: pd.DataFrame, title: str, index_col_name: str) -> str:
    clean_index_name = index_col_name.replace('-', ' ').replace('_', ' ').title()
    if isinstance(df.index, pd.MultiIndex):
         df.index.names = [n.replace('-', ' ').replace('_', ' ').title() for n in df.index.names]
    elif not df.empty:
         df.index.name = clean_index_name

    markdown = f"### {title}\n\n"
    if df.empty:
        markdown += "*No data available for this configuration.*\n\n"
    else:
        ordered_cols = []
        possible_cols_order = ["Server Accuracy", "Server F1 Score", "Client Accuracy", "Client F1 Score", "Client Asr"]
        for col in possible_cols_order:
             if col in df.columns: ordered_cols.append(col)
        ordered_cols += sorted([col for col in df.columns if col not in ordered_cols])
        df_ordered = df[ordered_cols]

        markdown += df_ordered.to_markdown() 
        markdown += "\n\n"
    return markdown

def main():
    print("📊 Starting table generation...")
    master_df = load_round40_data(OUTPUTS_DIR, TARGET_ROUND)

    if master_df.empty:
        print(f"\n❌ Could not find any 'results.json' files with data for round {TARGET_ROUND}.")
        return
    
    for table_def in TABLE_DEFINITIONS:
        title = table_def['title']
        filename = table_def['filename']
        filters = table_def.get('filters', {})
        index_col = table_def['index_col']
        metrics_to_show = table_def['metrics']

        print(f"⚙️ Generating: {title}")

        filtered_df = master_df.copy()
        try:
            for key, value in filters.items():
                if key not in filtered_df.columns:
                    print(f"   ⚠️ Warning: Filter key '{key}' not found. Skipping filter.")
                    continue

                if isinstance(value, list):
                    filter_list = []
                    col_dtype = filtered_df[key].dtype
                    is_numeric_col = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
                    for v in value:
                        if v is None or v == "":
                            if key == 'attack_name': filter_list.append('none')
                            elif key == 'personalization': filter_list.append(False)
                            elif is_numeric_col: filter_list.append(np.nan)
                            else: filter_list.append('none')
                        elif is_numeric_col: filter_list.append(pd.to_numeric(v, errors='coerce'))
                        elif pd.api.types.is_bool_dtype(col_dtype): filter_list.append(bool(v))
                        else: filter_list.append(str(v))

                    has_nan = any(pd.isna(v) for v in filter_list)
                    actual_values = [v for v in filter_list if pd.notna(v)]
                    mask = filtered_df[key].isin(actual_values)
                    if has_nan: mask = mask | filtered_df[key].isna()
                    filtered_df = filtered_df[mask]

                else: 
                    filter_val = value
                    if filter_val is None or filter_val == "":
                        if key == 'attack_name': filter_val = 'none'
                        elif key == 'personalization': filter_val = False
                        elif key in ["fraction-train", "dirichlet-alpha", "proximal-mu", "num_malicious_clients"]: 
                            filter_val = np.nan
                        else: filter_val = 'none'

                    if pd.isna(filter_val) and key in ["dirichlet-alpha"]:
                        filtered_df = filtered_df[filtered_df[key].isna()]
                    elif isinstance(filter_val, bool) and key == 'personalization':
                        filtered_df = filtered_df[filtered_df[key] == filter_val]
                    elif isinstance(filter_val, (int, float)) and key in ["fraction-train", "dirichlet-alpha", "proximal-mu", "num_malicious_clients"]:
                        numeric_col = pd.to_numeric(filtered_df[key], errors='coerce')
                        if isinstance(filter_val, float):
                            filtered_df = filtered_df[np.isclose(numeric_col, filter_val, equal_nan=True)]
                        else:
                            filtered_df = filtered_df[numeric_col == filter_val]
                    else: 
                        filtered_df = filtered_df[filtered_df[key].astype(str) == str(filter_val)]

            if filtered_df.empty:
                print(f"   ❌ No data found after filtering for this table. Skipping.")
                markdown_output = generate_markdown_table(pd.DataFrame(), title, index_col)
            else:
                print(f"   📊 Found {len(filtered_df)} run summaries matching filters (representing {filtered_df['run_id'].nunique()} unique runs).")

                # Aggregation
                missing_metrics = [m for m in metrics_to_show if m not in filtered_df.columns or filtered_df[m].isnull().all()]
                if missing_metrics:
                      # print(f" ⚠️ Warning: Metrics missing or all NaN for aggregation: {missing_metrics}. Using available.")
                      metrics_to_agg = [m for m in metrics_to_show if m in filtered_df.columns and not filtered_df[m].isnull().all()]
                else:
                      metrics_to_agg = metrics_to_show

                if not metrics_to_agg:
                     print(f"   ❌ No valid metrics to aggregate for this table. Skipping aggregation.")
                     markdown_output = generate_markdown_table(pd.DataFrame(), title, index_col)
                else:
                     agg_df = filtered_df.groupby(index_col)[metrics_to_agg].agg(['mean', 'std'])

                     formatted_cols = {}
                     for metric in metrics_to_agg:
                         clean_name = metric.replace(f"round{TARGET_ROUND}_", "").replace("_", " ").title()
                         if "Asr" in clean_name: clean_name = clean_name.replace("Asr", "ASR")

                         mean_col = (metric, 'mean')
                         std_col = (metric, 'std')
                         if mean_col in agg_df.columns:
                              std_series = agg_df[std_col] if std_col in agg_df.columns else pd.Series([0.0]*len(agg_df), index=agg_df.index)
                              formatted_cols[clean_name] = format_metric(agg_df[mean_col], std_series)
                         else:
                              formatted_cols[clean_name] = "N/A"

                     final_df = pd.DataFrame(formatted_cols)

                     # Sort index if it's strategy name
                     if index_col == 'strategy-name':
                          current_strategies_in_table = final_df.index.unique()
                          ordered_index = [s for s in STRATEGY_ORDER if s in current_strategies_in_table]
                          ordered_index += sorted([s for s in current_strategies_in_table if s not in STRATEGY_ORDER])
                          final_df = final_df.reindex(ordered_index)
                     elif index_col == 'scenario_name':
                          final_df = final_df.sort_index()

                     markdown_output = generate_markdown_table(final_df, title, index_col)
            table_path = TABLES_DIR / filename
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(markdown_output)
            print(f"   ✅ Saved table to {table_path.name}")

        except Exception as e:
            print(f"   🔥 Error processing table definition '{title}': {e}")
            traceback.print_exc()

    print("\n✅ Table generation complete. Files are in the 'tables' directory.")

if __name__ == "__main__":
    main()