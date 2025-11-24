import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback 
import argparse

OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR = Path("./plots")

# Definitions for the various plots to be generated
PLOTS_DEFINITIONS = [
    # ==============================================================================
    # --- FEDAVG IID vs NON-IID (Full Participation) ---
    # ==============================================================================
    {
        "title": "FedAvg Performance: IID vs Non-IID (Full Participation)",
        "filename": "fig_00_fedavg_iid_vs_noniid.pdf",
        "filters": {
            "strategy-name": "FedAvg", # Usa il nome pulito
            "fraction-train": 1.0, "attack_name": ["none", None, ""],
            "personalization": [False, None],
            "scenario_name": ["FedAvg_IID", "FedAvg_nonIID_a0_1", "FedAvg_nonIID_a0_5", "FedAvg_nonIID_a0_3", "FedAvg_nonIID_a0_03"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- SCENARI IID (Full Participation) ---
    # ==============================================================================
    {
        "title": "IID (Full Participation) - Baseline Strategies",
        "filename": "fig_01_iid_baseline.pdf",
        "filters": {
            "partitioner-name": "iid", "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_IID", "FedProx_IID", "FedAvgM_IID"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "IID (Full Participation) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_02_iid_adaptive.pdf",
        "filters": {
            "partitioner-name": "iid", "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_IID", "FedAdam_IID", "FedYogi_IID"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- SCENARI NON-IID α=0.5 (Full Participation) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.5, Full Participation) - Baseline Strategies",
        "filename": "fig_03_noniid_a05_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_5", "FedProx_nonIID_a0_5", "FedAvgM_nonIID_a0_5"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "Non-IID (α=0.5, Full Participation) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_04_noniid_a05_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_5", "FedAdam_nonIID_a0_5", "FedYogi_nonIID_a0_5"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- SCENARI NON-IID α=0.1 (Full Participation) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.1, Full Participation) - Baseline Strategies",
        "filename": "fig_05_noniid_a01_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_1", "FedProx_nonIID_a0_1", "FedAvgM_nonIID_a0_1"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "Non-IID (α=0.1, Full Participation) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_06_noniid_a01_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_1", "FedAdam_nonIID_a0_1", "FedYogi_nonIID_a0_1"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- SCENARI NON-IID α=0.03 (Full Participation) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.03, Full Participation) - Baseline Strategies",
        "filename": "fig_09_noniid_a003_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03", "FedProx_nonIID_a0_03", "FedAvgM_nonIID_a0_03"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "Non-IID (α=0.03, Full Participation) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_10_noniid_a003_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 1.0,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03", "FedAdam_nonIID_a0_03", "FedYogi_nonIID_a0_03"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- PERSONALIZATION vs ALL (Full Participation) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.1, Full Participation) - All Strategies vs Personalized FedAvg",
        "filename": "fig_11_noniid_a01_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 1.0,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_1", "FedProx_nonIID_a0_1", "FedAvgM_nonIID_a0_1",
                "FedAdam_nonIID_a0_1", "FedYogi_nonIID_a0_1",
                "FedAvg_nonIID_a0_1_personalized"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "Non-IID (α=0.5, Full Participation) - All Strategies vs Personalized FedAvg",
        "filename": "fig_12_noniid_a05_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 1.0,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_5", "FedProx_nonIID_a0_5", "FedAvgM_nonIID_a0_5",
                "FedAdam_nonIID_a0_5", "FedYogi_nonIID_a0_5",
                "FedAvg_nonIID_a0_5_personalized"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },
    {
        "title": "Non-IID (α=0.03, Full Participation) - All Strategies vs Personalized FedAvg",
        "filename": "fig_14_noniid_a003_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 1.0,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_03", "FedProx_nonIID_a0_03", "FedAvgM_nonIID_a0_03",
                "FedAdam_nonIID_a0_03", "FedYogi_nonIID_a0_03",
                "FedAvg_nonIID_a0_03_personalized" 
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard"
    },

    # ==============================================================================
    # --- SCENARI CON PARTECIPAZIONE PARZIALE (fraction=0.8) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.1, fraction=0.8) - Baseline Strategies",
        "filename": "fig_15_noniid_a01_frac08_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac08",
                "FedProx_nonIID_a0_1_frac08", 
                "FedAvgM_nonIID_a0_1_frac08"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.1, fraction=0.8) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_16_noniid_a01_frac08_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac08",
                "FedAdam_nonIID_a0_1_frac08", 
                "FedYogi_nonIID_a0_1_frac08"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.1, fraction=0.8) - All Strategies vs Personalized FedAvg",
        "filename": "fig_17_noniid_a01_frac08_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.8,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac08", 
                "FedProx_nonIID_a0_1_frac08", 
                "FedAvgM_nonIID_a0_1_frac08",
                "FedAdam_nonIID_a0_1_frac08", 
                "FedYogi_nonIID_a0_1_frac08",
                "FedAvg_nonIID_a0_1_personalized_frac08"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.8) - Baseline Strategies",
        "filename": "fig_18_noniid_a05_frac08_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac08",
                "FedProx_nonIID_a0_5_frac08",
                "FedAvgM_nonIID_a0_5_frac08"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.8) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_19_noniid_a05_frac08_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac08", 
                "FedAdam_nonIID_a0_5_frac08", 
                "FedYogi_nonIID_a0_5_frac08"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.8) - All Strategies vs Personalized FedAvg",
        "filename": "fig_20_noniid_a05_frac08_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.8,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac08", 
                "FedProx_nonIID_a0_5_frac08", 
                "FedAvgM_nonIID_a0_5_frac08",
                "FedAdam_nonIID_a0_5_frac08", 
                "FedYogi_nonIID_a0_5_frac08",
                "FedAvg_nonIID_a0_5_personalized_frac08"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.8) - Baseline Strategies",
        "filename": "fig_23_noniid_a003_frac08_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03_frac08", "FedProx_nonIID_a0_03_frac08", "FedAvgM_nonIID_a0_03_frac08"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.8) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_24_noniid_a003_frac08_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.8,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03_frac08", "FedAdam_nonIID_a0_03_frac08", "FedYogi_nonIID_a0_03_frac08"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.8) - All Strategies vs Personalized FedAvg",
        "filename": "fig_30_noniid_a003_frac08_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.8,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_03_frac08", 
                "FedProx_nonIID_a0_03_frac08", 
                "FedAvgM_nonIID_a0_03_frac08",
                "FedAdam_nonIID_a0_03_frac08", 
                "FedYogi_nonIID_a0_03_frac08",
                "FedAvg_nonIID_a0_03_personalized_frac08"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },

    # ==============================================================================
    # --- SCENARI CON PARTECIPAZIONE PARZIALE (fraction=0.7) ---
    # ==============================================================================
    {
        "title": "Non-IID (α=0.1, fraction=0.7) - Baseline Strategies",
        "filename": "fig_25_noniid_a01_frac07_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac07",
                "FedProx_nonIID_a0_1_frac07",
                "FedAvgM_nonIID_a0_1_frac07"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.1, fraction=0.7) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_26_noniid_a01_frac07_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac07",
                "FedAdam_nonIID_a0_1_frac07", 
                "FedYogi_nonIID_a0_1_frac07"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.1, fraction=0.7) - All Strategies vs Personalized FedAvg",
        "filename": "fig_27_noniid_a01_frac07_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1, "fraction-train": 0.7,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_1_frac07", 
                "FedProx_nonIID_a0_1_frac07", 
                "FedAvgM_nonIID_a0_1_frac07",
                "FedAdam_nonIID_a0_1_frac07", 
                "FedYogi_nonIID_a0_1_frac07",
                "FedAvg_nonIID_a0_1_personalized_frac07"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.7) - Baseline Strategies",
        "filename": "fig_28_noniid_a05_frac07_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac07", 
                "FedProx_nonIID_a0_5_frac07",
                "FedAvgM_nonIID_a0_5_frac07"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.7) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_29_noniid_a05_frac07_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac07", 
                "FedAdam_nonIID_a0_5_frac07", 
                "FedYogi_nonIID_a0_5_frac07"
            ]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.5, fraction=0.7) - All Strategies vs Personalized FedAvg",
        "filename": "fig_30_noniid_a05_frac07_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5, "fraction-train": 0.7,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_5_frac07", 
                "FedProx_nonIID_a0_5_frac07", 
                "FedAvgM_nonIID_a0_5_frac07",
                "FedAdam_nonIID_a0_5_frac07", 
                "FedYogi_nonIID_a0_5_frac07",
                "FedAvg_nonIID_a0_5_personalized_frac07"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.7) - Baseline Strategies",
        "filename": "fig_33_noniid_a003_frac07_baseline.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03_frac07", "FedProx_nonIID_a0_03_frac07", "FedAvgM_nonIID_a0_03_frac07"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.7) - Adaptive Optimizers vs FedAvg",
        "filename": "fig_34_noniid_a003_frac07_adaptive.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.7,
            "attack_name": ["none", None, ""], "personalization": [False, None],
            "scenario_name": ["FedAvg_nonIID_a0_03_frac07", "FedAdam_nonIID_a0_03_frac07", "FedYogi_nonIID_a0_03_frac07"]
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    {
        "title": "Non-IID (α=0.03, fraction=0.7) - All Strategies vs Personalized FedAvg",
        "filename": "fig_30_noniid_a003_frac07_all_vs_personalized.pdf",
        "filters": {
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.03, "fraction-train": 0.7,
            "attack_name": ["none", None, ""],
            "scenario_name": [
                "FedAvg_nonIID_a0_03_frac07", 
                "FedProx_nonIID_a0_03_frac07", 
                "FedAvgM_nonIID_a0_03_frac07",
                "FedAdam_nonIID_a0_03_frac07", 
                "FedYogi_nonIID_a0_03_frac07",
                "FedAvg_nonIID_a0_03_personalized_frac07"
            ],
        },
        "comparison_key": "scenario_name",
        "plot_type": "standard",
        "include_full_participation_baseline": True 
    },
    # ==============================================================================
    # --- PLOTS ATTACKS ---
    # ==============================================================================
    # --- Backdoor (Full Participation) ---
    {
        "title": "Robust Strategies under Backdoor Attack (IID, Full Participation) vs Baseline",
        "filename": "fig_35_attack_backdoor_iid_full.pdf",
        "filters": {
            "attack_name": "backdoor", 
            "partitioner-name": "iid", 
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "backdoor",
        "include_baseline": True 
    },
    {
        "title": "Robust Strategies under Backdoor Attack (non-IID, α=0.5, Full Participation) vs Baseline",
        "filename": "fig_36_attack_backdoor_noniid_a05_full.pdf",
        "filters": {
            "attack_name": "backdoor", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "backdoor",
        "include_baseline": True
    },
    {
        "title": "Robust Strategies under Backdoor Attack (non-IID, α=0.1, Full Participation) vs Baseline",
        "filename": "fig_37_attack_backdoor_noniid_a01_full.pdf",
        "filters": {
            "attack_name": "backdoor", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "backdoor",
        "include_baseline": True
    },
    # --- Byzantine (Full Participation) ---
    {
        "title": "Robust Strategies under Byzantine Attack (IID, Full Participation) vs Baseline",
        "filename": "fig_38_attack_byzantine_iid_full.pdf",
        "filters": {
            "attack_name": "byzantine", 
            "partitioner-name": "iid",
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
    {
        "title": "Robust Strategies under Byzantine Attack (non-IID, α=0.5, Full Participation) vs Baseline",
        "filename": "fig_39_attack_byzantine_noniid_a05_full.pdf",
        "filters": {
            "attack_name": "byzantine", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
     {
        "title": "Robust Strategies under Byzantine Attack (non-IID, α=0.1, Full Participation) vs Baseline",
        "filename": "fig_40_attack_byzantine_noniid_a01_full.pdf",
        "filters": {
            "attack_name": "byzantine", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
    # --- Label Flipping (Full Participation) ---
    {
        "title": "Robust Strategies under Label Flipping Attack (IID, Full Participation) vs Baseline",
        "filename": "fig_41_attack_labelflip_iid_full.pdf",
        "filters": {
            "attack_name": "label_flipping", 
            "partitioner-name": "iid",
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
    {
        "title": "Robust Strategies under Label Flipping Attack (non-IID, α=0.5, Full Participation) vs Baseline",
        "filename": "fig_42_attack_labelflip_noniid_a05_full.pdf",
        "filters": {
            "attack_name": "label_flipping", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.5,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
    {
        "title": "Robust Strategies under Label Flipping Attack (non-IID, α=0.1, Full Participation) vs Baseline",
        "filename": "fig_43_attack_labelflip_noniid_a01_full.pdf",
        "filters": {
            "attack_name": "label_flipping", 
            "partitioner-name": "dirichlet", "dirichlet-alpha": 0.1,
            "fraction-train": 1.0
        },
        "comparison_key": "strategy-name",
        "plot_type": "standard",
        "include_baseline": True
    },
]

# load_all_results
def load_all_results(outputs_dir: Path) -> pd.DataFrame:
    all_data = []
    processed_configs = set()

    for json_path in outputs_dir.rglob("results.json"):
        run_id = json_path.parent.name
        if run_id in processed_configs:
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = data.get("config", {})
            config["run_id"] = run_id

            if "scenario_name" not in config:
                continue

            # Set default values for key config columns
            config.setdefault("attack_name", "none")
            if config["attack_name"] is None or config["attack_name"] == "": config["attack_name"] = "none"
            config.setdefault("personalization", False)
            if config["personalization"] is None: config["personalization"] = False
            config.setdefault("fraction-train", 1.0)
            if config["fraction-train"] is None: config["fraction-train"] = 1.0
            if 'partitioner-name' in config and config['partitioner-name'] == 'iid':
                  config.setdefault("dirichlet-alpha", -1.0)
            elif 'dirichlet-alpha' not in config: config["dirichlet-alpha"] = np.nan
            if config["dirichlet-alpha"] is None: config["dirichlet-alpha"] = np.nan
            config.setdefault("strategy-name", "unknown")
            config.setdefault("proximal-mu", 0.0)


            # Process rounds
            rounds_data = data.get("rounds", [])
            if not rounds_data:
                processed_configs.add(run_id)
                continue

            for round_data in rounds_data:
                row = config.copy()
                round_num = round_data.get("round")
                if round_num is None: continue

                row["round"] = round_num

                # Populate metrics, ensuring NaN for missing values
                for eval_type in ["training", "client_evaluation", "server_evaluation"]:
                    metrics = round_data.get(eval_type, {})
                    expected_cols = []
                    if eval_type == "training": expected_cols = ["loss", "num_examples"]
                    elif eval_type == "client_evaluation": expected_cols = ["loss", "accuracy", "f1_score", "backdoor_asr", "num_examples"]
                    elif eval_type == "server_evaluation": expected_cols = ["loss", "accuracy", "f1_score", "num_examples"]

                    if isinstance(metrics, dict):
                        for name in expected_cols:
                            row[f"{eval_type}_{name}"] = metrics.get(name, np.nan)
                    else:
                        for name in expected_cols: row[f"{eval_type}_{name}"] = np.nan

                all_data.append(row)

            processed_configs.add(run_id)

        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            print(f"Skipping corrupted, unreadable, or incomplete file: {json_path} ({type(e).__name__}: {e})")
            processed_configs.add(run_id)

    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)

    # Data Cleaning and Type Conversion
    print("Performing data cleaning and type conversion...")
    numeric_cols = ["round", "dirichlet-alpha", "fraction-train", "proximal-mu"] + \
                   [col for col in df.columns if any(p in col for p in ["training_", "client_evaluation_", "server_evaluation_"]) and "num_examples" not in col]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    example_cols = [col for col in df.columns if "num_examples" in col]
    for col in example_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').fillna(0)

    if 'personalization' in df.columns: df['personalization'] = df['personalization'].fillna(False).astype(bool)
    else: df['personalization'] = False

    string_cols = ["run_id", "scenario_name", "strategy-name", "partitioner-name", "attack_name"]
    for col in string_cols:
         if col in df.columns: df[col] = df[col].fillna('none').astype(str)
         else: df[col] = 'none'

    # Handle FedAvg/FedProx naming based on proximal-mu AFTER converting mu to numeric
    if 'strategy-name' in df.columns and 'proximal-mu' in df.columns:
        fedprox_mask = (df['strategy-name'] == 'fedprox')
        avg_mask = (df['proximal-mu'] == 0.0) | df['proximal-mu'].isna()
        prox_mask = df['proximal-mu'] > 0.0
        df.loc[fedprox_mask & avg_mask, 'strategy-name'] = 'FedAvg'
        df.loc[fedprox_mask & prox_mask, 'strategy-name'] = 'FedProx'

    # Apply other name mappings
    name_map = { "fedavgm": "FedAvgM", "fedadam": "FedAdam", "fedyogi": "FedYogi",
                 "median": "FedMedian", "trimmed_avg": "TrimmedAvg", "multikrum": "MultiKrum"}
    if 'strategy-name' in df.columns: df['strategy-name'] = df['strategy-name'].replace(name_map)


    if 'dirichlet-alpha' in df.columns and 'partitioner-name' in df.columns:
        is_iid_mask = df['partitioner-name'] == 'iid'
        df.loc[is_iid_mask, 'dirichlet-alpha'] = df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)

    print(f"Data loading complete. DataFrame shape: {df.shape}")
    return df


# function to plot a single metric with mean and std shading
def plot_metric(ax, group, metric_col, plot_args, is_personalized=False):
    if ax is None: return False
    if metric_col not in group.columns or group[metric_col].isnull().all(): return False
    try:
        plot_data = group.dropna(subset=['round', metric_col]).copy()
        plot_data['round'] = pd.to_numeric(plot_data['round'], errors='coerce').astype(int)
        if plot_data.empty: return False
        agg = plot_data.groupby('round')[metric_col].agg(['mean', 'std']).fillna(0)
        if agg.empty: return False
        line, = ax.plot(agg.index, agg['mean'], **plot_args)
        shadow_color = 'red' if is_personalized else plot_args.get('color', 'blue')
        std_dev = agg['std'].replace([np.inf, -np.inf], 0).fillna(0)
        lower_bound = (agg['mean'] - std_dev).clip(lower=0)
        upper_bound = (agg['mean'] + std_dev).clip(upper=1.05)
        ax.fill_between(agg.index, lower_bound, upper_bound, color=shadow_color, alpha=0.1)
        return True
    except Exception as e:
        name = plot_args.get('label', 'Unknown')
        print(f"   🔥 Error in plot_metric for {name}, metric {metric_col}: {e}")
        traceback.print_exc()
        return False

# function to generate standard plots (2x2 or 1x2 for personalization)
def generate_standard_plot(df: pd.DataFrame, plot_def: dict, output_dir: Path, baseline_df: pd.DataFrame = None, is_attack_plot: bool = False, force_client_only: bool = False):
    title = plot_def["title"]
    filename = plot_def["filename"]
    comparison_key = plot_def["comparison_key"]

    if df.empty and (baseline_df is None or baseline_df.empty):
        print(f"Skipping plot '{title}' - no data found after filtering.")
        return

    scenarios_in_plot = df[comparison_key].unique() if not df.empty else []
    is_personalization_involved = any('personalized' in str(name).lower() for name in scenarios_in_plot) or \
                                  ('personalization' in df.columns and df['personalization'].any())

    TITLE_FS = 20
    LABEL_FS = 16
    TICK_FS = 12
    LEGEND_FS = 16
    SUPTITLE_FS = 24

   
    if is_personalization_involved or force_client_only:
        print(f"   Generating 2x1 layout (client-side only) for: {title}")
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharey=True, sharex=True) 
        fig.suptitle(title, fontsize=SUPTITLE_FS, y=0.98)
        try: 
            ax_client_acc, ax_client_f1 = axes
        except ValueError: 
            ax_client_acc = axes; ax_client_f1 = fig.add_subplot(212); print(f"   Warning: Unexpected axes shape for 2x1.")
        
        ax_server_acc, ax_server_f1 = None, None
        axes_list = [ax_client_acc, ax_client_f1]
        
        ax_client_acc.set_title("Client-Side Aggregated Accuracy", fontsize=TITLE_FS)
        ax_client_f1.set_title("Client-Side Aggregated F1-Score", fontsize=TITLE_FS)
        ax_client_acc.set_ylabel("Metric Value", fontsize=LABEL_FS)
        ax_client_f1.set_ylabel("Metric Value", fontsize=LABEL_FS)
    else:
        print(f"   Generating 2x2 layout for standard plot: {title}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True) 
        fig.suptitle(title, fontsize=SUPTITLE_FS, y=0.98)
        try:
             ax_server_acc, ax_server_f1 = axes[0, 0], axes[0, 1]
             ax_client_acc, ax_client_f1 = axes[1, 0], axes[1, 1]
             axes_list = axes.flatten()
        except (IndexError, TypeError):
             print(f"   Warning: Unexpected axes shape for 2x2 layout.")
             axes_list = np.array(axes).flatten(); ax_server_acc, ax_server_f1, ax_client_acc, ax_client_f1 = (axes_list[i] if i < len(axes_list) else None for i in range(4))
        
        if ax_server_acc: ax_server_acc.set_title("Server-Side Accuracy", fontsize=TITLE_FS)
        if ax_server_f1: ax_server_f1.set_title("Server-Side F1-Score", fontsize=TITLE_FS)
        if ax_client_acc: ax_client_acc.set_title("Client-Side Aggregated Accuracy", fontsize=TITLE_FS)
        if ax_client_f1: ax_client_f1.set_title("Client-Side Aggregated F1-Score", fontsize=TITLE_FS)
        
        if ax_server_acc: ax_server_acc.set_ylabel("Metric Value", fontsize=LABEL_FS)
        if ax_client_acc: ax_client_acc.set_ylabel("Metric Value", fontsize=LABEL_FS)

    max_round_data = max(df["round"].max() if not df.empty else 0, baseline_df["round"].max() if baseline_df is not None and not baseline_df.empty else 0)
    max_round = int(max_round_data) if pd.notna(max_round_data) else 40
    xticks = np.arange(0, max_round + 1, 2)
    
    for ax in axes_list:
        if ax is None: continue
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.01, 1.01); ax.set_xlim(-0.5, max_round + 0.5)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS) 
        ax.set_xticks(xticks)

    xlabel_text = "Federated Learning Round"
    if is_personalization_involved or force_client_only:
        if ax_client_f1: ax_client_f1.set_xlabel(xlabel_text, fontsize=LABEL_FS)
        if ax_client_acc: ax_client_acc.set_xlabel("") 
    else:
        if ax_server_acc: ax_server_acc.set_xlabel("")
        if ax_server_f1: ax_server_f1.set_xlabel("")
        if ax_client_acc: ax_client_acc.set_xlabel(xlabel_text, fontsize=LABEL_FS)
        if ax_client_f1: ax_client_f1.set_xlabel(xlabel_text, fontsize=LABEL_FS)

    handles = []; labels = []
    colors = plt.get_cmap('tab10').colors
    linestyles = ['-', '--', '-.', ':'] * 5

    if baseline_df is not None and not baseline_df.empty:
        baseline_name = "FedAvg (Full Participation)" if not is_attack_plot else "FedAvg (No Attack)"
        baseline_args = {'label': baseline_name, 'color': 'black', 'linestyle': '--', 'linewidth': 2.0}
        
        # Plot baseline metrics
        bl_plotted_server_acc = plot_metric(ax_server_acc, baseline_df, "server_evaluation_accuracy", baseline_args, is_personalized=False) if ax_server_acc and not (is_personalization_involved or force_client_only) else False
        bl_plotted_server_f1 = plot_metric(ax_server_f1, baseline_df, "server_evaluation_f1_score", baseline_args, is_personalized=False) if ax_server_f1 and not (is_personalization_involved or force_client_only) else False
        bl_plotted_client_acc = plot_metric(ax_client_acc, baseline_df, "client_evaluation_accuracy", baseline_args, is_personalized=False) if ax_client_acc else False
        bl_plotted_client_f1 = plot_metric(ax_client_f1, baseline_df, "client_evaluation_f1_score", baseline_args, is_personalized=False) if ax_client_f1 else False
        
        # Add baseline handle/label
        if bl_plotted_server_acc or bl_plotted_server_f1 or bl_plotted_client_acc or bl_plotted_client_f1:
            current_ax = ax_client_acc if (is_personalization_involved or force_client_only) else (ax_server_acc if ax_server_acc else ax_client_acc) # Get a valid axis
            if current_ax:
                line = next((l for l in current_ax.lines if l.get_label() == baseline_name), None)
                if line and baseline_name not in labels:
                    handles.append(line); labels.append(baseline_name)

    # Plot main scenarios
    scenarios = sorted(df[comparison_key].unique()) if not df.empty else []
    color_idx = 0
    for i, name in enumerate(scenarios):
        group = df[df[comparison_key] == name].copy()
        if group.empty: continue

        color = colors[color_idx % len(colors)]
    
        if (baseline_df is not None and not baseline_df.empty) and np.allclose(color[:3], [0,0,0]): # Check RGB part of color tuple/list
            color_idx += 1
            color = colors[color_idx % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        color_idx += 1

        is_current_line_personalized = 'personalization' in group.columns and not group['personalization'].empty and group['personalization'].iloc[0] == True

        plot_label = name

        if is_attack_plot and name == "FedAvg":
            plot_label = "FedAvg (Under Attack)"

        if is_current_line_personalized:
            linestyle = ':'
            plot_args = {'label': plot_label, 'color': 'red', 'linestyle': linestyle, 'linewidth': 4.0}
        else:
            plot_args = {'label': plot_label, 'color': color, 'linestyle': linestyle, 'linewidth': 2.5}

        # Plot metrics
        plotted_server_acc = plot_metric(ax_server_acc, group, "server_evaluation_accuracy", plot_args, is_current_line_personalized) if ax_server_acc and not (is_personalization_involved or force_client_only) else False
        plotted_server_f1 = plot_metric(ax_server_f1, group, "server_evaluation_f1_score", plot_args, is_current_line_personalized) if ax_server_f1 and not (is_personalization_involved or force_client_only) else False
        plotted_client_acc = plot_metric(ax_client_acc, group, "client_evaluation_accuracy", plot_args, is_current_line_personalized) if ax_client_acc else False
        plotted_client_f1 = plot_metric(ax_client_f1, group, "client_evaluation_f1_score", plot_args, is_current_line_personalized) if ax_client_f1 else False

        # Add handle/label
        if plotted_server_acc or plotted_server_f1 or plotted_client_acc or plotted_client_f1:
             current_ax = ax_client_acc if (is_personalization_involved or force_client_only) else (ax_server_acc if ax_server_acc else ax_client_acc)
             if current_ax:
                  line = next((l for l in current_ax.lines if l.get_label() == plot_label), None)
                  if line and plot_label not in labels:
                       handles.append(line); labels.append(plot_label)

    if handles:
        num_cols = 3 
        
        if is_personalization_involved or force_client_only: 
            fig.subplots_adjust(bottom=0.20, hspace=0.15) 
        else: 
            fig.subplots_adjust(bottom=0.15, hspace=0.2)
            
        fig.legend(handles, labels, title="Legend", loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=num_cols, fontsize=LEGEND_FS, title_fontsize=LEGEND_FS+2) 
    else:
        print(f"   ⚠️ No data plotted for '{title}'. Legend skipped.")

    output_dir.mkdir(exist_ok=True) 
    plot_path = output_dir / filename 
    try:
        fig.savefig(plot_path, bbox_inches='tight') 
        print(f"   ✅ Generated plot: {plot_path.name} in {output_dir.name}")
    except Exception as e_save:
        print(f"   🔥 Error saving plot '{filename}': {e_save}")
    plt.close(fig)

def generate_backdoor_plot(df: pd.DataFrame, plot_def: dict, output_dir: Path, baseline_df: pd.DataFrame = None, is_attack_plot: bool = False):
    title = plot_def["title"]
    filename = plot_def["filename"]
    comparison_key = plot_def["comparison_key"]

    if df.empty and (baseline_df is None or baseline_df.empty):
        print(f"Skipping plot '{title}' - no data found after filtering.")
        return

    TITLE_FS = 20
    LABEL_FS = 16
    TICK_FS = 12
    LEGEND_FS = 16
    SUPTITLE_FS = 24

    fig, axes = plt.subplots(3, 1, figsize=(12, 20), sharey=True, sharex=True) 
    fig.suptitle(title, fontsize=SUPTITLE_FS, y=0.98) 
    
    try:
        ax_client_acc, ax_client_f1, ax_client_asr = axes
    except ValueError:
        print("   Warning: Unexpected axes shape for 3x1.")
        ax_client_acc, ax_client_f1, ax_client_asr = axes[0], axes[1], axes[2]
        
    axes_list = axes

    ax_client_acc.set_title("Client-Side Aggregated Accuracy", fontsize=TITLE_FS)
    ax_client_f1.set_title("Client-Side Aggregated F1-Score", fontsize=TITLE_FS)
    ax_client_asr.set_title("Client-Side Attack Success Rate (ASR)", fontsize=TITLE_FS)
    ax_client_acc.set_ylabel("Metric Value", fontsize=LABEL_FS)
    ax_client_f1.set_ylabel("Metric Value", fontsize=LABEL_FS)
    ax_client_asr.set_ylabel("Metric Value", fontsize=LABEL_FS)

    max_round_data = max(df["round"].max() if not df.empty else 0, baseline_df["round"].max() if baseline_df is not None and not baseline_df.empty else 0)
    max_round = int(max_round_data) if pd.notna(max_round_data) else 40
    xticks = np.arange(0, max_round + 1, 2)
    
    xlabel_text = "Federated Learning Round"
    
    for ax in axes_list:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.01, 1.01); ax.set_xlim(-0.5, max_round + 0.5)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS) 
        ax.set_xticks(xticks)
        ax.set_xlabel("") 
    
    ax_client_asr.set_xlabel(xlabel_text, fontsize=LABEL_FS)

    handles = []; labels = []
    colors = plt.get_cmap('tab10').colors
    linestyles = ['-', '--', '-.', ':'] * 5

    if baseline_df is not None and not baseline_df.empty:
        baseline_name = "FedAvg (No Attack)"
        baseline_args = {'label': baseline_name, 'color': 'black', 'linestyle': '--', 'linewidth': 2.0}
        bl_plotted_acc = plot_metric(ax_client_acc, baseline_df, "client_evaluation_accuracy", baseline_args, is_personalized=False)
        bl_plotted_f1 = plot_metric(ax_client_f1, baseline_df, "client_evaluation_f1_score", baseline_args, is_personalized=False)
        bl_plotted_asr = plot_metric(ax_client_asr, baseline_df, "client_evaluation_backdoor_asr", baseline_args, is_personalized=False)
        if bl_plotted_acc or bl_plotted_f1 or bl_plotted_asr:
             current_ax = ax_client_acc
             if current_ax:
                  line = next((l for l in current_ax.lines if l.get_label() == baseline_name), None)
                  if line and baseline_name not in labels:
                       handles.append(line); labels.append(baseline_name)

    scenarios = sorted(df[comparison_key].unique()) if not df.empty else []
    color_idx = 0
    for i, name in enumerate(scenarios):
        group = df[df[comparison_key] == name].copy()
        if group.empty: continue

        color = colors[color_idx % len(colors)]
        if (baseline_df is not None and not baseline_df.empty) and np.allclose(color[:3], [0,0,0]):
            color_idx += 1; color = colors[color_idx % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        color_idx += 1

        plot_label = f"{name} (Under Attack)" if name == "FedAvg" and is_attack_plot else name
        plot_args = {'label': plot_label, 'color': color, 'linestyle': linestyle, 'linewidth': 2.5}
        is_personalized = False

        plotted_acc = plot_metric(ax_client_acc, group, "client_evaluation_accuracy", plot_args, is_personalized)
        plotted_f1 = plot_metric(ax_client_f1, group, "client_evaluation_f1_score", plot_args, is_personalized)
        plotted_asr = plot_metric(ax_client_asr, group, "client_evaluation_backdoor_asr", plot_args, is_personalized)

        if plotted_acc or plotted_f1 or plotted_asr:
             current_ax = ax_client_acc
             if current_ax:
                  line = next((l for l in current_ax.lines if l.get_label() == plot_label), None)
                  if line and plot_label not in labels:
                       handles.append(line); labels.append(plot_label)

    if handles:
        num_cols = 3 
        fig.subplots_adjust(bottom=0.15, hspace=0.2)
        fig.legend(handles, labels, title="Legend", loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=num_cols, fontsize=LEGEND_FS, title_fontsize=LEGEND_FS+2) 
    else:
        print(f"   ⚠️ No data plotted for '{title}'. Legend skipped.")

    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / filename 
    try:
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"   ✅ Generated plot: {plot_path.name} in {output_dir.name}")
    except Exception as e_save:
         print(f"   🔥 Error saving plot '{filename}': {e_save}")
    plt.close(fig)

if __name__ == "__main__":
    print("📊 Starting analysis...")
    PLOTS_DIR_STANDARD = PLOTS_DIR / "standard"
    PLOTS_DIR_CLIENT = PLOTS_DIR / "client_only"
    PLOTS_DIR_STANDARD.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR_CLIENT.mkdir(parents=True, exist_ok=True)
    print(f"Standard plots (2x2) will be saved to: {PLOTS_DIR_STANDARD}")
    print(f"Client-only plots (2x1) will be saved to: {PLOTS_DIR_CLIENT}")
        
    print("⏳ Loading results data...")
    master_df = load_all_results(OUTPUTS_DIR)

    if not master_df.empty:
        print(f"✅ Loaded data for {master_df['run_id'].nunique()} runs across {master_df['scenario_name'].nunique()} scenarios.")

        required_cols = {'personalization': bool, 'attack_name': str, 'fraction-train': float, 'dirichlet-alpha': float, 'proximal-mu': float, 'partitioner-name': str}
        default_values = {'personalization': False, 'attack_name': 'none', 'fraction-train': 1.0, 'dirichlet-alpha': np.nan, 'proximal-mu': 0.0, 'partitioner-name': 'unknown'}
        for col, dtype in required_cols.items():
             if col not in master_df.columns: master_df[col] = default_values[col]
             if dtype == bool: master_df[col] = master_df[col].fillna(default_values[col]).astype(bool)
             elif dtype == str: master_df[col] = master_df[col].fillna(default_values[col]).astype(str)
             elif dtype == float: master_df[col] = pd.to_numeric(master_df[col].fillna(default_values[col]), errors='coerce')
        if 'partitioner-name' in master_df.columns:
             is_iid_mask = master_df['partitioner-name'] == 'iid'
             master_df.loc[is_iid_mask, 'dirichlet-alpha'] = master_df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)

        print("\n⚙️ Generating plots...")
        for plot_def in PLOTS_DEFINITIONS:
            filtered_df = master_df.copy()
            title = plot_def.get('title', 'Unnamed Plot')
            print(f"\n🔍 Filtering data for: {title}")
            baseline_fedavg_df = None

            try:
                filters = plot_def.get("filters", {})
                if not filters: print("   ⚠️ No filters defined for this plot.")

                for key, value in filters.items():
                    if key not in filtered_df.columns: print(f"   ⚠️ Warning: Filter key '{key}' not found. Skipping."); continue
                    
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
                            elif key in ["fraction-train", "dirichlet-alpha", "proximal-mu"]: filter_val = np.nan
                            else: filter_val = 'none'
                        if pd.isna(filter_val) and key in ["dirichlet-alpha"]:
                            filtered_df = filtered_df[filtered_df[key].isna()]
                        elif isinstance(filter_val, bool) and key == 'personalization':
                            filtered_df = filtered_df[filtered_df[key] == filter_val]
                        elif isinstance(filter_val, (int, float)) and key in ["fraction-train", "dirichlet-alpha", "proximal-mu"]:
                            numeric_col = pd.to_numeric(filtered_df[key], errors='coerce')
                            if isinstance(filter_val, float): filtered_df = filtered_df[np.isclose(numeric_col, filter_val, equal_nan=True)]
                            else: filtered_df = filtered_df[numeric_col == filter_val]
                        else:
                            filtered_df = filtered_df[filtered_df[key].astype(str) == str(filter_val)]
                
                print(f"   📊 Found {filtered_df['run_id'].nunique()} runs ({len(filtered_df)} rows) matching main filters for '{title}'.")
                include_baseline_flag = plot_def.get("include_baseline", False)
                include_full_participation_baseline_flag = plot_def.get("include_full_participation_baseline", False)

                if include_baseline_flag:
                    print("   Finding corresponding baseline FedAvg (No Attack) data...")
                    
                    baseline_filters = {
                        "attack_name": "none",
                        "strategy-name": "FedAvg",
                        "personalization": False 
                    }
                    
                    key_filters_to_copy = ["partitioner-name", "dirichlet-alpha", "fraction-train"]
                    for key in key_filters_to_copy:
                        if key in filters:
                            baseline_filters[key] = filters[key]
                            
                    print(f"   Baseline filters (Attack): {baseline_filters}") 

                    baseline_fedavg_df = master_df.copy()
                    
                    for key, value in baseline_filters.items():
                        if key not in baseline_fedavg_df.columns: continue
                        
                        filter_val = value
                        if isinstance(filter_val, (int, float)):
                            numeric_col = pd.to_numeric(baseline_fedavg_df[key], errors='coerce')
                            if isinstance(filter_val, float): 
                                baseline_fedavg_df = baseline_fedavg_df[np.isclose(numeric_col, filter_val, equal_nan=True)]
                            else:
                                baseline_fedavg_df = baseline_fedavg_df[numeric_col == filter_val]
                        elif isinstance(filter_val, bool):
                             baseline_fedavg_df = baseline_fedavg_df[baseline_fedavg_df[key] == filter_val]
                        else: 
                             baseline_fedavg_df = baseline_fedavg_df[baseline_fedavg_df[key].astype(str) == str(filter_val)]

                    if baseline_fedavg_df.empty:
                         print("   ⚠️ Baseline FedAvg (No Attack) data not found for this configuration.")
                    else:
                          print(f"    Found {baseline_fedavg_df['run_id'].nunique()} baseline runs.")
                
                elif include_full_participation_baseline_flag:
                    print("   Finding corresponding FULL PARTICIPATION baseline FedAvg (No Attack, frac=1.0) data...")
                    
                    baseline_filters = {
                        "attack_name": "none",
                        "strategy-name": "FedAvg",
                        "personalization": False,
                        "fraction-train": 1.0 
                    }
                    
                    key_filters_to_copy = ["partitioner-name", "dirichlet-alpha"]
                    for key in key_filters_to_copy:
                        if key in filters:
                            baseline_filters[key] = filters[key]
                            
                    print(f"   Baseline filters (Full Participation): {baseline_filters}")

                    baseline_fedavg_df = master_df.copy()
                    
                    for key, value in baseline_filters.items():
                        if key not in baseline_fedavg_df.columns: continue
                        
                        filter_val = value
                        if isinstance(filter_val, (int, float)):
                            numeric_col = pd.to_numeric(baseline_fedavg_df[key], errors='coerce')
                            if isinstance(filter_val, float): 
                                baseline_fedavg_df = baseline_fedavg_df[np.isclose(numeric_col, filter_val, equal_nan=True)]
                            else:
                                baseline_fedavg_df = baseline_fedavg_df[numeric_col == filter_val]
                        elif isinstance(filter_val, bool):
                             baseline_fedavg_df = baseline_fedavg_df[baseline_fedavg_df[key] == filter_val]
                        else:
                             baseline_fedavg_df = baseline_fedavg_df[baseline_fedavg_df[key].astype(str) == str(filter_val)]

                    if baseline_fedavg_df.empty:
                         print("   ⚠️ Full Participation Baseline FedAvg (No Attack) data not found.")
                    else:
                          print(f"    Found {baseline_fedavg_df['run_id'].nunique()} full participation baseline runs.")

                if not filtered_df.empty or (baseline_fedavg_df is not None and not baseline_fedavg_df.empty):
                    if filtered_df['run_id'].nunique() < 2: print("   ⚠️ Only one main run found, standard deviation will be zero.")
                    if baseline_fedavg_df is not None and baseline_fedavg_df['run_id'].nunique() < 2: 
                        print("   ⚠️ Only one baseline run found, baseline standard deviation will be zero.")

                    plot_type = plot_def.get("plot_type", "standard")
                    
                    effective_baseline_df = baseline_fedavg_df if (include_baseline_flag or include_full_participation_baseline_flag) and baseline_fedavg_df is not None and not baseline_fedavg_df.empty else None
                    is_attack_plot_flag = include_baseline_flag

                    if plot_type == "backdoor":
                    
                        print(" -> Generating plot for 'standard' folder...")
                        generate_backdoor_plot(df=filtered_df, plot_def=plot_def, baseline_df=effective_baseline_df, is_attack_plot=is_attack_plot_flag, output_dir=PLOTS_DIR_STANDARD)
                        print(" -> Generating plot for 'client_only' folder...")
                        generate_backdoor_plot(df=filtered_df, plot_def=plot_def, baseline_df=effective_baseline_df, is_attack_plot=is_attack_plot_flag, output_dir=PLOTS_DIR_CLIENT)
                    else:
                        print(" -> Generating plot for 'standard' folder...")
                        generate_standard_plot(df=filtered_df, plot_def=plot_def, baseline_df=effective_baseline_df, is_attack_plot=is_attack_plot_flag, force_client_only=False, output_dir=PLOTS_DIR_STANDARD)
                        print(" -> Generating plot for 'client_only' folder...")
                        generate_standard_plot(df=filtered_df, plot_def=plot_def, baseline_df=effective_baseline_df, is_attack_plot=is_attack_plot_flag, force_client_only=True, output_dir=PLOTS_DIR_CLIENT)
                else:
                    print(f"   ❌ No data remaining after filtering (and no baseline requested/found) for '{title}'. Skipping plot generation.")

            except Exception as e:
                print(f"   🔥 Error processing plot definition '{title}': {e}")
                traceback.print_exc()

        print("\n✅ Analysis complete. Plots saved in 'plots/standard' and 'plots/client_only' directories.")
    else:
        print("\n❌ Could not find or load any valid 'results.json' files in the 'outputs' directory.")
        print("   Please run experiments using 'run_experiments.py' first.")