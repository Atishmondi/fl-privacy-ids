"""
run_baseline.py — FL-Privacy-IDS
Experiment 1: Baseline FL comparison (IID data, 150 rounds)
Runs all 4 algorithms and saves results for paper.
"""

import sys
import json
import time
sys.path.append(".")

from src.preprocess import get_fl_data
from src.evaluate import save_summary
from src.fl_algorithms.fedavg import run_fedavg
from src.fl_algorithms.fedprox import run_fedprox
from src.fl_algorithms.fedopt import run_fedopt
from src.fl_algorithms.fednova import run_fednova

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import os

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ROUNDS        = 150
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 5
EXPERIMENT        = "baseline_iid"


# ─────────────────────────────────────────────────────────────────────────────
# CENTRALIZED BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_centralized():
    print("\n" + "="*60)
    print("Running Centralized Baselines...")
    print("="*60)

    train_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv")

    drop_cols = ["id", "attack_cat"]
    cat_cols  = ["proto", "service", "state"]
    target    = "label"

    train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    test_df.drop(columns=drop_cols,  inplace=True, errors="ignore")

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

    X_train = train_df.drop(columns=[target]).values
    y_train = train_df[target].values
    X_test  = test_df.drop(columns=[target]).values
    y_test  = test_df[target].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = []

    print("\nTraining Random Forest...")
    start    = time.time()
    rf       = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_time  = round(time.time() - start, 2)
    rf_acc   = round(accuracy_score(y_test, rf_preds) * 100, 4)
    rf_f1    = round(f1_score(y_test, rf_preds, average="weighted") * 100, 4)
    print(f"Random Forest — Acc: {rf_acc}% | F1: {rf_f1}% | Time: {rf_time}s")
    results.append({
        "algorithm"    : "RandomForest",
        "experiment"   : EXPERIMENT,
        "best_accuracy": rf_acc,
        "best_f1"      : rf_f1,
        "type"         : "centralized",
        "train_time"   : rf_time,
    })

    print("\nTraining XGBoost...")
    start     = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, random_state=42,
        use_label_encoder=False, eval_metric="logloss",
        tree_method="hist", device="cpu",
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_time  = round(time.time() - start, 2)
    xgb_acc   = round(accuracy_score(y_test, xgb_preds) * 100, 4)
    xgb_f1    = round(f1_score(y_test, xgb_preds, average="weighted") * 100, 4)
    print(f"XGBoost — Acc: {xgb_acc}% | F1: {xgb_f1}% | Time: {xgb_time}s")
    results.append({
        "algorithm"    : "XGBoost",
        "experiment"   : EXPERIMENT,
        "best_accuracy": xgb_acc,
        "best_f1"      : xgb_f1,
        "type"         : "centralized",
        "train_time"   : xgb_time,
    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 1 — Baseline FL (IID, 150 rounds)")
    print("=" * 60)
    print(f"Rounds: {NUM_ROUNDS} | Clients/round: {CLIENTS_PER_ROUND}")
    print(f"Local epochs: {LOCAL_EPOCHS}")

    start_total = time.time()
    summaries   = []

    print("\nLoading data...")
    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    print("\n[1/4] Running FedAvg...")
    start        = time.time()
    tracker_avg  = run_fedavg(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = EXPERIMENT,
    )
    summary = tracker_avg.summary()
    summary["train_time"] = round(time.time() - start, 2)
    summary["type"]       = "federated"
    summaries.append(summary)

    print("\n[2/4] Running FedProx...")
    start        = time.time()
    tracker_prox = run_fedprox(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = EXPERIMENT,
    )
    summary = tracker_prox.summary()
    summary["train_time"] = round(time.time() - start, 2)
    summary["type"]       = "federated"
    summaries.append(summary)

    print("\n[3/4] Running FedOpt...")
    start       = time.time()
    tracker_opt = run_fedopt(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = EXPERIMENT,
    )
    summary = tracker_opt.summary()
    summary["train_time"] = round(time.time() - start, 2)
    summary["type"]       = "federated"
    summaries.append(summary)

    print("\n[4/4] Running FedNova...")
    start        = time.time()
    tracker_nova = run_fednova(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = EXPERIMENT,
    )
    summary = tracker_nova.summary()
    summary["train_time"] = round(time.time() - start, 2)
    summary["type"]       = "federated"
    summaries.append(summary)

    centralized = run_centralized()
    summaries.extend(centralized)

    save_summary(summaries, "baseline_summary.json")

    total_time = round(time.time() - start_total, 2)
    print("\n" + "="*60)
    print("EXPERIMENT 1 RESULTS")
    print("="*60)
    print(f"{'Algorithm':<15} {'Accuracy':>10} {'F1':>10} {'Type':<12}")
    print("-"*50)
    for s in summaries:
        print(f"{s['algorithm']:<15} {s['best_accuracy']:>9.4f}% "
              f"{s['best_f1']:>9.4f}% {s.get('type',''):<12}")

    print(f"\nTotal time: {total_time}s")
    print("\nExperiment 1 complete! Results saved to results/")
