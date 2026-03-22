"""
run_noniid.py — FL-Privacy-IDS
Experiment 2: Non-IID data distribution analysis
Tests all 4 algorithms under Dirichlet non-IID splits
with alpha = 0.1, 0.5, 1.0
"""

import sys
import time
sys.path.append(".")

from src.preprocess import get_fl_data
from src.evaluate import save_summary
from src.fl_algorithms.fedavg import run_fedavg
from src.fl_algorithms.fedprox import run_fedprox
from src.fl_algorithms.fedopt import run_fedopt
from src.fl_algorithms.fednova import run_fednova

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ROUNDS        = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 5
ALPHA_VALUES      = [0.5]  # high → low (easy → hard)


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL 4 ALGORITHMS FOR ONE ALPHA VALUE
# ─────────────────────────────────────────────────────────────────────────────
def run_all_algorithms(client_loaders, test_loader, input_dim, alpha):
    """Run all 4 FL algorithms on Non-IID data with given alpha."""
    experiment = f"noniid_alpha{alpha}"
    summaries  = []

    # FedAvg
    print(f"\n  [1/4] FedAvg (alpha={alpha})...")
    start = time.time()
    tracker = run_fedavg(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = experiment,
    )
    summary = tracker.summary()
    summary["alpha"]      = alpha
    summary["train_time"] = round(time.time() - start, 2)
    summaries.append(summary)

    # FedProx
    print(f"\n  [2/4] FedProx (alpha={alpha})...")
    start = time.time()
    tracker = run_fedprox(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = experiment,
    )
    summary = tracker.summary()
    summary["alpha"]      = alpha
    summary["train_time"] = round(time.time() - start, 2)
    summaries.append(summary)

    # FedOpt
    print(f"\n  [3/4] FedOpt (alpha={alpha})...")
    start = time.time()
    tracker = run_fedopt(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = experiment,
    )
    summary = tracker.summary()
    summary["alpha"]      = alpha
    summary["train_time"] = round(time.time() - start, 2)
    summaries.append(summary)

    # FedNova
    print(f"\n  [4/4] FedNova (alpha={alpha})...")
    start = time.time()
    tracker = run_fednova(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = NUM_ROUNDS,
        clients_per_round = CLIENTS_PER_ROUND,
        local_epochs      = LOCAL_EPOCHS,
        experiment        = experiment,
    )
    summary = tracker.summary()
    summary["alpha"]      = alpha
    summary["train_time"] = round(time.time() - start, 2)
    summaries.append(summary)

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 2 — Non-IID Analysis")
    print("=" * 60)
    print(f"Alpha values : {ALPHA_VALUES}")
    print(f"Rounds       : {NUM_ROUNDS}")
    print(f"Clients/round: {CLIENTS_PER_ROUND}")
    print(f"Local epochs : {LOCAL_EPOCHS}")

    start_total   = time.time()
    all_summaries = []

    for alpha in ALPHA_VALUES:
        print(f"\n{'='*60}")
        print(f"Non-IID Alpha = {alpha}")
        print(f"{'='*60}")

        print(f"Loading Non-IID data (alpha={alpha})...")
        client_loaders, test_loader, input_dim = get_fl_data(
            mode="noniid", alpha=alpha
        )

        summaries = run_all_algorithms(
            client_loaders, test_loader, input_dim, alpha
        )
        all_summaries.extend(summaries)

    # Save all results
    save_summary(all_summaries, "noniid_summary.json")

    # Print final table
    total_time = round(time.time() - start_total, 2)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Alpha':>6} {'Accuracy':>10} {'F1':>10} {'Composite':>11} {'Best Rnd':>9}")
    print("-" * 65)
    for s in all_summaries:
        print(f"{s['algorithm']:<12} {s['alpha']:>6} "
              f"{s['best_accuracy']:>9.4f}% "
              f"{s['best_f1']:>9.4f}% "
              f"{s['best_composite']:>10.4f}% "
              f"{s['best_round']:>9}")

    print(f"\nTotal time: {total_time}s")
    print("\nExperiment 2 complete! Results saved to results/")
