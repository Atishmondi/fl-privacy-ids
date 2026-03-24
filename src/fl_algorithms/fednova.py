"""
fednova.py — FL-Privacy-IDS
Federated Nova (FedNova) — Wang et al. 2020
Normalizes local updates by number of local steps to correct
objective inconsistency from heterogeneous client updates.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from src.models import get_model, get_model_weights, set_model_weights
from src.evaluate import evaluate, ResultTracker, compute_comm_cost

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_ROUNDS        = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 5
LEARNING_RATE     = 0.0005


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING — returns normalized update (delta / tau)
# ─────────────────────────────────────────────────────────────────────────────
def local_train_nova(
    model: nn.Module,
    global_weights: dict,
    dataloader: DataLoader,
    device: torch.device,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LEARNING_RATE,
) -> tuple:
    """
    Train locally and return normalized gradient update.

    FedNova key idea:
    - tau = total local steps (epochs × batches)
    - normalized_update = (w_local - w_global) / tau
    - This ensures all clients contribute equally regardless
      of how many local steps they took
    """
    model = model.to(device)
    model.train()

    optimizer   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    num_samples = len(dataloader.dataset)
    tau         = 0  # local step counter

    for epoch in range(local_epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tau        += 1  # count each gradient step

    # Get local weights after training
    local_weights = get_model_weights(model)

    # Compute normalized update: delta_i / tau_i
    normalized_update = {}
    for key in local_weights:
        if local_weights[key].dtype == torch.long:
            normalized_update[key] = local_weights[key].clone()
        else:
            delta = local_weights[key].float() - global_weights[key].float()
            normalized_update[key] = delta / tau

    avg_loss = total_loss / tau if tau > 0 else 0.0
    return normalized_update, num_samples, tau, avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# FEDNOVA AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_nova(
    global_weights: dict,
    normalized_updates: list,
    client_sizes: list,
    tau_list: list,
) -> dict:
    """
    FedNova aggregation:
    global_update = Σ (n_i/N) * tau_eff * normalized_update_i
    new_global    = global + global_update

    tau_eff = effective local steps (weighted average of tau values)
    """
    total_samples = sum(client_sizes)

    # Compute effective tau (weighted average)
    tau_eff = sum(
        (size / total_samples) * tau
        for size, tau in zip(client_sizes, tau_list)
    )

    # Weighted sum of normalized updates
    global_update = {}
    for key in global_weights:
        if global_weights[key].dtype == torch.long:
            global_update[key] = normalized_updates[0][key].clone()
            continue
        global_update[key] = torch.zeros_like(global_weights[key].float())

    for update, size in zip(normalized_updates, client_sizes):
        weight = size / total_samples
        for key in global_update:
            if global_weights[key].dtype == torch.long:
                continue
            global_update[key] += weight * update[key].float()

    # Apply update to global model
    new_global_weights = {}
    for key in global_weights:
        if global_weights[key].dtype == torch.long:
            new_global_weights[key] = global_update[key].clone()
        else:
            new_global_weights[key] = (
                global_weights[key].float() + tau_eff * global_update[key]
            )

    return new_global_weights


# ─────────────────────────────────────────────────────────────────────────────
# FEDNOVA TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_fednova(
    client_loaders: list,
    test_loader: DataLoader,
    input_dim: int,
    num_rounds: int        = NUM_ROUNDS,
    clients_per_round: int = CLIENTS_PER_ROUND,
    local_epochs: int      = LOCAL_EPOCHS,
    experiment: str        = "baseline",
    verbose: bool          = True,
) -> ResultTracker:
    """
    Full FedNova training loop.

    Args:
        client_loaders   : list of DataLoaders (one per client)
        test_loader      : global test DataLoader
        input_dim        : number of input features
        num_rounds       : total FL rounds
        clients_per_round: clients selected per round
        local_epochs     : local training epochs per round
        experiment       : experiment name for result tracking
        verbose          : print progress

    Returns:
        ResultTracker with full training history
    """
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = ResultTracker(algorithm="FedNova", experiment=experiment)

    # Initialize global model
    global_model   = get_model(input_dim=input_dim).to(device)
    global_weights = get_model_weights(global_model)
    comm_cost      = compute_comm_cost(global_model, clients_per_round)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FedNova | Experiment: {experiment}")
        print(f"Rounds: {num_rounds} | Clients/round: {clients_per_round}")
        print(f"Local epochs: {local_epochs} | Device: {device}")
        print(f"Comm cost/round: {comm_cost} MB")
        print(f"{'='*60}")

    for round_num in range(1, num_rounds + 1):
        # Select random clients
        selected = random.sample(range(len(client_loaders)), clients_per_round)

        normalized_updates = []
        client_sizes       = []
        tau_list           = []

        # Local training — normalized updates
        for client_id in selected:
            local_model = get_model(input_dim=input_dim)
            local_model = set_model_weights(local_model, copy.deepcopy(global_weights))

            norm_update, size, tau, _ = local_train_nova(
                model          = local_model,
                global_weights = global_weights,
                dataloader     = client_loaders[client_id],
                device         = device,
                local_epochs   = local_epochs,
            )
            normalized_updates.append(norm_update)
            client_sizes.append(size)
            tau_list.append(tau)

        # FedNova aggregation
        global_weights = aggregate_nova(
            global_weights, normalized_updates, client_sizes, tau_list
        )
        global_model = set_model_weights(global_model, global_weights)

        # Evaluate every 5 rounds and at round 1
        if round_num == 1 or round_num % 5 == 0:
            metrics = evaluate(global_model, test_loader, device)
            tracker.log(
                round_num=round_num,
                metrics=metrics,
                extra={"comm_cost_mb": comm_cost * round_num},
            )

    # Save results
    tracker.save()

    if verbose:
        summary = tracker.summary()
        print(f"\nFedNova Complete!")
        print(f"Best Accuracy : {summary['best_accuracy']}%")
        print(f"Best F1       : {summary['best_f1']}%")
        print(f"Best Round    : {summary['best_round']}")

    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — 3 rounds only
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.preprocess import get_fl_data

    print("=" * 60)
    print("Testing FedNova (3 rounds)...")
    print("=" * 60)

    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    tracker = run_fednova(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = 3,
        clients_per_round = 5,
        local_epochs      = 2,
        experiment        = "test",
    )

    print(f"\nBest accuracy in 3 rounds: {tracker.get_best_accuracy()}%")
    print("\n🎉 fednova.py is working correctly!")
