"""
fedavg.py — FL-Privacy-IDS
Federated Averaging (FedAvg) — McMahan et al. 2017
Weighted average of client model weights each round.
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
NUM_ROUNDS        = 150
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 5
LEARNING_RATE     = 0.0005
BATCH_SIZE        = 32


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING — one client trains for local_epochs
# ─────────────────────────────────────────────────────────────────────────────
def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LEARNING_RATE,
) -> tuple:
    """
    Train a model locally on one client's data.

    Returns:
        updated model weights, number of samples, average loss
    """
    model = model.to(device)
    model.train()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_samples = len(dataloader.dataset)

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

    num_batches = local_epochs * len(dataloader)
    avg_loss    = total_loss / num_batches if num_batches > 0 else 0.0
    return get_model_weights(model), num_samples, avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION — weighted average of client weights
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(client_weights: list, client_sizes: list) -> dict:
    """
    FedAvg aggregation — weighted average by dataset size.

    global_w = Σ (n_i / N) × w_i
    """
    total_samples  = sum(client_sizes)
    global_weights = copy.deepcopy(client_weights[0])

    for key in global_weights:
        global_weights[key] = torch.zeros_like(global_weights[key])

    for weights, size in zip(client_weights, client_sizes):
        weight = size / total_samples
        for key in global_weights:
            if weights[key].dtype == torch.long:
                global_weights[key] = weights[key].clone()
            else:
                global_weights[key] += weight * weights[key].float()

    return global_weights


# ─────────────────────────────────────────────────────────────────────────────
# FEDAVG TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_fedavg(
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
    Full FedAvg training loop.

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
    device  = torch.device("cpu")
    tracker = ResultTracker(algorithm="FedAvg", experiment=experiment)

    global_model = get_model(input_dim=input_dim).to(device)
    comm_cost    = compute_comm_cost(global_model, clients_per_round)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FedAvg | Experiment: {experiment}")
        print(f"Rounds: {num_rounds} | Clients/round: {clients_per_round}")
        print(f"Local epochs: {local_epochs} | Device: {device}")
        print(f"Comm cost/round: {comm_cost} MB")
        print(f"{'='*60}")

    for round_num in range(1, num_rounds + 1):
        selected = random.sample(range(len(client_loaders)), clients_per_round)

        client_weights = []
        client_sizes   = []

        for client_id in selected:
            local_model = get_model(input_dim=input_dim)
            local_model = set_model_weights(local_model, copy.deepcopy(
                get_model_weights(global_model)
            ))
            weights, size, _ = local_train(
                local_model,
                client_loaders[client_id],
                device,
                local_epochs,
            )
            client_weights.append(weights)
            client_sizes.append(size)

        global_weights = aggregate(client_weights, client_sizes)
        global_model   = set_model_weights(global_model, global_weights)

        if round_num == 1 or round_num % 5 == 0:
            metrics = evaluate(global_model, test_loader, device)
            tracker.log(
                round_num=round_num,
                metrics=metrics,
                extra={"comm_cost_mb": comm_cost * round_num},
            )

    tracker.save()

    if verbose:
        summary = tracker.summary()
        print(f"\nFedAvg Complete!")
        print(f"Best Accuracy : {summary['best_accuracy']}%")
        print(f"Best F1       : {summary['best_f1']}%")
        print(f"Best Round    : {summary['best_round']}")

    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.preprocess import get_fl_data

    print("=" * 60)
    print("Testing FedAvg (3 rounds)...")
    print("=" * 60)

    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    tracker = run_fedavg(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = 3,
        clients_per_round = 5,
        local_epochs      = 2,
        experiment        = "test",
    )

    print(f"\nBest accuracy in 3 rounds: {tracker.get_best_accuracy()}%")
    print("\nfedavg.py is working correctly!")
