"""
fedopt.py — FL-Privacy-IDS
Federated Optimization (FedOpt) — Reddi et al. 2020
Server-side optimization with momentum for faster convergence.
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
from src.fl_algorithms.fedavg import local_train

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_ROUNDS        = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 10
LEARNING_RATE     = 0.001
SERVER_LR         = 0.01
BETA1             = 0.9
BETA2             = 0.99
TAU               = 1e-3   # stability constant


# ─────────────────────────────────────────────────────────────────────────────
# SERVER OPTIMIZER — Adam on server side
# ─────────────────────────────────────────────────────────────────────────────
class ServerOptimizer:
    """
    Server-side Adam optimizer for FedOpt.
    Applies momentum to the pseudo-gradient (difference between
    global model and aggregated client updates).
    """

    def __init__(self, global_weights: dict, lr: float = SERVER_LR,
                 beta1: float = BETA1, beta2: float = BETA2, tau: float = TAU):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.tau     = tau
        self.t       = 0  # step counter

        # Momentum buffers
        self.m = {k: torch.zeros_like(v) for k, v in global_weights.items()
                  if v.dtype == torch.float32}
        self.v = {k: torch.ones_like(v) * tau ** 2 for k, v in global_weights.items()
                  if v.dtype == torch.float32}

    def step(self, global_weights: dict, aggregated_weights: dict) -> dict:
        """
        Apply server Adam update.

        pseudo_gradient = aggregated_weights - global_weights
        new_global = global_weights + lr * Adam(pseudo_gradient)
        """
        self.t += 1
        new_weights = copy.deepcopy(global_weights)

        for key in global_weights:
            if global_weights[key].dtype != torch.float32:
                # Keep integer buffers (e.g. BatchNorm counters) as-is
                new_weights[key] = aggregated_weights[key].clone()
                continue

            # Pseudo-gradient
            delta = aggregated_weights[key].float() - global_weights[key].float()

            # Adam momentum update
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * delta ** 2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update global weights
            new_weights[key] = global_weights[key] + self.lr * m_hat / (
                torch.sqrt(v_hat) + self.tau
            )

        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# FEDOPT AGGREGATION — weighted average then server optimizer step
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_fedopt(
    global_weights: dict,
    client_weights: list,
    client_sizes: list,
    server_opt: ServerOptimizer,
) -> dict:
    """
    FedOpt aggregation:
    1. Weighted average of client weights (same as FedAvg)
    2. Apply server-side Adam optimizer step
    """
    total_samples   = sum(client_sizes)
    averaged_weights = copy.deepcopy(client_weights[0])

    for key in averaged_weights:
        averaged_weights[key] = torch.zeros_like(averaged_weights[key])

    for weights, size in zip(client_weights, client_sizes):
        w = size / total_samples
        for key in averaged_weights:
            if weights[key].dtype == torch.long:
                averaged_weights[key] = weights[key].clone()
            else:
                averaged_weights[key] += w * weights[key].float()

    # Server optimizer step
    new_global = server_opt.step(global_weights, averaged_weights)
    return new_global


# ─────────────────────────────────────────────────────────────────────────────
# FEDOPT TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_fedopt(
    client_loaders: list,
    test_loader: DataLoader,
    input_dim: int,
    num_rounds: int        = NUM_ROUNDS,
    clients_per_round: int = CLIENTS_PER_ROUND,
    local_epochs: int      = LOCAL_EPOCHS,
    server_lr: float       = SERVER_LR,
    experiment: str        = "baseline",
    verbose: bool          = True,
) -> ResultTracker:
    """
    Full FedOpt training loop.

    Args:
        client_loaders   : list of DataLoaders (one per client)
        test_loader      : global test DataLoader
        input_dim        : number of input features
        num_rounds       : total FL rounds
        clients_per_round: clients selected per round
        local_epochs     : local training epochs per round
        server_lr        : server-side Adam learning rate
        experiment       : experiment name for result tracking
        verbose          : print progress

    Returns:
        ResultTracker with full training history
    """
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracker = ResultTracker(algorithm="FedOpt", experiment=experiment)

    # Initialize global model
    global_model  = get_model(input_dim=input_dim).to(device)
    global_weights= get_model_weights(global_model)
    comm_cost     = compute_comm_cost(global_model, clients_per_round)

    # Initialize server optimizer
    server_opt = ServerOptimizer(global_weights, lr=server_lr)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FedOpt | Experiment: {experiment} | server_lr={server_lr}")
        print(f"Rounds: {num_rounds} | Clients/round: {clients_per_round}")
        print(f"Local epochs: {local_epochs} | Device: {device}")
        print(f"Comm cost/round: {comm_cost} MB")
        print(f"{'='*60}")

    for round_num in range(1, num_rounds + 1):
        # Select random clients
        selected = random.sample(range(len(client_loaders)), clients_per_round)

        client_weights = []
        client_sizes   = []

        # Local training — same as FedAvg
        for client_id in selected:
            local_model = get_model(input_dim=input_dim)
            local_model = set_model_weights(local_model, copy.deepcopy(global_weights))
            weights, size, _ = local_train(
                local_model,
                client_loaders[client_id],
                device,
                local_epochs,
            )
            client_weights.append(weights)
            client_sizes.append(size)

        # Aggregate with server optimizer
        global_weights = aggregate_fedopt(
            global_weights, client_weights, client_sizes, server_opt
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
        print(f"\nFedOpt Complete!")
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
    print("Testing FedOpt (3 rounds)...")
    print("=" * 60)

    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    tracker = run_fedopt(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = 3,
        clients_per_round = 5,
        local_epochs      = 2,
        server_lr         = 0.01,
        experiment        = "test",
    )

    print(f"\nBest accuracy in 3 rounds: {tracker.get_best_accuracy()}%")
    print("\n🎉 fedopt.py is working correctly!")
