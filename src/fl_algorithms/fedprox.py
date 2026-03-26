"""
fedprox.py — FL-Privacy-IDS
Federated Proximal (FedProx) — Li et al. 2020
Adds a proximal term to local loss to limit client drift.
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
from src.fl_algorithms.fedavg import aggregate

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
MU                = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING WITH PROXIMAL TERM
# ─────────────────────────────────────────────────────────────────────────────
def local_train_prox(
    model: nn.Module,
    global_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LEARNING_RATE,
    mu: float = MU,
) -> tuple:
    """
    Train locally with FedProx proximal term.

    Loss = CrossEntropy + (mu/2) * ||w - w_global||^2
    """
    model        = model.to(device)
    global_model = global_model.to(device)
    model.train()

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    num_samples = len(dataloader.dataset)

    global_weights = {k: v.detach().clone() for k, v in global_model.named_parameters()}

    for epoch in range(local_epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output  = model(X_batch)
            ce_loss = criterion(output, y_batch)

            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in global_weights:
                    prox_term += torch.norm(param - global_weights[name]) ** 2
            prox_loss = (mu / 2) * prox_term

            loss = ce_loss + prox_loss
            loss.backward()
            optimizer.step()
            total_loss += ce_loss.item()

    num_batches = local_epochs * len(dataloader)
    avg_loss    = total_loss / num_batches if num_batches > 0 else 0.0
    return get_model_weights(model), num_samples, avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# FEDPROX TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_fedprox(
    client_loaders: list,
    test_loader: DataLoader,
    input_dim: int,
    num_rounds: int        = NUM_ROUNDS,
    clients_per_round: int = CLIENTS_PER_ROUND,
    local_epochs: int      = LOCAL_EPOCHS,
    mu: float              = MU,
    experiment: str        = "baseline",
    verbose: bool          = True,
) -> ResultTracker:
    """
    Full FedProx training loop.
    """
    device  = torch.device("cpu")
    tracker = ResultTracker(algorithm="FedProx", experiment=experiment)

    global_model = get_model(input_dim=input_dim).to(device)
    comm_cost    = compute_comm_cost(global_model, clients_per_round)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FedProx | Experiment: {experiment} | mu={mu}")
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
            weights, size, _ = local_train_prox(
                model        = local_model,
                global_model = global_model,
                dataloader   = client_loaders[client_id],
                device       = device,
                local_epochs = local_epochs,
                mu           = mu,
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
                extra={"comm_cost_mb": comm_cost * round_num, "mu": mu},
            )

    tracker.save()

    if verbose:
        summary = tracker.summary()
        print(f"\nFedProx Complete!")
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
    print("Testing FedProx (3 rounds)...")
    print("=" * 60)

    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    tracker = run_fedprox(
        client_loaders    = client_loaders,
        test_loader       = test_loader,
        input_dim         = input_dim,
        num_rounds        = 3,
        clients_per_round = 5,
        local_epochs      = 2,
        mu                = 0.01,
        experiment        = "test",
    )

    print(f"\nBest accuracy in 3 rounds: {tracker.get_best_accuracy()}%")
    print("\nfedprox.py is working correctly!")
