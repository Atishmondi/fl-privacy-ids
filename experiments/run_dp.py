"""
run_dp.py — FL-Privacy-IDS
Experiment 3: Differential Privacy analysis
Tests all 4 algorithms with Opacus DP-SGD under epsilon = 1, 10, and inf (no DP).

Design notes:
- Device forced to CPU for all runs — Opacus does not support MPS (Apple Metal)
- Each algorithm uses its own proper local training and aggregation logic
- FedProx: proximal term preserved with DP-SGD (Opacus clips the total gradient including proximal)
- FedOpt: server-side Adam preserved — server optimizer never touches gradients so safe with DP
- FedNova: included with tau=0 guard — may collapse at eps=1, documented as finding from E2
- epsilon=inf = no DP on CPU — isolates privacy cost from MPS->CPU device switch cost
- GroupNorm already in model so Opacus compatibility guaranteed
- drop_last=True in preprocess so no ZeroDivisionError on empty batches
"""

import sys
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append(".")

from src.preprocess import get_fl_data
from src.models import get_model, get_model_weights, set_model_weights
from src.evaluate import evaluate, ResultTracker, compute_comm_cost, save_summary
from src.dp_wrapper import attach_dp, get_epsilon
from src.fl_algorithms.fedavg import aggregate
from src.fl_algorithms.fednova import aggregate_nova

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ROUNDS        = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS      = 5
LEARNING_RATE     = 0.0005
MU                = 0.01
SERVER_LR         = 0.01
BETA1, BETA2, TAU = 0.9, 0.99, 1e-3
DELTA             = 1e-5
MAX_GRAD_NORM     = 1.0

EPSILON_VALUES    = ["inf"]
ALGORITHMS        = ["FedAvg", "FedProx", "FedOpt", "FedNova"]

DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — attach Opacus if epsilon is finite
# ─────────────────────────────────────────────────────────────────────────────
def maybe_attach_dp(model, optimizer, dataloader, epsilon):
    if epsilon is None:
        return model, optimizer, dataloader, None
    model, optimizer, dataloader, engine = attach_dp(
        model          = model,
        optimizer      = optimizer,
        dataloader     = dataloader,
        target_epsilon = epsilon,
        target_delta   = DELTA,
        max_grad_norm  = MAX_GRAD_NORM,
        epochs         = LOCAL_EPOCHS,
    )
    return model, optimizer, dataloader, engine


def extract_weights(model, engine):
    if engine is not None:
        return {k: v.clone() for k, v in model._module.state_dict().items()}
    return get_model_weights(model)


# ─────────────────────────────────────────────────────────────────────────────
# FEDAVG LOCAL TRAIN WITH DP
# ─────────────────────────────────────────────────────────────────────────────
def local_train_fedavg_dp(model, dataloader, epsilon):
    model = model.cpu()
    model.train()
    optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    num_samples = len(dataloader.dataset)

    model, optimizer, dataloader, engine = maybe_attach_dp(
        model, optimizer, dataloader, epsilon
    )

    for epoch in range(LOCAL_EPOCHS):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.cpu(), y_batch.cpu()
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    num_batches = LOCAL_EPOCHS * len(dataloader)
    avg_loss    = total_loss / num_batches if num_batches > 0 else 0.0
    eps_spent   = get_epsilon(engine) if engine else float("inf")
    return extract_weights(model, engine), num_samples, avg_loss, eps_spent


# ─────────────────────────────────────────────────────────────────────────────
# FEDPROX LOCAL TRAIN WITH DP
# Proximal term preserved — Opacus clips total gradient (CE + proximal)
# ─────────────────────────────────────────────────────────────────────────────
def local_train_fedprox_dp(model, global_weights_frozen, dataloader, epsilon):
    model = model.cpu()
    model.train()
    optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    num_samples = len(dataloader.dataset)

    model, optimizer, dataloader, engine = maybe_attach_dp(
        model, optimizer, dataloader, epsilon
    )

    for epoch in range(LOCAL_EPOCHS):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.cpu(), y_batch.cpu()
            optimizer.zero_grad()

            ce_loss = criterion(model(X_batch), y_batch)

            prox_term = 0.0
            named_params = model._module.named_parameters() \
                           if engine is not None else model.named_parameters()
            for name, param in named_params:
                if name in global_weights_frozen:
                    prox_term += torch.norm(
                        param - global_weights_frozen[name].cpu()
                    ) ** 2

            loss = ce_loss + (MU / 2) * prox_term
            loss.backward()
            optimizer.step()
            total_loss += ce_loss.item()

    num_batches = LOCAL_EPOCHS * len(dataloader)
    avg_loss    = total_loss / num_batches if num_batches > 0 else 0.0
    eps_spent   = get_epsilon(engine) if engine else float("inf")
    return extract_weights(model, engine), num_samples, avg_loss, eps_spent


# ─────────────────────────────────────────────────────────────────────────────
# FEDNOVA LOCAL TRAIN WITH DP
# tau=0 guard included — collapse at eps=1 documented as valid finding
# ─────────────────────────────────────────────────────────────────────────────
def local_train_fednova_dp(model, global_weights, dataloader, epsilon):
    model = model.cpu()
    model.train()
    optimizer   = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    num_samples = len(dataloader.dataset)
    tau         = 0

    model, optimizer, dataloader, engine = maybe_attach_dp(
        model, optimizer, dataloader, epsilon
    )

    for epoch in range(LOCAL_EPOCHS):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.cpu(), y_batch.cpu()
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tau        += 1

    local_weights = extract_weights(model, engine)

    normalized_update = {}
    for key in local_weights:
        if local_weights[key].dtype == torch.long:
            normalized_update[key] = local_weights[key].clone()
        else:
            if tau > 0:
                delta = local_weights[key].float() - global_weights[key].cpu().float()
                normalized_update[key] = delta / tau
            else:
                normalized_update[key] = torch.zeros_like(local_weights[key].float())

    avg_loss  = total_loss / tau if tau > 0 else 0.0
    eps_spent = get_epsilon(engine) if engine else float("inf")
    return normalized_update, num_samples, tau, avg_loss, eps_spent


# ─────────────────────────────────────────────────────────────────────────────
# FEDOPT SERVER OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class FedOptServerState:
    def __init__(self, global_weights):
        self.t = 0
        self.m = {k: torch.zeros_like(v)
                  for k, v in global_weights.items() if v.dtype == torch.float32}
        self.v = {k: torch.ones_like(v) * TAU ** 2
                  for k, v in global_weights.items() if v.dtype == torch.float32}

    def step(self, global_weights, aggregated_weights):
        self.t += 1
        new_weights = copy.deepcopy(global_weights)
        for key in global_weights:
            if global_weights[key].dtype != torch.float32:
                new_weights[key] = aggregated_weights[key].clone()
                continue
            delta        = aggregated_weights[key].float() - global_weights[key].float()
            self.m[key]  = BETA1 * self.m[key] + (1 - BETA1) * delta
            self.v[key]  = BETA2 * self.v[key] + (1 - BETA2) * delta ** 2
            m_hat        = self.m[key] / (1 - BETA1 ** self.t)
            v_hat        = self.v[key] / (1 - BETA2 ** self.t)
            new_weights[key] = global_weights[key] + \
                               SERVER_LR * m_hat / (torch.sqrt(v_hat) + TAU)
        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP — one algorithm, one epsilon
# ─────────────────────────────────────────────────────────────────────────────
def run_dp_experiment(
    algorithm: str,
    client_loaders: list,
    test_loader: DataLoader,
    input_dim: int,
    epsilon,
) -> ResultTracker:

    experiment     = f"dp_eps{epsilon}"
    tracker        = ResultTracker(algorithm=algorithm, experiment=experiment)
    eps_value      = None if epsilon == "inf" else float(epsilon)

    global_model   = get_model(input_dim=input_dim).cpu()
    global_weights = get_model_weights(global_model)
    comm_cost      = compute_comm_cost(global_model, CLIENTS_PER_ROUND)

    server_state   = FedOptServerState(global_weights) if algorithm == "FedOpt" else None

    print(f"\n{'='*60}")
    print(f"{algorithm} | epsilon={epsilon} | delta={DELTA}")
    print(f"Rounds: {NUM_ROUNDS} | Clients/round: {CLIENTS_PER_ROUND}")
    print(f"Local epochs: {LOCAL_EPOCHS} | Device: cpu")
    print(f"Comm cost/round: {comm_cost} MB")
    print(f"{'='*60}")

    for round_num in range(1, NUM_ROUNDS + 1):
        selected           = random.sample(range(len(client_loaders)), CLIENTS_PER_ROUND)
        client_weights     = []
        client_sizes       = []
        tau_list           = []
        epsilon_spent_list = []

        for client_id in selected:
            local_model = get_model(input_dim=input_dim)
            local_model = set_model_weights(local_model, copy.deepcopy(global_weights))

            if algorithm == "FedAvg":
                w, size, _, eps = local_train_fedavg_dp(
                    local_model, client_loaders[client_id], eps_value
                )
                client_weights.append(w)
                client_sizes.append(size)
                epsilon_spent_list.append(eps)

            elif algorithm == "FedProx":
                frozen = {k: v.detach().clone()
                          for k, v in global_model.named_parameters()}
                w, size, _, eps = local_train_fedprox_dp(
                    local_model, frozen, client_loaders[client_id], eps_value
                )
                client_weights.append(w)
                client_sizes.append(size)
                epsilon_spent_list.append(eps)

            elif algorithm == "FedOpt":
                w, size, _, eps = local_train_fedavg_dp(
                    local_model, client_loaders[client_id], eps_value
                )
                client_weights.append(w)
                client_sizes.append(size)
                epsilon_spent_list.append(eps)

            elif algorithm == "FedNova":
                norm_update, size, tau, _, eps = local_train_fednova_dp(
                    local_model, global_weights,
                    client_loaders[client_id], eps_value
                )
                client_weights.append(norm_update)
                client_sizes.append(size)
                tau_list.append(tau)
                epsilon_spent_list.append(eps)

        # Aggregation
        if algorithm in ["FedAvg", "FedProx"]:
            global_weights = aggregate(client_weights, client_sizes)

        elif algorithm == "FedOpt":
            averaged       = aggregate(client_weights, client_sizes)
            global_weights = server_state.step(global_weights, averaged)

        elif algorithm == "FedNova":
            if all(t == 0 for t in tau_list):
                pass  # all clients empty — skip round, keep weights
            else:
                global_weights = aggregate_nova(
                    global_weights, client_weights, client_sizes, tau_list
                )

        global_model = set_model_weights(global_model, global_weights)

        if round_num == 1 or round_num % 5 == 0:
            finite_eps = [e for e in epsilon_spent_list if e != float("inf")]
            avg_eps    = sum(finite_eps) / len(finite_eps) if finite_eps else float("inf")

            metrics = evaluate(global_model, test_loader, DEVICE)
            tracker.log(
                round_num = round_num,
                metrics   = metrics,
                extra     = {
                    "epsilon_target" : epsilon,
                    "epsilon_spent"  : round(avg_eps, 4) if avg_eps != float("inf") else "inf",
                    "comm_cost_mb"   : comm_cost * round_num,
                },
            )

    tracker.save()
    summary = tracker.summary()
    print(f"\n{algorithm} (eps={epsilon}) Complete!")
    print(f"Best Accuracy  : {summary['best_accuracy']}%")
    print(f"Best Composite : {summary['best_composite']}%")
    print(f"Best Round     : {summary['best_round']}")
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 3 — Differential Privacy Analysis")
    print("=" * 60)
    print(f"Algorithms     : {ALGORITHMS}")
    print(f"Epsilon values : {EPSILON_VALUES}")
    print(f"Delta          : {DELTA}")
    print(f"Rounds         : {NUM_ROUNDS}")
    print(f"Clients/round  : {CLIENTS_PER_ROUND}")
    print(f"Local epochs   : {LOCAL_EPOCHS}")
    print(f"Device         : cpu (Opacus does not support MPS)")

    start_total   = time.time()
    all_summaries = []

    print("\nLoading IID data...")
    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    for epsilon in EPSILON_VALUES:
        print(f"\n{'='*60}")
        print(f"Epsilon = {epsilon}")
        print(f"{'='*60}")

        for algo in ALGORITHMS:
            start   = time.time()
            tracker = run_dp_experiment(
                algorithm      = algo,
                client_loaders = client_loaders,
                test_loader    = test_loader,
                input_dim      = input_dim,
                epsilon        = epsilon,
            )
            summary = tracker.summary()
            summary["epsilon"]    = epsilon
            summary["train_time"] = round(time.time() - start, 2)
            all_summaries.append(summary)

    save_summary(all_summaries, "dp_summary.json")

    total_time = round(time.time() - start_total, 2)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Epsilon':>8} {'Accuracy':>10} {'F1':>10} {'Composite':>11} {'BestRnd':>8}")
    print("-" * 65)
    for s in all_summaries:
        print(f"{s['algorithm']:<12} {str(s['epsilon']):>8} "
              f"{s['best_accuracy']:>9.4f}% "
              f"{s['best_f1']:>9.4f}% "
              f"{s['best_composite']:>10.4f}% "
              f"{s['best_round']:>8}")

    print(f"\nTotal time: {total_time}s")
    print("\nExperiment 3 complete! Results saved to results/")
