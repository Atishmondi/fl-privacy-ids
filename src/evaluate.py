"""
evaluate.py — FL-Privacy-IDS
Evaluation metrics, result saving, and convergence tracking.
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from datetime import datetime

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── Results directory ─────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────
def composite_score(accuracy: float, f1: float, precision: float, recall: float) -> float:
    """
    Weighted composite score for selecting the best round.
    Fairer than accuracy alone on slightly imbalanced datasets.

    composite = 0.30*F1 + 0.25*Precision + 0.25*Recall + 0.20*Accuracy
    All inputs and output are in percentage (0-100 scale).
    """
    return round(0.30 * f1 + 0.25 * precision + 0.25 * recall + 0.20 * accuracy, 4)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE MODEL ON A DATALOADER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with accuracy, f1, precision, recall, loss, composite
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1   = round(f1_score(all_labels, all_preds, average="weighted") * 100, 4)
    prec = round(precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 4)
    rec  = round(recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 4)

    metrics = {
        "accuracy"  : acc,
        "f1"        : f1,
        "precision" : prec,
        "recall"    : rec,
        "loss"      : round(total_loss / len(dataloader), 4),
        "composite" : composite_score(acc, f1, prec, rec),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TRACK RESULTS ACROSS ROUNDS
# ─────────────────────────────────────────────────────────────────────────────
class ResultTracker:
    """
    Tracks metrics across FL rounds and saves results to CSV.
    Best round is selected by composite score, not accuracy alone.
    """

    def __init__(self, algorithm: str, experiment: str):
        self.algorithm  = algorithm
        self.experiment = experiment
        self.history    = []
        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, round_num: int, metrics: dict, extra: dict = None):
        """Log metrics for one round."""
        entry = {
            "round"     : round_num,
            "algorithm" : self.algorithm,
            "experiment": self.experiment,
            **metrics,
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)

        print(f"  Round {round_num:03d} | "
              f"Acc: {metrics['accuracy']:.2f}% | "
              f"F1: {metrics['f1']:.2f}% | "
              f"Composite: {metrics['composite']:.2f}% | "
              f"Loss: {metrics['loss']:.4f}")

    def save(self) -> str:
        """Save history to CSV and return filepath."""
        filename = f"{self.algorithm}_{self.experiment}_{self.timestamp}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        if not self.history:
            print("No results to save.")
            return ""

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

        print(f"Results saved to {filepath}")
        return filepath

    def get_best_composite(self) -> float:
        """Return the best composite score achieved across all rounds."""
        if not self.history:
            return 0.0
        return max(entry["composite"] for entry in self.history)

    def get_best_accuracy(self) -> float:
        """Return the best accuracy achieved across all rounds."""
        if not self.history:
            return 0.0
        return max(entry["accuracy"] for entry in self.history)

    def get_convergence_round(self, threshold: float = 90.0) -> int:
        """Return the first round where accuracy exceeded threshold."""
        for entry in self.history:
            if entry["accuracy"] >= threshold:
                return entry["round"]
        return -1

    def summary(self) -> dict:
        """
        Return a summary of the experiment results.
        Best round is chosen by composite score.
        """
        if not self.history:
            return {}

        best = max(self.history, key=lambda x: x["composite"])
        return {
            "algorithm"         : self.algorithm,
            "experiment"        : self.experiment,
            "best_accuracy"     : best["accuracy"],
            "best_f1"           : best["f1"],
            "best_precision"    : best["precision"],
            "best_recall"       : best["recall"],
            "best_composite"    : best["composite"],
            "best_round"        : best["round"],
            "convergence_round" : self.get_convergence_round(),
            "total_rounds"      : len(self.history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMMUNICATION COST TRACKER
# ─────────────────────────────────────────────────────────────────────────────
def compute_comm_cost(model: nn.Module, clients_per_round: int) -> float:
    """
    Estimate communication cost per round in MB.
    Cost = model_size x clients_per_round x 2 (send + receive)
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_mb      = (total_params * 4) / (1024 * 1024)
    return round(size_mb * clients_per_round * 2, 4)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE SUMMARY TABLE (all experiments combined)
# ─────────────────────────────────────────────────────────────────────────────
def save_summary(summaries: list, filename: str = "summary.json"):
    """Save a list of experiment summaries to JSON."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.models import get_model
    from src.preprocess import get_fl_data

    print("=" * 60)
    print("Testing evaluate.py...")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")

    model = get_model(input_dim=input_dim).to(device)

    print("\nEvaluating untrained model (expect ~50% accuracy)...")
    metrics = evaluate(model, test_loader, device)
    print(f"Metrics: {metrics}")
    assert "composite" in metrics, "composite key missing from metrics!"
    assert "precision" in metrics, "precision key missing from metrics!"
    assert "recall"    in metrics, "recall key missing from metrics!"

    print("\nTesting ResultTracker (composite-based best round)...")
    tracker = ResultTracker(algorithm="FedAvg", experiment="baseline")
    # Round 2 has lower accuracy but higher F1/precision/recall
    # so composite should pick round 2 as best
    fake_rounds = [
        {"accuracy": 86.0, "f1": 79.0, "precision": 80.0, "recall": 79.0, "loss": 0.45},
        {"accuracy": 84.0, "f1": 85.0, "precision": 86.0, "recall": 85.0, "loss": 0.40},
        {"accuracy": 85.0, "f1": 82.0, "precision": 83.0, "recall": 82.0, "loss": 0.42},
    ]
    for r, m in enumerate(fake_rounds, start=1):
        m["composite"] = composite_score(m["accuracy"], m["f1"], m["precision"], m["recall"])
        tracker.log(round_num=r, metrics=m)

    summary = tracker.summary()
    print(f"\nSummary: {summary}")
    assert "best_precision" in summary, "best_precision missing from summary!"
    assert "best_recall"    in summary, "best_recall missing from summary!"
    assert "best_composite" in summary, "best_composite missing from summary!"
    assert summary["best_round"] == 2,  "Best round should be 2 (highest composite)!"
    print("Best round correctly selected by composite score.")

    filepath = tracker.save()

    comm_cost = compute_comm_cost(model, clients_per_round=10)
    print(f"\nComm cost per round: {comm_cost} MB")

    print("\nevaluate.py is working correctly!")
