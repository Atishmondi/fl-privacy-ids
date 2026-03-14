"""
dp_wrapper.py — FL-Privacy-IDS
Wraps a model and optimizer with Opacus Differential Privacy (DP-SGD).
Note: Opacus only works on CPU — device is forced to CPU here.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── DP Defaults ───────────────────────────────────────────────────────────────
MAX_GRAD_NORM = 1.0
DELTA         = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE & FIX MODEL FOR OPACUS
# ─────────────────────────────────────────────────────────────────────────────
def make_dp_compatible(model: nn.Module) -> nn.Module:
    """
    Opacus does not support BatchNorm1d — replace with GroupNorm automatically.
    This keeps the model architecture intact but makes it DP-compatible.
    """
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ATTACH DP TO MODEL + OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def attach_dp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    target_epsilon: float,
    target_delta: float = DELTA,
    max_grad_norm: float = MAX_GRAD_NORM,
    epochs: int = 10,
) -> tuple:
    # Force CPU — Opacus does not support MPS
    model = model.cpu()

    # Make model DP-compatible FIRST
    model = make_dp_compatible(model)

    # Recreate optimizer with fixed model's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Attach PrivacyEngine
    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    return dp_model, dp_optimizer, dp_dataloader, privacy_engine


# ─────────────────────────────────────────────────────────────────────────────
# GET CURRENT EPSILON (after training)
# ─────────────────────────────────────────────────────────────────────────────
def get_epsilon(privacy_engine: PrivacyEngine, delta: float = DELTA) -> float:
    """Get the actual epsilon spent so far during training."""
    return privacy_engine.get_epsilon(delta=delta)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.models import get_model
    from src.preprocess import get_fl_data

    print("=" * 60)
    print("Testing DP Wrapper...")
    print("=" * 60)

    # Load data
    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")
    first_loader = client_loaders[0]

    # Create model and optimizer
    model     = get_model(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nOriginal model valid for Opacus: {ModuleValidator.is_valid(model)}")

    # Attach DP with epsilon=10 (light privacy)
    print("\nAttaching DP (ε=10)...")
    dp_model, dp_optimizer, dp_loader, engine = attach_dp(
        model=model,
        optimizer=optimizer,
        dataloader=first_loader,
        target_epsilon=10.0,
        epochs=10,
    )

    print(f"DP model on device : {next(dp_model.parameters()).device}")
    print(f"DP-compatible model: {ModuleValidator.is_valid(dp_model)}")

    # Run one training step to verify
    criterion = nn.CrossEntropyLoss()
    dp_model.train()
    X_batch, y_batch = next(iter(dp_loader))
    X_batch, y_batch = X_batch.cpu(), y_batch.cpu()

    dp_optimizer.zero_grad()
    output = dp_model(X_batch)
    loss   = criterion(output, y_batch)
    loss.backward()
    dp_optimizer.step()

    epsilon_spent = get_epsilon(engine)
    print(f"\nOne step complete!")
    print(f"Loss          : {loss.item():.4f}")
    print(f"Epsilon spent : {epsilon_spent:.4f}")

    print("\n🎉 dp_wrapper.py is working correctly!")
