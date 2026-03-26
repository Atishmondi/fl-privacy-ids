"""
models.py — FL-Privacy-IDS
Defines the MLP neural network used by all FL clients.
Upgraded architecture: 42 -> 512 -> 256 -> 128 -> 64 -> 2
"""

import torch
import torch.nn as nn

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# MLP MODEL
# ─────────────────────────────────────────────────────────────────────────────
class IDSModel(nn.Module):
    def __init__(self, input_dim: int = 42, num_classes: int = 2):
        super(IDSModel, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(256, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_model(input_dim: int = 42, num_classes: int = 2) -> IDSModel:
    """Return a fresh untrained model instance."""
    return IDSModel(input_dim=input_dim, num_classes=num_classes)


def get_model_weights(model: IDSModel) -> dict:
    """Extract model weights as a state dict (for FL aggregation)."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def set_model_weights(model: IDSModel, weights: dict) -> IDSModel:
    """Load weights into a model (after FL aggregation)."""
    model.load_state_dict(weights)
    return model


def count_parameters(model: IDSModel) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: IDSModel) -> float:
    """Estimate model size in MB (for communication cost tracking)."""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)
    return round(size_mb, 4)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing IDSModel...")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = get_model(input_dim=42).to(device)
    print(f"\nModel Architecture:\n{model}")
    print(f"\nTotal Parameters : {count_parameters(model):,}")
    print(f"Model Size       : {get_model_size_mb(model)} MB")

    dummy_input = torch.randn(32, 42).to(device)
    output = model(dummy_input)
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    assert output.shape == (32, 2), "Output shape mismatch!"

    weights = get_model_weights(model)
    new_model = get_model(input_dim=42).to(device)
    new_model = set_model_weights(new_model, weights)
    print(f"\nWeight extraction & loading : OK")

    print("\nmodels.py is working correctly!")
