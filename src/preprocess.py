"""
preprocess.py — FL-Privacy-IDS
Loads UNSW-NB15, cleans, encodes, scales, and splits into FL clients.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/unsw_nb15"
TRAIN_CSV       = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_CSV        = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
PROCESSED_DIR   = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLIENTS     = 20
BATCH_SIZE      = 32
DROP_COLS       = ["id", "attack_cat"]
CAT_COLS        = ["proto", "service", "state"]
TARGET_COL      = "label"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean():
    """Load CSVs, drop useless columns, encode categoricals, scale numerics."""
    print("[1/5] Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # Drop irrelevant columns
    train_df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    test_df.drop(columns=DROP_COLS,  inplace=True, errors="ignore")

    print(f"      Train: {train_df.shape} | Test: {test_df.shape}")

    # ── Encode categorical columns ────────────────────────────────────────────
    print("[2/5] Encoding categorical columns...")
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        # fit on combined so test set has no unseen labels
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        encoders[col] = le

    # ── Separate features and labels ─────────────────────────────────────────
    X_train = train_df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.int64)
    X_test  = test_df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_test  = test_df[TARGET_COL].values.astype(np.int64)

    # ── Scale numerical features ──────────────────────────────────────────────
    print("[3/5] Scaling features...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler and encoders for dashboard reuse
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print(f"      Features: {X_train.shape[1]} | "
          f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"      Label distribution — "
          f"Normal: {(y_train==0).sum()} | Attack: {(y_train==1).sum()}")

    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. IID SPLIT — equal random distribution across clients
# ─────────────────────────────────────────────────────────────────────────────
def iid_split(X, y, num_clients=NUM_CLIENTS):
    """Split data equally and randomly across clients (IID)."""
    print(f"[4/5] IID split into {num_clients} clients...")
    indices = np.random.permutation(len(X))
    client_indices = np.array_split(indices, num_clients)
    clients = []
    for i, idx in enumerate(client_indices):
        clients.append((X[idx], y[idx]))
    print(f"      Each client gets ~{len(client_indices[0])} samples")
    return clients


# ─────────────────────────────────────────────────────────────────────────────
# 3. NON-IID SPLIT — Dirichlet distribution
# ─────────────────────────────────────────────────────────────────────────────
def noniid_split(X, y, num_clients=NUM_CLIENTS, alpha=0.5):
    """
    Split data across clients using Dirichlet distribution.
    
    alpha=1.0 → slightly uneven (close to IID)
    alpha=0.5 → moderately uneven
    alpha=0.1 → extremely uneven (worst case Non-IID)
    """
    print(f"[4/5] Non-IID split (α={alpha}) into {num_clients} clients...")
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        np.random.shuffle(class_idx)

        # Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Convert proportions to actual counts
        proportions = (proportions * len(class_idx)).astype(int)

        # Fix rounding so all samples are assigned
        diff = len(class_idx) - proportions.sum()
        proportions[0] += diff

        # Assign indices to clients
        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(class_idx[start:start + count])
            start += count

    clients = []
    for idx in client_indices:
        idx = np.array(idx)
        if len(idx) == 0:
            idx = np.random.choice(len(X), 10, replace=False)
        clients.append((X[idx], y[idx]))

    sizes = [len(c[0]) for c in clients]
    print(f"      Client sizes — min: {min(sizes)} | "
          f"max: {max(sizes)} | avg: {int(np.mean(sizes))}")
    return clients


# ─────────────────────────────────────────────────────────────────────────────
# 4. CREATE DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────
def make_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset  = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def make_client_loaders(clients, batch_size=BATCH_SIZE):
    """Create a DataLoader for each client."""
    return [make_dataloader(X, y, batch_size) for X, y in clients]


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE — returns everything needed for FL training
# ─────────────────────────────────────────────────────────────────────────────
def get_fl_data(mode="iid", alpha=0.5, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE):
    """
    Master function — call this from any experiment file.
    
    Args:
        mode:        "iid" or "noniid"
        alpha:       Dirichlet alpha (only used if mode="noniid")
        num_clients: number of FL clients
        batch_size:  DataLoader batch size
    
    Returns:
        client_loaders: list of DataLoaders (one per client)
        test_loader:    global test DataLoader
        input_dim:      number of features (for model init)
    """
    X_train, y_train, X_test, y_test = load_and_clean()

    if mode == "iid":
        clients = iid_split(X_train, y_train, num_clients)
    else:
        clients = noniid_split(X_train, y_train, num_clients, alpha)

    print("[5/5] Creating DataLoaders...")
    client_loaders = make_client_loaders(clients, batch_size)
    test_loader    = make_dataloader(X_test, y_test, batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f"\n✅ Preprocessing complete!")
    print(f"   Input dimension : {input_dim}")
    print(f"   Clients         : {len(client_loaders)}")
    print(f"   Test samples    : {len(X_test)}")

    return client_loaders, test_loader, input_dim


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify everything works
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing IID split...")
    print("=" * 60)
    client_loaders, test_loader, input_dim = get_fl_data(mode="iid")
    print(f"\nFirst client — batches: {len(client_loaders[0])}")

    print("\n" + "=" * 60)
    print("Testing Non-IID split (α=0.5)...")
    print("=" * 60)
    client_loaders, test_loader, input_dim = get_fl_data(mode="noniid", alpha=0.5)

    print("\n" + "=" * 60)
    print("Testing Non-IID split (α=0.1)...")
    print("=" * 60)
    client_loaders, test_loader, input_dim = get_fl_data(mode="noniid", alpha=0.1)

    print("\n🎉 preprocess.py is working correctly!")
