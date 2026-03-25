"""
analyze_results.py — FL-Privacy-IDS
Reads all result CSVs and prints a complete stats table.
Run this after all experiments are done to get paper-ready numbers.
"""

import pandas as pd
import glob
import os
import json

RESULTS_DIR = "results"


def analyze(pattern: str, label: str):
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Algorithm':<12} {'Best':>8} {'Worst':>8} {'Mean':>8} {'Std':>8} {'BestRnd':>8} {'WorstRnd':>9}")
    print("-" * 70)

    rows = []
    for f in sorted(files):
        df = pd.read_csv(f)
        algo = df["algorithm"].iloc[0]
        rows.append({
            "algo"      : algo,
            "best"      : df["accuracy"].max(),
            "worst"     : df["accuracy"].min(),
            "mean"      : df["accuracy"].mean(),
            "std"       : df["accuracy"].std(),
            "best_rnd"  : int(df.loc[df["accuracy"].idxmax(), "round"]),
            "worst_rnd" : int(df.loc[df["accuracy"].idxmin(), "round"]),
            "best_composite": df["composite"].max() if "composite" in df.columns else None,
        })

    # Sort by best accuracy descending
    rows.sort(key=lambda x: x["best"], reverse=True)
    for r in rows:
        print(f"{r['algo']:<12} {r['best']:>7.4f}% {r['worst']:>7.4f}% "
              f"{r['mean']:>7.4f}% {r['std']:>7.4f}% "
              f"{r['best_rnd']:>8} {r['worst_rnd']:>9}")

    return rows


if __name__ == "__main__":
    print("\nFL-PRIVACY-IDS — COMPLETE RESULTS ANALYSIS")

    # Baseline
    analyze("*baseline_iid*.csv", "EXPERIMENT 1 — Baseline (IID)")

    # Non-IID
    for alpha in [1.0, 0.5, 0.1]:
        analyze(f"*noniid_alpha{alpha}*.csv", f"EXPERIMENT 2 — Non-IID (alpha={alpha})")

    # DP (after E3 is done)
    for eps in ["1", "10", "inf"]:
        files = glob.glob(os.path.join(RESULTS_DIR, f"*dp_eps{eps}*.csv"))
        if files:
            analyze(f"*dp_eps{eps}*.csv", f"EXPERIMENT 3 — DP (epsilon={eps})")

    print("\nDone!")
