#!/usr/bin/env python3
"""
plots_compare.py

Conference-ready, fully annotated **post-processing and visualization** script
used in the WCNC-2026 paper:

    "Detecting Convolutional Codes via a Markovian Statistic"

This file generates the **comparative figures** shown in Section V, where the
proposed hybrid Markov-based detector is compared against the parity-template
baseline.

──────────────────────────────────────────────────────────────────────────────
ROLE IN THE PAPER (important clarification)
──────────────────────────────────────────────────────────────────────────────

• This script performs **no detection and no inference**.
• It consumes CSV outputs produced by `Pd_plotter.py` (proposed method)
  and `comp_parity.py` (baseline method).
• Its sole purpose is **visualization and fair comparison**.

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Helper: probability of error
# ──────────────────────────────────────────────────────────────────────────────

def p_error(Pc):
    """
    Compute probability of error from probability of correct decision:

        P_error = 1 − P_c.
    """
    return np.clip(1.0 - Pc, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Curve extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_by_N(df, N, x_col="p", y_col="Pc"):
    """Extract (x, y) curve at fixed blocklength N."""
    q = df[df["N"] == int(N)]
    if q.empty:
        return np.array([]), np.array([])
    q = q.sort_values(by=x_col)
    return q[x_col].to_numpy(), q[y_col].to_numpy()


def extract_by_p(df, p, x_col="N", y_col="Pc"):
    """Extract (x, y) curve at fixed channel crossover probability p."""
    q = df[np.isclose(df["p"], p)]
    if q.empty:
        return np.array([]), np.array([])
    q = q.sort_values(by=x_col)
    return q[x_col].to_numpy(), q[y_col].to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────

def main(hybrid_csv, baseline_csv, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Load results
    df_h = pd.read_csv(hybrid_csv)
    df_b = pd.read_csv(baseline_csv)

    # Normalize column naming
    if "Pc" not in df_h.columns and "Pd" in df_h.columns:
        df_h["Pc"] = df_h["Pd"]
    if "Pc" not in df_b.columns and "Pd" in df_b.columns:
        df_b["Pc"] = df_b["Pd"]

    Ns = sorted(set(df_h["N"]).union(df_b["N"]))
    ps = sorted(set(df_h["p"]).union(df_b["p"]))

    # ──────────────────────────────────────────────────────────────────────────
    # P_error vs p (fixed N)
    # ──────────────────────────────────────────────────────────────────────────

    for N in Ns:
        plt.figure(figsize=(6, 5))

        xh, yh = extract_by_N(df_h, N)
        xb, yb = extract_by_N(df_b, N)

        if len(xh):
            plt.plot(xh, p_error(yh), marker='o', label=f"Hybrid (N={N})")
        if len(xb):
            plt.plot(xb, p_error(yb), marker='s', linestyle='--', label=f"Parity baseline (N={N})")

        plt.xlabel("BSC crossover probability p")
        plt.ylabel("Probability of error $P_{\\mathrm{err}}$")
        plt.title(f"$P_{{\\mathrm{{err}}}}$ vs $p$ (N={N})")
        plt.grid(True)
        plt.legend()

        fname = os.path.join(outdir, f"Perr_vs_p_N{N}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

    # ──────────────────────────────────────────────────────────────────────────
    # P_error vs N (fixed p)
    # ──────────────────────────────────────────────────────────────────────────

    for p in ps:
        plt.figure(figsize=(6, 5))

        xh, yh = extract_by_p(df_h, p)
        xb, yb = extract_by_p(df_b, p)

        if len(xh):
            plt.plot(xh, p_error(yh), marker='o', label=f"Hybrid (p={p})")
        if len(xb):
            plt.plot(xb, p_error(yb), marker='s', linestyle='--', label=f"Parity baseline (p={p})")

        plt.xlabel("Blocklength N")
        plt.ylabel("Probability of error $P_{\\mathrm{err}}$")
        plt.title(f"$P_{{\\mathrm{{err}}}}$ vs $N$ (p={p})")
        plt.grid(True)
        plt.legend()

        fname = os.path.join(outdir, f"Perr_vs_N_p{p}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare hybrid and parity-template detectors")
    parser.add_argument("--hybrid", required=True, help="CSV from Pd_plotter.py")
    parser.add_argument("--baseline", required=True, help="CSV from comp_parity.py")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    main(args.hybrid, args.baseline, args.outdir)
