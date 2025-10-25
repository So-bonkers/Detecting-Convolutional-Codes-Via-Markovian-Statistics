#!/usr/bin/env python3
"""
sidebyside_plots.py

Modified to:
 - Use log scale on the y-axis (P_error plots).
 - Remove all P_error_log computations/plots entirely.

Plots:
 - P_error vs p (for each threshold, ours vs literature, for various N)
 - P_error vs N (for each threshold, ours vs literature, for various p)

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Defaults ----------
DEFAULT_HYBRID = "path of the saved hybrid results csv"
DEFAULT_LIT    = "path of lit results csv"
DEFAULT_SAVE_DIR = "path of save directory"
P_ROUND_DECIMALS = 6
# -----------------------------

def p_error_linear(p_correct_series):
    """Linear P_error = 1 - P_correct; clipped to [0, 1]"""
    val = 1.0 - p_correct_series
    return np.clip(val, 0.0, 1.0)

def extract_curve_by_N(df, N, x_col='p_rnd', y_col='Pc'):
    q = df[df['N'] == int(N)]
    if q.empty:
        return np.array([]), np.array([])
    q_sorted = q.sort_values(by=x_col)
    xs = q_sorted[x_col].to_numpy()
    ys = q_sorted[y_col].to_numpy()
    return xs, ys

def extract_curve_by_p(df, p_rnd, x_col='N', y_col='Pc'):
    q = df[np.isclose(df['p_rnd'], p_rnd)]
    if q.empty:
        return np.array([]), np.array([])
    q_sorted = q.sort_values(by=x_col)
    xs = q_sorted[x_col].to_numpy()
    ys = q_sorted[y_col].to_numpy()
    return xs, ys

def main(hybrid_csv, lit_csv, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # --------- Load data ----------
    df_h = pd.read_csv(hybrid_csv)
    df_l = pd.read_csv(lit_csv)

    if 'Pc' not in df_h.columns:
        if 'Pd' in df_h.columns:
            print("Note: hybrid CSV missing 'Pc' column — using 'Pd' as probability-correct.")
            df_h['Pc'] = df_h['Pd']
        else:
            raise RuntimeError("Hybrid CSV must contain 'Pc' or 'Pd' column.")

    if 'Pd_mean' not in df_l.columns:
        raise RuntimeError("Literature CSV must contain 'Pd_mean' column.")

    df_h['p_rnd'] = df_h['p'].round(P_ROUND_DECIMALS)
    df_l['p_rnd'] = df_l['p'].round(P_ROUND_DECIMALS)

    df_h['N'] = df_h['N'].astype(int)
    df_l['N'] = df_l['N'].astype(int)

    df_l = df_l.rename(columns={'Pd_mean': 'Pc_lit'})
    thresholds = sorted(df_l['threshold'].unique())
    print("Found thresholds:", thresholds)

    for th in thresholds:
        print(f"\nProcessing threshold = {th}")
        df_l_th = df_l[df_l['threshold'] == th]

        # ========== P_error vs p (for each N) ==========
        Ns_union = sorted(set(df_h['N'].unique()).union(set(df_l_th['N'].unique())))
        plt.figure(figsize=(6, 5))

        for idx, N in enumerate(Ns_union):
            # ours
            xs_h, ys_h = extract_curve_by_N(df_h, N, x_col='p_rnd', y_col='Pc')
            if len(xs_h) > 0:
                plt.plot(xs_h, p_error_linear(pd.Series(ys_h)),
                         label=f"ours N={N}", linestyle='solid')
            # literature
            xs_l, ys_l = extract_curve_by_N(df_l_th, N, x_col='p_rnd', y_col='Pc_lit')
            if len(xs_l) > 0:
                plt.plot(xs_l, p_error_linear(pd.Series(ys_l)),
                         label=f"lit N={N}", linestyle='dotted')

        plt.xlabel("p")
        plt.ylabel("P_error")
        plt.title(f"P_error vs p (threshold={th})")
        # plt.yscale("log")   # <<<<<<<<<< log scale here
        plt.grid(True, which="both")
        plt.legend()
        fname = os.path.join(save_dir, f"Perror_vs_p_threshold_{th:.6f}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", fname)

        # ========== P_error vs N (for each p) ==========
        p_union = sorted(set(df_h['p_rnd'].unique()).union(set(df_l_th['p_rnd'].unique())))
        plt.figure(figsize=(6, 5))

        for idx, pval in enumerate(p_union):
            xs_h, ys_h = extract_curve_by_p(df_h, pval, x_col='N', y_col='Pc')
            if len(xs_h) > 0:
                plt.plot(xs_h, p_error_linear(pd.Series(ys_h)),
                         label=f"ours p={pval}", linestyle='solid')
            xs_l, ys_l = extract_curve_by_p(df_l_th, pval, x_col='N', y_col='Pc_lit')
            if len(xs_l) > 0:
                plt.plot(xs_l, p_error_linear(pd.Series(ys_l)),
                         label=f"lit p={pval}", linestyle='dotted')

        plt.xlabel("N")
        plt.ylabel("P_error")
        plt.title(f"P_error vs N (threshold={th})")
        # plt.yscale("log")   # <<<<<<<<<< log scale here
        plt.grid(True, which="both")
        plt.legend()
        fname = os.path.join(save_dir, f"Perror_vs_N_threshold_{th:.6f}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare P_error vs p and vs N (log scale y-axis).")
    parser.add_argument("--hybrid", type=str, default=DEFAULT_HYBRID, help="Path to hybrid CSV (ours).")
    parser.add_argument("--lit", type=str, default=DEFAULT_LIT, help="Path to literature CSV.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_SAVE_DIR, help="Directory to save output plots.")
    args = parser.parse_args()
    main(args.hybrid, args.lit, args.outdir)
