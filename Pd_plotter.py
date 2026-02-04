#!/usr/bin/env python3
"""
Pd_plotter.py

Implementation of the **hybrid Markov-based
convolutional code detector** used for the *main numerical results* in the
WCNC-2026 paper:

    "Detecting Convolutional Codes via a Markovian Statistic"

This file corresponds primarily to **Section V (Numerical Results)** of the
paper and implements the *hybrid detection framework* described therein.

──────────────────────────────────────────────────────────────────────────────
MATHEMATICAL CONTEXT (hybrid detector)
──────────────────────────────────────────────────────────────────────────────

From Sections III-A and III-B, the relative Viterbi metric sequence

    D_0, D_1, …, D_N

forms a finite-state Markov chain whose transition matrix depends on the
underlying convolutional code.

Let:
  • P_1  be the true (unknown) transition matrix under H₁
  • T(p) be the analytically derived transition matrix under H₂,
    evaluated at p = 1/2 (uninformative reference model)

The hybrid detector uses:
  • an *empirical estimate* \hat{P}_1 learned from data, and
  • a *symbolic reference* T(p=1/2)

and performs a likelihood comparison on observed metric sequences.

Given an observed metric trajectory {D_t}_{t=0}^N, the test statistic is

    Λ_N = log P_{\hat{P}_1}(D_0^N) − log P_{T(1/2)}(D_0^N).

The decision rule is:

    decide H₁ if Λ_N > 0,
    decide H₂ otherwise.

This detector is **parameter-free** (no thresholds) and converges
exponentially fast with exponent characterized by Eq. (7).
"""

import os
import time
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import viterbi_markov as vm

# ──────────────────────────────────────────────────────────────────────────────
# Experiment defaults (as used in the paper unless stated otherwise)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "num_iter": 10000,          # Monte-Carlo trials per (N, p)
    "p_vec": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    "seed": 12345,
    "learn_len": None,          # adaptive if None
    "learn_burn": 200,
    "laplace": 1.0,
    "save_dir": "results_experiments",
}

# Blocklength spectra used in the paper (depend on memory m)
N_SPECTRUM_BY_M = {
    1: [5, 10, 20, 50, 100, 200],
    2: [500],
    3: [500],
    4: [50, 100, 200, 300, 500],
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility: evaluate symbolic transition matrix
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_symbolic_T(T_sym, p_sym, p_val):
    """
    Evaluate a symbolic transition matrix T(p) at a fixed p value and
    renormalize rows to guard against numerical error.
    """
    T_num = np.array([[float(sp.N(x.subs(p_sym, p_val))) for x in row]
                      for row in T_sym.tolist()])

    row_sums = T_num.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T_num / row_sums


# ──────────────────────────────────────────────────────────────────────────────
# Likelihood of a metric sequence under a Markov model
# ──────────────────────────────────────────────────────────────────────────────

def log_prob_sequence(metrics, state_index, T):
    """
    Compute log P(D_0^N) under a Markov chain with transition matrix T.
    """
    logp = 0.0
    for t in range(len(metrics) - 1):
        i = state_index[metrics[t]]
        j = state_index[metrics[t + 1]]
        pij = max(T[i, j], 1e-300)
        logp += math.log(pij)
    return logp


# ──────────────────────────────────────────────────────────────────────────────
# Empirical learning of P₁ (cached)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def learn_P1_empirical(
    gens_tuple, k, n, m, p,
    learn_len, learn_burn, laplace, seed
):
    """
    Learn the empirical transition matrix \hat{P}_1 for hypothesis H₁
    via long Monte-Carlo simulation.

    This approximates the true Markov transition matrix appearing in Eq. (7).
    """
    generator_matrix = list(gens_tuple)

    states, transitions, _ = vm.enumerate_markov_states_allzero(
        generator_matrix, m, k, n
    )
    state_index = {s: i for i, s in enumerate(states)}
    S = len(states)

    # Choose learning length proportional to state size
    if learn_len is None:
        learn_len_eff = max(5000, 200 * S)
    else:
        learn_len_eff = learn_len

    # Simulate long metric sequence
    sim = vm.simulate_markov_sequence(
        generator_matrix, m, k, n,
        learn_len_eff,
        p_val=p,
        random_input=True,
        seed=seed,
    )

    metrics = sim["metrics"]
    counts = np.zeros((S, S))

    for t in range(learn_burn, len(metrics) - 1):
        i = state_index[metrics[t]]
        j = state_index[metrics[t + 1]]
        counts[i, j] += 1.0

    # Laplace smoothing and normalization
    P = counts + laplace
    P /= P.sum(axis=1, keepdims=True)

    return states, state_index, P


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (Section V)
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    k, n, m,
    gen1, gen2,
    num_iter, p_vec,
    learn_len, learn_burn, laplace,
    seed,
):
    """
    Run the hybrid detector for all (N, p) pairs and return results as a DataFrame.
    """
    results = []

    # Enumerate Markov states once (decoder is fixed to H₁)
    states, _, _ = vm.enumerate_markov_states_allzero(gen1, m, k, n)
    state_index = {s: i for i, s in enumerate(states)}

    # Build symbolic reference matrix T(p)
    p_sym, T_sym = vm.build_symbolic_T(*vm.enumerate_markov_states_allzero(gen1, m, k, n))
    T_ref = evaluate_symbolic_T(T_sym, p_sym, 0.5)

    N_spectrum = N_SPECTRUM_BY_M.get(m, [50, 100, 200])

    for N in N_spectrum:
        for p in p_vec:
            success_h1 = 0
            success_h2 = 0

            # Learn empirical P₁ for this p
            states_L, sidx_L, P1 = learn_P1_empirical(
                tuple(tuple(tuple(x) for x in row) for row in gen1),
                k, n, m, p,
                learn_len, learn_burn, laplace, seed
            )

            for _ in tqdm(range(num_iter), desc=f"N={N}, p={p}"):
                # H₁ trial
                sim1 = vm.simulate_markov_sequence(gen1, m, k, n, N, p, True)
                logp1 = log_prob_sequence(sim1["metrics"], sidx_L, P1)
                logp1_ref = log_prob_sequence(sim1["metrics"], sidx_L, T_ref)
                if logp1 > logp1_ref:
                    success_h1 += 1

                # H₂ trial
                sim2 = vm.simulate_markov_sequence(gen2, m, k, n, N, p, True)
                logp2 = log_prob_sequence(sim2["metrics"], sidx_L, P1)
                logp2_ref = log_prob_sequence(sim2["metrics"], sidx_L, T_ref)
                if logp2 <= logp2_ref:
                    success_h2 += 1

            Pd = success_h1 / num_iter
            Pc = (success_h1 + success_h2) / (2 * num_iter)

            results.append({
                "N": N,
                "p": p,
                "Pd": Pd,
                "Pc": Pc,
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hybrid Markov-based detector (WCNC-2026)")

    # Example parameters (rate-1/2, memory m=2)
    k, n, m = 1, 2, 2
    gen1 = [[[1, 1, 1]], [[1, 0, 1]]]
    gen2 = [[[1, 1, 0]], [[1, 0, 1]]]

    df = run_experiment(
        k, n, m,
        gen1, gen2,
        DEFAULTS["num_iter"],
        DEFAULTS["p_vec"],
        DEFAULTS["learn_len"],
        DEFAULTS["learn_burn"],
        DEFAULTS["laplace"],
        DEFAULTS["seed"],
    )

    os.makedirs(DEFAULTS["save_dir"], exist_ok=True)
    out_csv = os.path.join(DEFAULTS["save_dir"], "Pd_hybrid_results.csv")
    df.to_csv(out_csv, index=False)
    print("Saved results to", out_csv)
