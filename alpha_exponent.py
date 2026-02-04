#!/usr/bin/env python3
"""
alpha_exponent.py

Implementation corresponding to **Section III-C**
of the WCNC-2026 paper:

    "Detecting Convolutional Codes via a Markovian Statistic"

This script computes the **asymptotic error exponent** for the hypothesis testing
problem between two Markov chains induced by relative Viterbi metrics.

──────────────────────────────────────────────────────────────────────────────
MATHEMATICAL CONTEXT (from the paper)
──────────────────────────────────────────────────────────────────────────────

From Section III-A and III-B, the sequence of relative Viterbi metrics

    D_0, D_1, ..., D_N

forms a finite-state, first-order Markov chain whose transition matrix depends on
the underlying convolutional code.

Let P_1 and P_2 denote the state-transition matrices under hypotheses H_1 and H_2,
respectively. The optimal asymptotic performance of the binary hypothesis test

    H_1 : D_t ~ MC(P_1)
    H_2 : D_t ~ MC(P_2)

is characterized by the **error exponent** I_err, defined as

    I_err = lim_{N→∞} - (1/N) log P_e(N).

For Markov chains, the Chernoff information yields (see Eq. (7) in the paper):

    I_err = min_{u ∈ [0,1]}  [ - log ρ(M(u)) ],                      (Eq. (7))

where ρ(·) denotes the spectral radius, and

    M(u)[i,j] = ∑_r  P_1(i→j, r)^u  ·  P_2(i→j, r)^{1−u}.

Here, r ranges over all possible received channel output vectors at one time step.

This file provides:
  • Empirical estimation of P_1(i→j, r) and P_2(i→j, r) via Monte Carlo simulation
  • Exact numerical computation of Eq. (7)
  • Empirical verification by fitting P_e(N) ≈ A e^{−I_err N}

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from viterbi_markov import (
    octal_to_taps,
    enumerate_markov_states_allzero,
    build_trellis,
    viterbi_metric_step,
    simulate_markov_sequence,
)

# ──────────────────────────────────────────────────────────────────────────────
# Utility: spectral radius
# ──────────────────────────────────────────────────────────────────────────────

def spectral_radius(A):
    """
    Compute the spectral radius ρ(A), i.e., the maximum magnitude eigenvalue.

    This corresponds directly to the quantity appearing in Eq. (7).
    """
    eigvals = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigvals)))


# ──────────────────────────────────────────────────────────────────────────────
# Learning joint transition probabilities P(i→j, r)
# ──────────────────────────────────────────────────────────────────────────────

def learn_transition_tensor(
    encoder_taps,
    decoder_taps,
    m,
    p,
    length=300_000,
    burn_in=5_000,
    laplace=1.0,
    seed=None,
):
    """
    Empirically learn the joint probabilities

        P(i → j, r) = P(D_t = j, Y_t = r | D_{t−1} = i),

    via Monte Carlo simulation, where D_t denotes the relative Viterbi metric
    state and r is the received channel output.

    These quantities are required to form the matrix M(u) in Eq. (7).

    Laplace smoothing is applied to ensure numerical stability.
    """
    if seed is not None:
        np.random.seed(seed)

    # Enumerate Markov states assuming all-zero codeword (Section III-B)
    states, _, all_r = enumerate_markov_states_allzero(decoder_taps, m)
    sidx = {s: i for i, s in enumerate(states)}

    K = len(states)
    R = len(all_r)
    r_index = {r: i for i, r in enumerate(all_r)}

    trellis = build_trellis(decoder_taps, m)

    # Initialize metric state
    cur = tuple([0] * (1 << m))
    cur_i = sidx[cur]

    # Encoder state (integer representation)
    enc_state = 0

    # Count tensor C[i,j,r]
    C = np.zeros((K, K, R), dtype=float)

    # Burn-in phase
    for _ in range(burn_in):
        u = np.random.randint(0, 2)
        y, enc_state = _encoder_step(enc_state, u, encoder_taps, m)
        r = tuple(bit ^ np.random.binomial(1, p) for bit in y)
        cur = viterbi_metric_step(list(cur), trellis, r)
        cur_i = sidx.get(cur, cur_i)

    # Main Monte Carlo loop
    for _ in range(length):
        u = np.random.randint(0, 2)
        y, enc_state = _encoder_step(enc_state, u, encoder_taps, m)
        r = tuple(bit ^ np.random.binomial(1, p) for bit in y)
        nxt = viterbi_metric_step(list(cur), trellis, r)
        nxt_i = sidx.get(nxt, None)

        if nxt_i is not None:
            C[cur_i, nxt_i, r_index[r]] += 1.0
            cur, cur_i = nxt, nxt_i

    # Laplace smoothing and normalization
    C += laplace
    C /= np.maximum(C.sum(axis=(1, 2), keepdims=True), 1.0)

    return C, states, sidx, all_r


# ──────────────────────────────────────────────────────────────────────────────
# Chernoff information and Eq. (7)
# ──────────────────────────────────────────────────────────────────────────────

def compute_error_exponent(P1_ijr, P2_ijr, u_grid=401):
    """
    Compute the error exponent I_err using Eq. (7):

        I_err = min_{u ∈ [0,1]} [ − log ρ(M(u)) ],

    where

        M(u)[i,j] = ∑_r P_1(i→j,r)^u · P_2(i→j,r)^{1−u}.
    """
    P1 = np.clip(P1_ijr, 1e-300, 1.0)
    P2 = np.clip(P2_ijr, 1e-300, 1.0)

    u_vals = np.linspace(0.0, 1.0, u_grid)
    best_rho = None
    best_u = None

    for u in u_vals:
        M = np.sum((P1 ** u) * (P2 ** (1.0 - u)), axis=2)
        rho = max(spectral_radius(M), 1e-300)

        if best_rho is None or rho < best_rho:
            best_rho = rho
            best_u = u

    return float(-np.log(best_rho)), float(best_u)


# ──────────────────────────────────────────────────────────────────────────────
# Empirical tail fitting: P_e(N) ≈ A e^{−I_err N}
# ──────────────────────────────────────────────────────────────────────────────

def fit_error_exponent(N_vals, P_e_vals, tail_cap=0.2):
    """
    Empirically estimate I_err by fitting

        P_e(N) ≈ A e^{−I_err N}

    on the low-error tail, consistent with large-deviations theory.
    """
    N = np.asarray(N_vals, dtype=float)
    P_e = np.asarray(P_e_vals, dtype=float)

    mask = (P_e > 0) & (P_e <= tail_cap)
    if np.sum(mask) < 3:
        return 0.0, np.nan

    y = np.log(P_e[mask])
    X = np.vstack([np.ones_like(N[mask]), -N[mask]]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    A = np.exp(beta[0])
    I_emp = beta[1]

    return float(I_emp), float(A)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: single encoder step
# ──────────────────────────────────────────────────────────────────────────────

def _encoder_step(state, u, taps, m):
    """
    One encoder update for a rate-1 convolutional code.

    This helper exists solely to keep the main learning loop readable.
    """
    state_bits = [(state >> i) & 1 for i in range(m)]
    x = [u] + state_bits

    y = []
    for g in taps:
        y.append(sum((g[i] & x[i]) for i in range(len(g))) % 2)

    next_state = ((u << (m - 1)) | (state >> 1)) if m > 0 else 0
    return tuple(y), next_state
