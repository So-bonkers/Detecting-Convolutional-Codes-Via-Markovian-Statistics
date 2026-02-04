#!/usr/bin/env python3
"""
viterbi_markov.py

Annotated implementation corresponding to the WCNC-2026 paper:

    "Detecting Convolutional Codes via a Markovian Statistic"

This file implements the construction of the **relative Viterbi-metric Markov
chain** described in **Section III-A and III-B** of the paper.

──────────────────────────────────────────────────────────────────────────────
MATHEMATICAL CONTEXT (from the paper)
──────────────────────────────────────────────────────────────────────────────

Let the received sequence be Y_1^N over a BSC(p). When decoding using a candidate
convolutional code via the Viterbi algorithm, define the *relative Viterbi metric*
vector at time t as

    D_t = (D_t(1), D_t(2), …, D_t(2^m)),

where each component corresponds to a trellis state.

For each state i at time t, the un-normalized metric is

    D'_t(i) = min_{s ∈ S_t(i)} { D_{t−1}(s) + d_H(Y_t, V_t(s)) }      (Eq. (4))

and the normalized (relative) metric is

    D_t(i) = D'_t(i) − min_j D'_t(j).                                 (Eq. (5))

As shown in Section III-A, the sequence {D_t} forms a **finite-state, first-order
Markov chain**. Under a BSC(p), the transition probabilities are

    P(D_t = d_t | D_{t−1} = d_{t−1})
      = ∑_{y_t ∈ Y(d_{t−1}, d_t)} p^{w(y_t)} (1−p)^{n−w(y_t)}.          (Eq. (6))

This file provides code to:
  • Enumerate all reachable metric states D_t
  • Enumerate the sets Y(d_{t−1}, d_t)
  • Construct symbolic and numeric transition matrices T(p)

These objects are later used for hypothesis testing and error-exponent
computation (Section III-C, Eq. (7)), implemented in alpha_exponent.py.
"""

import itertools
import math
from collections import defaultdict, deque
import numpy as np
import sympy as sp
import json
import csv
import os

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions for trellis and state handling
# ──────────────────────────────────────────────────────────────────────────────

def state_bits_from_int(state_int, m):
    """
    Convert an encoder state (integer) into its binary register contents.

    The state is represented using m memory elements, consistent with the
    convolutional encoder description in Section II of the paper.
    """
    return [(state_int >> i) & 1 for i in range(m)]


def bits_to_int(bit_list):
    """Convert a list of bits (LSB-first) into an integer."""
    val = 0
    for i, b in enumerate(bit_list):
        val |= (int(b) & 1) << i
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Encoder branch computation
# ──────────────────────────────────────────────────────────────────────────────

def branch_output_and_next_state(state_int, input_bits, generator_matrix, m, k):
    """
    Compute encoder branch output and next state.

    This corresponds to one trellis branch in the convolutional encoder.
    The operation is standard convolution over GF(2).
    """
    state_bits = state_bits_from_int(state_int, m)
    outputs = []

    for gen in generator_matrix:
        output_bit = 0
        for i in range(k):
            x = [input_bits[i]] + state_bits[:m]
            taps = gen[i]
            for j in range(min(len(taps), len(x))):
                output_bit ^= (taps[j] & x[j])
        outputs.append(output_bit)

    # Shift register update
    new_regs = list(input_bits) + state_bits[:max(0, m - k)] if m > 0 else []
    new_regs = new_regs[:m]
    next_state = bits_to_int(new_regs)

    return tuple(outputs), next_state


def hamming_distance(a, b):
    """Hamming distance d_H(a, b) used in Eq. (4)."""
    return sum(x != y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Trellis construction
# ──────────────────────────────────────────────────────────────────────────────

def build_trellis(generator_matrix, m, k):
    """
    Build the Viterbi trellis: for each state, list all incoming branches.

    This trellis is used to compute the minimization in Eq. (4).
    """
    num_states = 1 << m
    incoming = {s: [] for s in range(num_states)}

    for s in range(num_states):
        for u in itertools.product([0, 1], repeat=k):
            out, ns = branch_output_and_next_state(s, u, generator_matrix, m, k)
            incoming[ns].append((s, u, out))

    return incoming


# ──────────────────────────────────────────────────────────────────────────────
# One Viterbi metric update (Eq. (4) and Eq. (5))
# ──────────────────────────────────────────────────────────────────────────────

def viterbi_metric_step(D_prev, trellis, y_t):
    """
    Perform one Viterbi metric update.

    Implements:
      D'_t(i) = min_s { D_{t−1}(s) + d_H(y_t, V_t(s)) }      (Eq. (4))
      D_t(i)  = D'_t(i) − min_j D'_t(j)                      (Eq. (5))
    """
    num_states = len(D_prev)
    D_next = [math.inf] * num_states

    for ns in range(num_states):
        best = math.inf
        for (ps, _, out) in trellis[ns]:
            val = D_prev[ps] + hamming_distance(out, y_t)
            if val < best:
                best = val
        D_next[ns] = best

    min_val = min(D_next)
    return tuple(int(v - min_val) for v in D_next)


# ──────────────────────────────────────────────────────────────────────────────
# Enumeration of Markov states (Section III-B)
# ──────────────────────────────────────────────────────────────────────────────

def enumerate_markov_states_allzero(generator_matrix, m, k, n):
    """
    Enumerate all reachable relative-metric states assuming the all-zero
    codeword is transmitted.

    This uses the linearity argument described in Section III-B and leads
    to the transition probabilities in Eq. (6).
    """
    trellis = build_trellis(generator_matrix, m, k)
    all_r = list(itertools.product([0, 1], repeat=n))

    start = tuple([0] * (1 << m))
    queue = deque([start])
    visited = {start: 0}
    states = [start]
    transitions = defaultdict(lambda: defaultdict(list))

    while queue:
        cur = queue.popleft()
        cur_idx = visited[cur]

        for r in all_r:
            nxt = viterbi_metric_step(list(cur), trellis, r)
            if nxt not in visited:
                visited[nxt] = len(states)
                states.append(nxt)
                queue.append(nxt)
            transitions[cur_idx][visited[nxt]].append(r)

    return states, transitions, all_r


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic transition matrix T(p) (Eq. (6))
# ──────────────────────────────────────────────────────────────────────────────

def build_symbolic_T(states, transitions, all_r, normalize=True):
    """
    Construct the symbolic Markov transition matrix T(p).

    Each entry T_{i,j}(p) equals

        ∑_{r ∈ Y(i,j)} p^{w(r)} (1−p)^{n−w(r)}                (Eq. (6))

    where w(r) is the Hamming weight of r.
    """
    p = sp.symbols('p')
    S = len(states)
    n = len(all_r[0])

    T = sp.MutableDenseMatrix(S, S, lambda i, j: 0)
    prob_r = {r: p**sum(r) * (1 - p)**(n - sum(r)) for r in all_r}

    for i in range(S):
        row_sum = 0
        for j, rlist in transitions[i].items():
            val = sum(prob_r[r] for r in rlist)
            T[i, j] = sp.simplify(val)
            row_sum += val

        if normalize and row_sum != 0:
            for j in range(S):
                T[i, j] = sp.simplify(T[i, j] / row_sum)

    return p, sp.simplify(T)
