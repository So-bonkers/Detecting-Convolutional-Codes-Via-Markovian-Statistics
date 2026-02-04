#!/usr/bin/env python3
"""
parity_eqn_check.py

Implementation corresponding to the **parity-template baseline** used for comparison in **Section IV** 
of the WCNC-2026 paper:

    "Detecting Convolutional Codes via a Markovian Statistic"

This file derives **exact parity-check equations** for a convolutional encoder
from its generator polynomials and expresses them both in polynomial form and in
time-domain form. These equations are later used to construct *parity templates*
for hypothesis testing, following the literature baseline (e.g., Moosavi &
Larsson, GLOBECOM 2011).

──────────────────────────────────────────────────────────────────────────────
MATHEMATICAL CONTEXT (baseline method)
──────────────────────────────────────────────────────────────────────────────

Consider a rate k/n feed-forward convolutional encoder with generator matrix

    G(D) = [ g_{j,i}(D) ] ,     j = 1,…,n ,  i = 1,…,k

where g_{j,i}(D) are binary polynomials of degree ≤ m.

A parity-check polynomial vector

    h(D) = ( h_1(D), …, h_n(D) )

satisfies

    ∑_{j=1}^n h_j(D) v_j(D) = 0    (over GF(2)),

for *all* valid codewords v(D). Expanding in time yields parity equations of the
form

    ⊕_{(j,s) ∈ S} v_j[t − s] = 0,                                  (P-Eq)

where S is a finite set of output-index / delay pairs.

In Section IV of the paper, such equations are used as *templates*: the fraction
of time indices t for which the noisy received sequence satisfies (P-Eq) is used
as a test statistic for code detection.

This file performs:
  • Construction of the linear system defining parity-check polynomials
  • Nullspace computation over GF(2)
  • Conversion to explicit time-domain parity equations

"""

import re
import numpy as np
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Polynomial parsing utilities
# ──────────────────────────────────────────────────────────────────────────────

def parse_poly_token(token: str) -> List[int]:
    """
    Parse a generator polynomial token into a binary coefficient list.

    Accepted formats:
      • Octal (e.g., "7", "13")
      • Binary MSB-first string (e.g., "111")
      • Explicit comma-separated list (e.g., "1,0,1")

    Output is **LSB-first**, i.e., coefficient of D^0 first.
    """
    token = token.strip()

    # Explicit comma-separated taps
    if ',' in token:
        return [int(x) for x in token.split(',')]

    # Binary string
    if re.fullmatch(r"[01]+", token):
        return [int(b) for b in token[::-1]]

    # Octal representation
    if re.fullmatch(r"[0-7]+", token):
        val = int(token, 8)
        return [(val >> i) & 1 for i in range(val.bit_length())]

    raise ValueError(f"Cannot parse polynomial token: {token}")


# ──────────────────────────────────────────────────────────────────────────────
# Linear algebra over GF(2)
# ──────────────────────────────────────────────────────────────────────────────

def nullspace_mod2(A: np.ndarray) -> np.ndarray:
    """
    Compute a basis for the nullspace of matrix A over GF(2).

    That is, find all x ≠ 0 such that

        A x = 0  (mod 2).

    This corresponds to finding all parity-check polynomial vectors h(D).
    """
    A = A.copy().astype(np.uint8)
    m, n = A.shape
    R = A.copy()

    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        sel = None
        for r in range(row, m):
            if R[r, col] == 1:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            R[[row, sel]] = R[[sel, row]]
        for r in range(m):
            if r != row and R[r, col] == 1:
                R[r] ^= R[row]
        pivot_cols.append(col)
        row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8)

    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for r in range(len(pivot_cols)):
            pc = pivot_cols[r]
            if R[r, f] == 1:
                x[pc] = 1
        basis.append(x)

    return np.array(basis, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Parity-check system construction
# ──────────────────────────────────────────────────────────────────────────────

def build_parity_system(generators: List[List[List[int]]], deg_h: int):
    """
    Build the linear system defining parity-check polynomials.

    Let h_j(D) = ∑_{s=0}^{deg_h} h_{j,s} D^s.
    The parity condition

        ∑_j h_j(D) g_{j,i}(D) = 0

    for each input stream i yields a homogeneous linear system in the
    coefficients h_{j,s}.
    """
    n = len(generators)
    k = len(generators[0])

    deg_g = max(len(g) - 1 for out in generators for g in out)
    Kmax = deg_g + deg_h

    rows = k * (Kmax + 1)
    cols = n * (deg_h + 1)
    A = np.zeros((rows, cols), dtype=np.uint8)

    def idx(j, s):
        return j * (deg_h + 1) + s

    for i in range(k):
        for t in range(Kmax + 1):
            r = i * (Kmax + 1) + t
            for j in range(n):
                for u, bit in enumerate(generators[j][i]):
                    if bit and 0 <= t - u <= deg_h:
                        A[r, idx(j, t - u)] ^= 1

    return A


# ──────────────────────────────────────────────────────────────────────────────
# Conversion to time-domain parity equations
# ──────────────────────────────────────────────────────────────────────────────

def parity_vector_to_equation(h_vec: List[List[int]]) -> str:
    """
    Convert a parity-check polynomial vector h(D) into a time-domain equation
    of the form

        v_{j1}[t − s1] ⊕ v_{j2}[t − s2] ⊕ … = 0.
    """
    terms = []
    for j, poly in enumerate(h_vec):
        for s, bit in enumerate(poly):
            if bit:
                terms.append(f"v{j}[t-{s}]")
    return " ⊕ ".join(terms) + " = 0"


# ──────────────────────────────────────────────────────────────────────────────
# Example usage (baseline setup)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: rate-1/2, memory m=2 code
    gens = [
        [parse_poly_token("7")],   # g1(D) = 1 + D + D^2
        [parse_poly_token("5")],   # g2(D) = 1 + D^2
    ]

    deg_h = 5  # typical choice: m + 3
    A = build_parity_system(gens, deg_h)
    basis = nullspace_mod2(A)

    print(f"Found {len(basis)} parity-check vectors")
    for row in basis:
        h = []
        for j in range(len(gens)):
            h.append(row[j*(deg_h+1):(j+1)*(deg_h+1)].tolist())
        print(parity_vector_to_equation(h))
