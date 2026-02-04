#!/usr/bin/env python3
"""

Simple, user-facing **demo script** for the WCNC-2026 codebase:

    "Detecting Convolutional Codes via a Markovian Statistic"

This script is intentionally **minimal and instructional**. It allows a user to:

  • Select parameters for two convolutional codes (H₁ and H₂)
  • Either choose from **predefined example codes** used in the paper
    OR enter custom (k, n, m) and generator polynomials
  • Automatically run the hybrid Markov detector
  • Directly generate plots:
        – P_d vs p
        – P_d vs N

This file is meant for:
  • First-time readers of the repository
  • Demos and sanity checks
  • Teaching / presentation purposes

It is **not** used to generate paper figures (see Pd_plotter.py for that).
"""

import os
import matplotlib.pyplot as plt

from Pd_plotter import run_experiment

# ──────────────────────────────────────────────────────────────────────────────
# Predefined example codes (used in the paper / common benchmarks)
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_CODES = {
    "1": {
        "name": "Rate-1/2, m=2 (7,5) vs (6,5)",
        "k": 1,
        "n": 2,
        "m": 2,
        "gen1": [[[1, 1, 1]], [[1, 0, 1]]],   # (7,5)
        "gen2": [[[1, 1, 0]], [[1, 0, 1]]],   # (6,5)
    },
    "2": {
        "name": "Rate-1/2, m=3 (15,13) vs (13,15)",
        "k": 1,
        "n": 2,
        "m": 3,
        "gen1": [[[1, 1, 1, 1]], [[1, 0, 1, 1]]],
        "gen2": [[[1, 0, 1, 1]], [[1, 1, 1, 1]]],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: read generator taps from user
# ──────────────────────────────────────────────────────────────────────────────

def read_generators(k, n, m, label):
    print(f"\nEnter generator polynomials for {label}")
    print(f"Format: {k} tap vectors per output, each of length {m+1}")
    print(f"Example (rate 1/2): 1,1,1")

    gens = []
    for j in range(n):
        while True:
            s = input(f"  Output v{j}: ").strip()
            try:
                taps = [int(b) for b in s.split(',')]
                if len(taps) != m + 1:
                    raise ValueError
                gens.append([taps])
                break
            except Exception:
                print("  Invalid format. Try again.")
    return gens


# ──────────────────────────────────────────────────────────────────────────────
# Main interactive flow
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Convolutional Code Detector Demo ===\n")

    print("Choose an option:")
    print("  [1] Use predefined example codes")
    print("  [2] Enter custom codes manually")

    mode = input("Selection [1/2]: ").strip()

    if mode == "1":
        print("\nAvailable examples:")
        for k_ex, info in EXAMPLE_CODES.items():
            print(f"  [{k_ex}] {info['name']}")

        choice = input("Select example: ").strip()
        cfg = EXAMPLE_CODES[choice]

        k = cfg["k"]
        n = cfg["n"]
        m = cfg["m"]
        gen1 = cfg["gen1"]
        gen2 = cfg["gen2"]

    else:
        k = int(input("Enter k (inputs per time step): "))
        n = int(input("Enter n (outputs per time step): "))
        m = int(input("Enter m (memory): "))

        gen1 = read_generators(k, n, m, "Code-1 (H1)")
        gen2 = read_generators(k, n, m, "Code-2 (H2)")

    # Experiment parameters (kept simple)
    p_vec = [0.01, 0.05, 0.1, 0.2, 0.3]
    num_iter = 2000

    print("\nRunning hybrid detector… (this may take a minute)\n")

    df = run_experiment(
        k=k,
        n=n,
        m=m,
        gen1=gen1,
        gen2=gen2,
        num_iter=num_iter,
        p_vec=p_vec,
        learn_len=None,
        learn_burn=200,
        laplace=1.0,
        seed=123
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Plot Pd vs p
    # ──────────────────────────────────────────────────────────────────────────

    plt.figure(figsize=(6, 5))
    for N in sorted(df['N'].unique()):
        q = df[df['N'] == N]
        plt.plot(q['p'], q['Pd'], marker='o', label=f"N={N}")

    plt.xlabel("BSC crossover probability p")
    plt.ylabel("Probability of detection $P_d$")
    plt.title("Hybrid detector: $P_d$ vs $p$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # Plot Pd vs N
    # ──────────────────────────────────────────────────────────────────────────

    plt.figure(figsize=(6, 5))
    for p in sorted(df['p'].unique()):
        q = df[df['p'] == p]
        plt.plot(q['N'], q['Pd'], marker='o', label=f"p={p}")

    plt.xlabel("Blocklength N")
    plt.ylabel("Probability of detection $P_d$")
    plt.title("Hybrid detector: $P_d$ vs $N$")
    plt.grid(True)
    plt.legend()
    plt.show()
