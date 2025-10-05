#!/usr/bin/env python3
# alpha_markov_exponent.py
#
# Empirical & theoretical error exponents for a Viterbi-metric Markov detector.
#
# Empirical:
#   Fit P_e(N) = 1 - P_c(N) ≈ A e^(−α N) using log-linear least squares on the
#   low-error tail (P_e ≤ tail_cap).
#
# Theory (Levy–Bernard, Sec. 12.5):
#   M(u)[i,j] = Σ_r P1(i→j,r)^u · P0(i→j,r)^(1−u)
#   α(p)      = min_{u∈[0,1]} [ −ln ρ( M(u) ) ]
#
# P_h(i→j,r) are learned by simulation on a single, fixed decoder metric-state
# space, keeping counts per (i,j,r).  We then form M(u) exactly as above.

import os, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from viterbi_markov import (
    octal_to_taps, enumerate_markov_states_allzero,
    build_trellis, viterbi_metric_step, simulate_markov_sequence
)

# --------------------------- cosmetic defaults ----------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})

# ------------------------------ metric helpers ----------------------------
def metrics_from_noisy(noisy_outputs, decoder_taps, m):
    """Decoder’s Viterbi relative-metric sequence for a given noisy output sequence."""
    trellis = build_trellis(decoder_taps, m)
    cur = tuple([0]*len(trellis))
    seq = [cur]
    for r in noisy_outputs:
        cur = viterbi_metric_step(list(cur), trellis, tuple(r))
        seq.append(cur)
    return seq

def log_likelihood(seq_metrics, P_row, sidx, eps=1e-300):
    """Sum log of row-kernel P_row[i,j] along the path (unknown states → −∞)."""
    if len(seq_metrics) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(seq_metrics[:-1], seq_metrics[1:]):
        i = sidx.get(a); j = sidx.get(b)
        if i is None or j is None: return -np.inf
        pij = P_row[i, j]
        if pij <= 0.0 or not np.isfinite(pij): return -np.inf
        s += np.log(max(pij, eps))
    return s

# ------------------------------ learn P(i->j,r) ---------------------------
def learn_tensor_by_simulation(encoder_taps, decoder_taps, m, p,
                               length=300_000, burn_in=5_000, laplace=1.0, seed=None):
    """
    Learn decoder-state HMM kernel with outputs kept:
      P(i->j, r),  r ∈ {0,1}^n.
    Returns: P_ijr (K,K,R), states, sidx, all_r
    """
    if seed is not None:
        np.random.seed(seed)

    states, transitions, all_r = enumerate_markov_states_allzero(decoder_taps, m)
    sidx = {s:i for i,s in enumerate(states)}
    K = len(states)
    R = len(all_r)
    r2i = {r:i for i,r in enumerate(all_r)}

    trellis = build_trellis(decoder_taps, m)
    cur = tuple([0]*len(trellis))
    cur_i = sidx[cur]

    # tiny inline encoder
    def branch_output_and_next_state(state_int, u, taps_list, m_):
        def state_bits(s, mm): return [(s>>i)&1 for i in range(mm)]
        def bits_to_int(bs):
            v=0
            for i,b in enumerate(bs): v |= (int(b)&1) << i
            return v
        sb = state_bits(state_int, m_)
        x  = [u] + sb[:m_]
        y  = [sum((t[i]&x[i]) for i in range(len(t))) % 2 for t in taps_list]
        ns = bits_to_int(([u]+sb[:m_-1]) if m_>0 else [])
        return tuple(y), ns

    enc_state = 0
    C = np.zeros((K, K, R), dtype=np.float64)  # counts over (i,j,r)

    # burn-in
    for _ in range(burn_in):
        u = np.random.randint(0,2)
        y, enc_state = branch_output_and_next_state(enc_state, u, encoder_taps, m)
        r = tuple([bit ^ np.random.binomial(1,p) for bit in y])
        cur = viterbi_metric_step(list(cur), trellis, r)
        cur_i = sidx.get(cur, cur_i)

    # counting
    for _ in range(length):
        u = np.random.randint(0,2)
        y, enc_state = branch_output_and_next_state(enc_state, u, encoder_taps, m)
        r = tuple([bit ^ np.random.binomial(1,p) for bit in y])
        nxt = viterbi_metric_step(list(cur), trellis, r)
        nxt_i = sidx.get(nxt, None)
        if nxt_i is not None:
            C[cur_i, nxt_i, r2i[r]] += 1.0
            cur, cur_i = nxt, nxt_i

    # Laplace smoothing; normalize over (j,r) for each i
    C += laplace
    row_sums = C.sum(axis=(1,2), keepdims=True)
    P_ijr = C / np.maximum(row_sums, 1.0)
    return P_ijr, states, sidx, all_r

# ------------------------------ theory: α(p) ------------------------------
def spectral_radius(A):
    lam = np.linalg.eigvals(A)
    return float(np.max(np.abs(lam)))

def alpha_theory_from_tensor(P0_ijr, P1_ijr, u_grid=401):
    """
    α(p) = min_u  −ln ρ( M(u) ),  with  M(u)[i,j] = Σ_r P1(i→j,r)^u · P0(i→j,r)^(1−u)
    """
    P0 = np.clip(P0_ijr, 1e-300, 1.0)
    P1 = np.clip(P1_ijr, 1e-300, 1.0)

    us = np.linspace(0.0, 1.0, u_grid)
    best_rho, best_u = None, None
    for u in us:
        M = np.sum((P1**u) * (P0**(1.0-u)), axis=2)   # (K,K)
        rho = max(1e-300, spectral_radius(M))         # λ(u)=ρ(M(u))
        if (best_rho is None) or (rho < best_rho):    # minimize ρ(M(u))
            best_rho, best_u = rho, float(u)
    alpha = -np.log(best_rho)
    return float(alpha), best_u

# ------------------------- empirical Pc(N) & fit --------------------------
def run_pc_vs_N(p, N_list, taps_enc0, taps_enc1, taps_decoder, m,
                P0_row, P1_row, sidx, trials=2000):
    """Empirical P_c(N) using row-kernels (sum over r) for the detector."""
    Pc = []
    for N in N_list:
        correct, total = 0, 0
        for t in tqdm(range(trials), desc=f"p={p:.3f}, N={N}", unit="trial", leave=False):
            if t < trials//2:   # H0 “encoder 0” true
                sim = simulate_markov_sequence(taps_enc0, m, N, p, random_input=True)
                noisy = sim["noisy_outputs"]
                seq = metrics_from_noisy(noisy, taps_decoder, m)
                ll0 = log_likelihood(seq, P0_row, sidx)
                ll1 = log_likelihood(seq, P1_row, sidx)
                correct += int(ll0 >= ll1)
            else:               # H1 “encoder 1” true
                sim = simulate_markov_sequence(taps_enc1, m, N, p, random_input=True)
                noisy = sim["noisy_outputs"]
                seq = metrics_from_noisy(noisy, taps_decoder, m)
                ll0 = log_likelihood(seq, P0_row, sidx)
                ll1 = log_likelihood(seq, P1_row, sidx)
                correct += int(ll0 < ll1)
            total += 1
        Pc.append(correct / max(1,total))
    return np.array(Pc, dtype=float)

def fit_alpha_from_Pe(N_list, Pe_list, tail_cap=0.2, min_points=3):
    """Robust log-linear tail fit: keep points with P_e ≤ tail_cap."""
    N  = np.asarray(N_list, float)
    Pe = np.asarray(Pe_list, float)
    mask = (Pe > 0) & (Pe <= tail_cap)
    N_use, Pe_use = N[mask], Pe[mask]
    if len(N_use) < min_points:
        return 0.0, np.nan
    y = np.log(Pe_use)
    X = np.vstack([np.ones_like(N_use), -N_use]).T  # y = ln A − α N
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, alpha = beta[0], beta[1]
    return float(alpha), float(np.exp(a))

# --------------------------------- main -----------------------------------
def main():
    # ===== knobs =====
    n = 2; m = 2
    g1 = ["14","10"]            # Encoder 0 (octal)
    g2 = ["16","11"]            # Encoder 1 (octal)
    decoder_choice = "code1"  # use Code1 or Code2 as the decoder state space
    p_list  = [0.001, 0.01, 0.10, 0.15, 0.175, 0.20, 0.30, 0.50]
    N_list  = [10, 20, 50, 100, 200, 500]
    learn_len  = 1_000_000
    learn_burn = 5_000
    laplace    = 1.0
    trials_emp = 2_000
    u_grid     = 401
    np.random.seed(1234)
    # =================

    taps1 = [octal_to_taps(o, m+1) for o in g1]
    taps2 = [octal_to_taps(o, m+1) for o in g2]
    dec_taps = taps1 if decoder_choice.lower()=="code1" else taps2
    dec_name = "Code1" if decoder_choice.lower()=="code1" else "Code2"

    os.makedirs("results_alpha", exist_ok=True)

    # decoder indexing (shared)
    states, _, all_r = enumerate_markov_states_allzero(dec_taps, m)
    sidx = {s:i for i,s in enumerate(states)}

    learned   = {}   # p -> (P0_ijr, P1_ijr)
    alpha_th  = {}   # p -> {"alpha":..., "u_star":...}
    alpha_emp, A_emp = {}, {}
    curves    = {}   # p -> {N, Pc, Pe, Pe_fit}

    # ---------- learn kernels & theory ----------
    print("\nLearning per-output kernels and computing theoretical α(p)...\n")
    for p in p_list:
        P0_ijr, _, _, _ = learn_tensor_by_simulation(taps1, dec_taps, m, p,
                                                     length=learn_len, burn_in=learn_burn,
                                                     laplace=laplace)
        P1_ijr, _, _, _ = learn_tensor_by_simulation(taps2, dec_taps, m, p,
                                                     length=learn_len, burn_in=learn_burn,
                                                     laplace=laplace)
        learned[p] = (P0_ijr, P1_ijr)
        a_th, u_star = alpha_theory_from_tensor(P0_ijr, P1_ijr, u_grid=u_grid)
        alpha_th[p] = {"alpha": float(a_th), "u_star": float(u_star)}
        print(f"p={p:.3f}: theory alpha={a_th:.4f}, u*≈{u_star:.3f}")

    # ---------- empirical P_c(N) per p with fits ----------
    print("\nEmpirical P_c(N), fits on the tail P_e ≤ 0.2 ...\n")
    for p in p_list:
        P0_ijr, P1_ijr = learned[p]
        P0_row = P0_ijr.sum(axis=2)  # (K,K)
        P1_row = P1_ijr.sum(axis=2)

        Pc = run_pc_vs_N(p, N_list, taps1, taps2, dec_taps, m, P0_row, P1_row, sidx, trials=trials_emp)
        Pe = 1.0 - Pc
        a_emp, A = fit_alpha_from_Pe(N_list, Pe, tail_cap=0.2, min_points=3)
        alpha_emp[p] = float(a_emp); A_emp[p] = float(A)
        Pe_fit = (A if np.isfinite(A) else 1.0) * np.exp(-a_emp*np.array(N_list))
        curves[p] = {"N": N_list, "Pc": Pc.tolist(), "Pe": Pe.tolist(), "Pe_fit": Pe_fit.tolist()}

        # ----- Per-p: Empirical P_e(N) with fit (log-y) -----
        plt.figure(figsize=(7.2,5.2))
        plt.semilogy(N_list, Pe, 'o-', label=r'Empirical $P_e(N)=1-P_c(N)$')
        plt.semilogy(N_list, Pe_fit, '--', label=fr'Fit: $A e^{{-\alpha N}}$  ( $\alpha_{{emp}}\approx {a_emp:.3f}$ )')
        plt.xlabel("Sequence length N")
        plt.ylabel("Error probability  $P_e(N)$")
        plt.title(fr"Empirical $P_e$ vs $N$  (p={p}, decoder={dec_name})")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join("results_alpha_m3", f"empirical_Pe_vs_N_p{p}.png"), dpi=300)
        plt.close()
        print(f"p={p:.3f}: empirical alpha≈{a_emp:.4f}, A≈{A:.3g}")

    # ===================== SUMMARY FIGURES (additions) ======================

    # -- (S1) Overlay: ALL empirical P_e(N) with fitted A e^(−α N)  [log-y]
    N_dense = np.linspace(min(N_list), max(N_list), 400)
    plt.figure(figsize=(8.6,6.2))
    for p in p_list:
        data = curves[p]
        Np  = np.array(data["N"],  dtype=float)
        Pe  = np.array(data["Pe"], dtype=float)
        A   = A_emp[p]
        a   = alpha_emp[p]
        Pe_fit_dense = (A if np.isfinite(A) else 1.0) * np.exp(-a * N_dense)
        plt.semilogy(Np, Pe, 'o', alpha=0.85)
        plt.semilogy(N_dense, Pe_fit_dense, '-', linewidth=2,
                     label=fr"p={p:.2f}  ( $\alpha_{{emp}}={a:.3f}$ )")
    plt.xlabel("Sequence length $N$")
    plt.ylabel(r"Error probability $P_e(N)=1-P_c(N)$")
    plt.title(r"Empirical $P_e(N)$ with fitted $A e^{-\alpha N}$ (all $p$, log scale)")
    plt.legend(title="Curves", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "SUMMARY_empirical_all_Pe_with_fits_LOG.png"), dpi=300)
    plt.close()

    # -- (S2) The same overlay on LINEAR y-axis (intuitive absolute errors)
    plt.figure(figsize=(8.6,6.2))
    for p in p_list:
        data = curves[p]
        Np  = np.array(data["N"],  dtype=float)
        Pe  = np.array(data["Pe"], dtype=float)
        A   = A_emp[p]
        a   = alpha_emp[p]
        Pe_fit_dense = (A if np.isfinite(A) else 1.0) * np.exp(-a * N_dense)
        plt.plot(Np, Pe, 'o', alpha=0.85)
        plt.plot(N_dense, Pe_fit_dense, '-', linewidth=2,
                 label=fr"p={p:.2f}  ( $\alpha_{{emp}}={a:.3f}$ )")
    plt.xlabel("Sequence length $N$")
    plt.ylabel(r"$P_e(N)=1-P_c(N)$")
    plt.title(r"Empirical $P_e(N)$ with fitted $A e^{-\alpha N}$ (all $p$, linear scale)")
    plt.legend(title="Curves", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "SUMMARY_empirical_all_Pe_with_fits_LIN.png"), dpi=300)
    plt.close()

    # -- (S3) Simple summary: P_c(N) (no log), all p in one panel
    plt.figure(figsize=(8.6,6.2))
    for p in p_list:
        data = curves[p]
        Np = np.array(data["N"], dtype=float)
        Pc = np.array(data["Pc"], dtype=float)
        plt.plot(Np, Pc, 'o-', label=fr"p={p:.2f}")
    plt.xlabel("Sequence length $N$")
    plt.ylabel(r"Correct detection probability $P_c(N)$")
    plt.title(fr"$P_c(N)$ vs $N$ (decoder={dec_name})")
    plt.ylim(0.0, 1.05)
    plt.legend(title="BSC crossover p")
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "SUMMARY_Pc_vs_N_linear.png"), dpi=300)
    plt.close()

    # -- (S4) Overlay of “theoretical slopes only”: C_p e^(−α_theory N), scaled
    #          to match the first empirical point so slopes are comparable.
    plt.figure(figsize=(8.6,6.2))
    for p in p_list:
        a_th = alpha_th[p]["alpha"]
        N_min = float(min(N_list))
        Pe0   = float(curves[p]["Pe"][0])  # empirical Pe at smallest N
        C_p   = Pe0 * np.exp(a_th * N_min)
        Pe_th_dense = C_p * np.exp(-a_th * N_dense)
        plt.semilogy(N_dense, Pe_th_dense, '-', linewidth=2,
                     label=fr"p={p:.2f}  ( $\alpha_{{theory}}={a_th:.3f}$ )")
    plt.xlabel("Sequence length $N$")
    plt.ylabel(r"$C_p\,e^{-\alpha_{\rm theory}(p)\,N}$  (scaled at $N_{\min}$)")
    plt.title("Theoretical slopes only (scaled to first point) — all p")
    plt.legend(title="Curves", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "SUMMARY_theory_all_surrogates.png"), dpi=300)
    plt.close()

    # ===================== end of added summary figures =====================

    # ---------- α vs p: theory ----------
    ps = np.array(p_list, float)
    a_th_vals = np.array([alpha_th[p]["alpha"] for p in p_list], float)
    u_stars   = np.array([alpha_th[p]["u_star"]  for p in p_list], float)

    plt.figure(figsize=(7.2,5.2))
    plt.plot(ps, a_th_vals, 'o-', linewidth=2)
    for p, a, u in zip(ps, a_th_vals, u_stars):
        plt.annotate(f"u*≈{u:.2f}", (p, a), textcoords="offset points",
                     xytext=(0,8), ha='center', fontsize=8)
    plt.xlabel("Crossover probability p")
    plt.ylabel("Theoretical error exponent  α_theory(p)")
    plt.title("Theoretical α(p)  =  min_{u∈[0,1]}  −ln ρ( Σ_r P1^u · P0^(1−u) )")
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "theoretical_alpha_vs_p.png"), dpi=300)
    plt.close()

    # ---------- α vs p: empirical ----------
    a_emp_vals = np.array([alpha_emp[p] for p in p_list], float)
    plt.figure(figsize=(7.2,5.2))
    plt.plot(ps, a_emp_vals, 's--', linewidth=2, label=r'$\alpha_{\rm emp}(p)$')
    plt.xlabel("Crossover probability p")
    plt.ylabel("Empirical error exponent  α_emp(p)")
    plt.title("Empirical α(p) from tail fit of ln P_e(N) vs N")
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "empirical_alpha_vs_p.png"), dpi=300)
    plt.close()

    # ---------- α vs p: overlay ----------
    plt.figure(figsize=(7.2,5.2))
    plt.plot(ps, a_th_vals,  'o-', label=r'$\alpha_{\rm theory}(p)$')
    plt.plot(ps, a_emp_vals, 's--', label=r'$\alpha_{\rm emp}(p)$')
    plt.xlabel("Crossover probability p")
    plt.ylabel("Error exponent α")
    plt.title(f"Error exponent vs p  (decoder={dec_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results_alpha", "alpha_vs_p_combined.png"), dpi=300)
    plt.close()

    # ---------- Save numeric results ----------
    out = {
        "parameters": {
            "n": n, "m": m, "g1": g1, "g2": g2,
            "decoder": dec_name, "p_list": p_list, "N_list": N_list,
            "learn_len": learn_len, "learn_burn": learn_burn,
            "laplace": laplace, "trials_emp": trials_emp, "u_grid": u_grid
        },
        "alpha_theory": {p: alpha_th[p] for p in p_list},
        "alpha_emp": {p: alpha_emp[p] for p in p_list},
        "A_emp": {p: A_emp[p] for p in p_list},
        "curves": curves
    }
    with open(os.path.join("results_alpha", "alpha_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved plots to results_alpha/:")
    print("  empirical_Pe_vs_N_p*.png                    (one per p)")
    print("  SUMMARY_empirical_all_Pe_with_fits_LOG.png  (all p, log)")
    print("  SUMMARY_empirical_all_Pe_with_fits_LIN.png  (all p, linear)")
    print("  SUMMARY_Pc_vs_N_linear.png                  (all p)")
    print("  SUMMARY_theory_all_surrogates.png           (all p, theory slopes)")
    print("  theoretical_alpha_vs_p.png, empirical_alpha_vs_p.png, alpha_vs_p_combined.png")
    print("Numeric results: results_alpha/alpha_results.json\n")

if __name__ == "__main__":
    main()
