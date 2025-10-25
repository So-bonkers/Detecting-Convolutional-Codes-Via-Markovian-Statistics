#!/usr/bin/env python3
"""
comp_parity_generalised.py (true parity checks, single chosen template for all N)

- Computes true parity-check polynomial vectors h(D) for Encoder A (deg_h = m + 3).
- Displays basis parity vectors and asks user to choose one template (index or 'a' for auto).
- Uses chosen template for the whole experiment (all N).
- Dynamically recomputes anchors per N, uses equal-anchor sampling per trial.

"""
import os
import re
import math
import random
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Experiment params ----------------------
TRIALS = 20000
THRESH_LIST = [0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.58, 0.57, 0.55, 0.54, 0.53, 0.52, 0.51]
N_SPECTRA_BY_M = {
    1:  [5,10,20,50,100,200],
    2:  [500],
    3:  [20,50,100,200,300,500],
    4:  [50,100,200,300,500]
}
N_FALLBACK = [20,50,100,200,500,1000]
P_VEC_BY_M = {
    1:  [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    2:  [0.001, 0.01, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    3:  [0.001, 0.01, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    4:  [0.001, 0.01, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
}
RESULTS_DIR = "results_comp_parity_truechecks_choose"

# ---------------------- small helpers ----------------------
def parse_poly_token(token: str) -> List[int]:
    token = token.strip()
    if token == "":
        raise ValueError("Empty polynomial token")
    if "," in token:
        parts = [p.strip() for p in token.split(",") if p.strip() != ""]
        return [int(x) for x in parts]
    if re.fullmatch(r"[01]+", token):
        return [int(ch) for ch in token[::-1]]  # MSB-first string -> return lsb-first list
    if re.fullmatch(r"[0-7]+", token):
        val = int(token, 8)
        b = bin(val)[2:]
        return [int(ch) for ch in b[::-1]]
    try:
        val = int(token, 0)
        b = bin(val)[2:]
        return [int(ch) for ch in b[::-1]]
    except Exception as e:
        raise ValueError(f"Cannot parse polynomial token: '{token}' ({e})")

def normalize_tapvecs(gens_by_n_k: List[List[List[int]]]) -> List[List[List[int]]]:
    maxlen = 0
    for out in gens_by_n_k:
        for taps in out:
            maxlen = max(maxlen, len(taps))
    if maxlen == 0:
        return gens_by_n_k
    out = []
    for outgens in gens_by_n_k:
        new_out = []
        for taps in outgens:
            if len(taps) < maxlen:
                new_out.append(taps + [0] * (maxlen - len(taps)))
            else:
                new_out.append(taps[:maxlen])
        out.append(new_out)
    return out

def poly_to_str(poly: List[int]) -> str:
    if not poly:
        return "0"
    deg = len(poly)-1
    while deg>0 and poly[deg]==0:
        deg -= 1
    return "".join(str(poly[i]) for i in range(deg, -1, -1))

def pretty_equation_from_hvec(hvec: List[List[int]]) -> str:
    terms = []
    for j, poly in enumerate(hvec):
        for s, coeff in enumerate(poly):
            if coeff & 1:
                if s == 0:
                    terms.append(f"v{j}[t]")
                else:
                    terms.append(f"v{j}[t-{s}]")
    if not terms:
        return "0=0"
    return "+".join(terms) + "=0"

# ---------------------- GF(2) nullspace ----------------------
def nullspace_mod2(A: np.ndarray) -> np.ndarray:
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
            R[[sel, row], :] = R[[row, sel], :]
        for r in range(m):
            if r != row and R[r, col] == 1:
                R[r, :] ^= R[row, :]
        pivot_cols.append(col)
        row += 1
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8)
    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for r in range(m):
            pcs = np.where(R[r, :] == 1)[0]
            if pcs.size == 0:
                continue
            pc = pcs[0]
            cur = int(np.dot(R[r, :], x) % 2)
            x[pc] = cur
        basis.append(x)
    return np.array(basis, dtype=np.uint8)

# ---------------------- build polynomial system ----------------------
def build_system_for_h(gens_by_n_k: List[List[List[int]]], deg_h: int):
    n = len(gens_by_n_k)
    k = len(gens_by_n_k[0]) if n>0 else 0
    deg_g = 0
    for j in range(n):
        for i in range(k):
            deg_g = max(deg_g, len(gens_by_n_k[j][i]) - 1)
    deg_g = max(0, deg_g)
    Kmax = deg_g + deg_h
    rows = k * (Kmax + 1)
    cols = n * (deg_h + 1)
    A = np.zeros((rows, cols), dtype=np.uint8)
    def var_idx(j, s):
        return j*(deg_h+1) + s
    for i in range(k):
        for t in range(Kmax+1):
            row_idx = i*(Kmax+1) + t
            for j in range(n):
                gji = gens_by_n_k[j][i]
                for u, gbit in enumerate(gji):
                    if gbit & 1:
                        s = t - u
                        if 0 <= s <= deg_h:
                            A[row_idx, var_idx(j, s)] ^= 1
    return A, deg_g, Kmax

# ---------------------- convolution & helpers ----------------------
def encode_conv_general(u_bits_by_k: List[List[int]], gens_by_n_k: List[List[List[int]]], terminate: bool=True):
    k = len(u_bits_by_k)
    n = len(gens_by_n_k)
    Lmax = 0
    for outgens in gens_by_n_k:
        for g in outgens:
            Lmax = max(Lmax, len(g))
    m = max(0, Lmax-1)
    N = len(u_bits_by_k[0]) if k>0 else 0
    if terminate:
        T = N + m
        u_ext = [list(u) + [0]*m for u in u_bits_by_k]
    else:
        T = N
        u_ext = [list(u) for u in u_bits_by_k]
    outputs = [[0]*T for _ in range(n)]
    for t in range(T):
        for j in range(n):
            s = 0
            outgens = gens_by_n_k[j]
            for i in range(k):
                g = outgens[i]
                for shift, bit in enumerate(g):
                    if bit and (t - shift) >= 0:
                        s ^= u_ext[i][t - shift]
            outputs[j][t] = s
    return outputs, T

def bsc_apply(bits: List[int], p: float, rng: random.Random) -> List[int]:
    return [b ^ (1 if rng.random() < p else 0) for b in bits]

def check_h_on_sequence(hvec: List[List[int]], vlist: List[List[int]]):
    n = len(hvec)
    T = len(vlist[0])
    max_s = max((len(hvec[j]) - 1) if len(hvec[j])>0 else 0 for j in range(n))
    satisfied = 0
    total = 0
    for t in range(max_s, T):
        xor = 0
        ok = True
        for j in range(n):
            hj = hvec[j]
            for s, coeff in enumerate(hj):
                if coeff & 1:
                    idx = t - s
                    if idx < 0 or idx >= T:
                        ok = False
                        break
                    xor ^= vlist[j][idx]
            if not ok:
                break
        if ok:
            total += 1
            if xor == 0:
                satisfied += 1
    frac = satisfied / total if total>0 else float('nan')
    return frac, satisfied, total

def basis_row_to_hvec(row: np.ndarray, n: int, deg_h: int) -> List[List[int]]:
    vec = row.tolist()
    hvec = []
    for j in range(n):
        coeffs = vec[j*(deg_h+1):(j+1)*(deg_h+1)]
        hvec.append(coeffs)
    return hvec

def anchors_for_template(gens_by_n_k: List[List[List[int]]], offsets: Tuple[Tuple[str,int], ...], K: int, terminate: bool=True):
    gens_by_n_k = normalize_tapvecs(gens_by_n_k)
    Lmax = 0
    for outgens in gens_by_n_k:
        for g in outgens:
            Lmax = max(Lmax, len(g))
    m = max(0, Lmax-1)
    T = K + m if terminate else K
    display_times = list(range(0, T))
    display_set = set(display_times)
    valid_anchors = []
    for a in display_times:
        ok = True
        for (_kind, off) in offsets:
            if (a + off) not in display_set:
                ok = False
                break
        if ok:
            valid_anchors.append(a)
    return sorted(valid_anchors)

# ---------------------- main ----------------------
def main():
    print("=== comp_parity_generalised.py (true parity checks; choose one template once) ===")
    # Pair A
    print("=== Pair A (reference encoder) ===")
    k_a = int(input("Enter k for Pair A (inputs per time step, integer >=1): ").strip() or "1")
    n_a = int(input("Enter n for Pair A (outputs per time step, integer >=1): ").strip() or "2")
    print(f"Enter generators for Pair A: for each of the {n_a} outputs supply {k_a} tokens (space-separated).")
    gens_a = []
    for j in range(n_a):
        while True:
            s = input(f"  v{j} gens (k={k_a} tokens): ").strip()
            parts = s.split()
            if len(parts) == 1 and "," in s:
                cand = [p.strip() for p in s.split(",") if p.strip()!=""]
                if len(cand)==k_a:
                    parts = cand
            if len(parts) != k_a:
                print(f"Expected {k_a} tokens, got {len(parts)}. Try again.")
                continue
            try:
                taps = [parse_poly_token(tok) for tok in parts]
                gens_a.append(taps)
                break
            except Exception as e:
                print("Parse error:", e)
    # Pair B
    print("\n=== Pair B (competitor encoder) ===")
    k_b = int(input("Enter k for Pair B (inputs per time step, integer >=1): ").strip() or "1")
    n_b = int(input("Enter n for Pair B (outputs per time step, integer >=1): ").strip() or "2")
    print(f"Enter generators for Pair B: for each of the {n_b} outputs supply {k_b} tokens (space-separated).")
    gens_b = []
    for j in range(n_b):
        while True:
            s = input(f"  v{j} gens (k={k_b} tokens): ").strip()
            parts = s.split()
            if len(parts) == 1 and "," in s:
                cand = [p.strip() for p in s.split(",") if p.strip()!=""]
                if len(cand)==k_b:
                    parts = cand
            if len(parts) != k_b:
                print(f"Expected {k_b} tokens, got {len(parts)}. Try again.")
                continue
            try:
                taps = [parse_poly_token(tok) for tok in parts]
                gens_b.append(taps)
                break
            except Exception as e:
                print("Parse error:", e)
    rand_flag = input("Use random input? (y/n) [n]: ").strip().lower() == "y"
    seed = int(input("Random seed (integer, e.g. 123): ").strip() or "123")

    gens_a = normalize_tapvecs(gens_a)
    gens_b = normalize_tapvecs(gens_b)

    # memory and degree bound
    L_a = max(len(g) for out in gens_a for g in out) if gens_a else 1
    m_a = L_a - 1
    L_b = max(len(g) for out in gens_b for g in out) if gens_b else 1
    m_b = L_b - 1
    m = max(m_a, m_b)
    deg_h = m + 3
    print(f"\nDetected max memory m = {m}. Using deg_h = m + 3 = {deg_h} as degree bound for parity polynomials.")

    # build linear system and nullspace for encoder A
    A_sys, deg_g, Kmax = build_system_for_h(gens_a, deg_h)
    rows, cols = A_sys.shape
    print(f"Linear system: {rows} equations x {cols} unknowns (GF(2)). Computing nullspace...")
    basis = nullspace_mod2(A_sys)
    null_dim = basis.shape[0]
    print(f"Nullspace dimension = {null_dim}")
    if null_dim == 0:
        print("No nontrivial parity vectors found with deg_h. Consider increasing deg_h.")
        return

    # convert to hvecs and display options
    h_basis = [basis_row_to_hvec(basis[r,:], n_a, deg_h) for r in range(basis.shape[0])]
    print("\nParity-check basis vectors (index, polynomial form, num terms, time-domain):")
    for idx, hvec in enumerate(h_basis):
        poly_strs = [poly_to_str(hvec[j]) for j in range(n_a)]
        # count terms:
        term_count = sum(sum(1 for c in hvec[j] if c&1) for j in range(n_a))
        print(f"[{idx}] " + " , ".join([f"h{j}={poly_strs[j]}" for j in range(n_a)]) + f"  (terms={term_count})")
        print("     ->", pretty_equation_from_hvec(hvec))

    # Ask user to choose one template (or auto)
    print("\nChoose the template to use for ALL N. Enter an index (e.g. 0), or 'a' to auto-select a best candidate.")
    while True:
        choice = input("Your choice (index or 'a'): ").strip().lower()
        if choice == 'a':
            # auto-select: prefer those with the fewest total polynomial degree (heuristic) or most terms
            # We'll pick the basis vector with largest term_count
            term_counts = [sum(sum(1 for c in h_basis[i][j] if c&1) for j in range(n_a)) for i in range(len(h_basis))]
            chosen_idx = int(np.argmax(term_counts))
            print(f"Auto-selected index {chosen_idx} (max term count = {term_counts[chosen_idx]})")
            break
        else:
            try:
                chosen_idx = int(choice)
                if 0 <= chosen_idx < len(h_basis):
                    break
                else:
                    print("Index out of range. Try again.")
            except Exception:
                print("Invalid input. Enter an integer index or 'a'.")
    chosen_hvec = h_basis[chosen_idx]
    chosen_sym = pretty_equation_from_hvec(chosen_hvec)
    print(f"\nChosen template [{chosen_idx}] -> {chosen_sym}\nProceeding to run experiments using this single template for all N.\n")

    # prepare templates structure: offsets for chosen template
    # offsets use (kind, offset) where offset = -s for term vj[t-s]
    parsed_template = tuple(sorted([(f'v{j}', -s) for j in range(n_a) for s,coeff in enumerate(chosen_hvec[j]) if coeff & 1]))
    # template_sym already provided
    template_sym = chosen_sym

    # experiment N and P lists
    N_LIST = N_SPECTRA_BY_M.get(m, N_FALLBACK)
    P_LIST = P_VEC_BY_M[1] if m==1 else P_VEC_BY_M.get(m, P_VEC_BY_M[4])
    print(f"Experiment will use N values: {N_LIST}")
    print(f"Experiment will use p values: {P_LIST}")
    print(f"Thresholds: {THRESH_LIST}")
    print(f"Trials per (N,p): {TRIALS}")
    print("Proceeding... (Ctrl-C to abort)\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd_results = []

    # run experiments (same chosen template for all N)
    for N in N_LIST:
        # generate inputs (fixed per N)
        if rand_flag:
            rngA = random.Random(seed + 1000 + N)
            rngB = random.Random(seed + 2000 + N)
            u_a = [[rngA.randint(0,1) for _ in range(N)] for __ in range(k_a)]
            u_b = [[rngB.randint(0,1) for _ in range(N)] for __ in range(k_b)]
        else:
            u_a = [[0]*N for _ in range(k_a)]
            u_b = [[0]*N for _ in range(k_b)]

        vlist_a, T_a = encode_conv_general(u_a, gens_a, terminate=True)
        vlist_b, T_b = encode_conv_general(u_b, gens_b, terminate=True)

        # dynamic anchors for chosen template on this N
        anchors_a = anchors_for_template(gens_a, parsed_template, N, terminate=True)
        anchors_b = anchors_for_template(gens_b, parsed_template, N, terminate=True)

        # compute clean fractions (sanity check)
        def build_streams(vlist):
            Tloc = len(vlist[0])
            streams = {}
            for idx in range(Tloc):
                row = {}
                for j, stream in enumerate(vlist):
                    row[f'v{j}'] = stream[idx]
                    row[f'y{j}'] = stream[idx]
                streams[idx] = row
            return streams
        streams_a_clean = build_streams(vlist_a)
        streams_b_clean = build_streams(vlist_b)

        succ_a = 0
        for a in anchors_a:
            xor = 0
            ok = True
            for kind, off in parsed_template:
                tref = a + off
                if tref not in streams_a_clean:
                    ok = False
                    break
                xor ^= int(streams_a_clean[tref][kind])
            if ok and xor == 0:
                succ_a += 1
        clean_frac_a = succ_a / len(anchors_a) if anchors_a else float('nan')

        succ_b = 0
        for b in anchors_b:
            xor = 0
            ok = True
            for kind, off in parsed_template:
                tref = b + off
                if tref not in streams_b_clean:
                    ok = False
                    break
                xor ^= int(streams_b_clean[tref][kind])
            if ok and xor == 0:
                succ_b += 1
        clean_frac_b = succ_b / len(anchors_b) if anchors_b else float('nan')

        print(f"N={N}: anchors_A_avail={len(anchors_a)}, anchors_B_avail={len(anchors_b)}, clean_frac_A={clean_frac_a:.6f}, clean_frac_B={clean_frac_b:.6f}")

        if len(anchors_a) == 0 or len(anchors_b) == 0:
            print(f"Skipping N={N}: no anchors on at least one side.")
            continue

        M_eff = min(len(anchors_a), len(anchors_b))
        print(f"N={N}: using M_eff={M_eff} anchors per trial (equal sampling)\n")

        # run Monte-Carlo trials for p values
        for p in P_LIST:
            print(f" Processing p={p}")
            thresh_count = np.zeros(len(THRESH_LIST), dtype=int)
            base_seed = seed + int(round(p * 1e6)) + N*1000

            # precompute clean flattened streams, we'll re-noise each trial
            for ttrial in range(TRIALS):
                trial_rng = random.Random(base_seed + ttrial)
                # noisy A
                flat_a = []
                for s in vlist_a:
                    flat_a.extend(s)
                noisy_flat_a = bsc_apply(flat_a, p, trial_rng)
                noisy_streams_a = []
                Tloc_a = len(vlist_a[0])
                for j in range(len(vlist_a)):
                    start = j * Tloc_a
                    noisy_streams_a.append(noisy_flat_a[start:start+Tloc_a])
                streams_a = {}
                for idx in range(Tloc_a):
                    row = {}
                    for j in range(len(vlist_a)):
                        row[f'v{j}'] = vlist_a[j][idx]
                        row[f'y{j}'] = noisy_streams_a[j][idx]
                    streams_a[idx] = row

                # noisy B
                trial_rng_b = random.Random(base_seed + ttrial + 999999)
                flat_b = []
                for s in vlist_b:
                    flat_b.extend(s)
                noisy_flat_b = bsc_apply(flat_b, p, trial_rng_b)
                noisy_streams_b = []
                Tloc_b = len(vlist_b[0])
                for j in range(len(vlist_b)):
                    start = j * Tloc_b
                    noisy_streams_b.append(noisy_flat_b[start:start+Tloc_b])
                streams_b = {}
                for idx in range(Tloc_b):
                    row = {}
                    for j in range(len(vlist_b)):
                        row[f'v{j}'] = vlist_b[j][idx]
                        row[f'y{j}'] = noisy_streams_b[j][idx]
                    streams_b[idx] = row

                # sample anchors without replacement
                anchors_a_sample = trial_rng.sample(anchors_a, M_eff) if M_eff < len(anchors_a) else list(anchors_a)
                anchors_b_sample = trial_rng.sample(anchors_b, M_eff) if M_eff < len(anchors_b) else list(anchors_b)

                # count satisfied anchors
                C_a = 0
                for a in anchors_a_sample:
                    xor = 0
                    ok = True
                    for kind, off in parsed_template:
                        tref = a + off
                        if tref not in streams_a:
                            ok = False
                            break
                        ykey = 'y' + kind[1:]
                        xor ^= int(streams_a[tref][ykey])
                    if ok and xor == 0:
                        C_a += 1

                C_b = 0
                for b in anchors_b_sample:
                    xor = 0
                    ok = True
                    for kind, off in parsed_template:
                        tref = b + off
                        if tref not in streams_b:
                            ok = False
                            break
                        ykey = 'y' + kind[1:]
                        xor ^= int(streams_b[tref][ykey])
                    if ok and xor == 0:
                        C_b += 1

                P_emp_a = C_a / float(M_eff)
                P_emp_b = C_b / float(M_eff)

                p_a_ge = np.array([P_emp_a >= th for th in THRESH_LIST], dtype=int)
                p_b_ge = np.array([P_emp_b >= th for th in THRESH_LIST], dtype=int)
                succ = p_a_ge + (1 - p_b_ge)
                thresh_count += succ

            # aggregate Pd
            for idx_th, th in enumerate(THRESH_LIST):
                total_succ = int(thresh_count[idx_th])
                Pd_mean = float(total_succ / (2.0 * TRIALS))
                Pd_stderr = math.sqrt(max(Pd_mean * (1.0 - Pd_mean), 0.0) / TRIALS)
                win_len = (0 - min([off for (_k, off) in parsed_template])) if parsed_template else 0
                pd_results.append((N, p, float(th), Pd_mean, Pd_stderr, win_len, M_eff, len(anchors_a), len(anchors_b), clean_frac_a, clean_frac_b))
                print(f"    th={th:.3f} -> Pd = {Pd_mean:.6f} ± {Pd_stderr:.6f} (N={N}, p={p})")

    # save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "pd_results_chosen_template.csv")
    cols = ["N","p","threshold","Pd_mean","Pd_stderr","win_len","M_eff","anchorsA","anchorsB","clean_frac_A","clean_frac_B"]
    df = pd.DataFrame(pd_results, columns=cols)
    df["template_sym"] = template_sym
    df.to_csv(out_csv, index=False)
    print("\nSaved CSV ->", os.path.abspath(out_csv))

    # plots grouped by threshold
    by_thresh = defaultdict(list)
    for row in pd_results:
        N_r, p_r, th_r, Pd_mean_r, Pd_stderr_r, *rest = row
        by_thresh[th_r].append((N_r, p_r, Pd_mean_r, Pd_stderr_r))
    for th, entries in sorted(by_thresh.items()):
        plt.figure(figsize=(9,6))
        cmap = plt.get_cmap("tab10")
        Ns_sorted = sorted(set([e[0] for e in entries]))
        for idx, N in enumerate(Ns_sorted):
            xs = [p for (N_r, p, Pd_m, Pd_se) in entries if N_r == N]
            if not xs:
                continue
            order = np.argsort(xs)
            xs_sorted = [xs[i] for i in order]
            ys = [Pd_m for (N_r, p, Pd_m, Pd_se) in entries if N_r == N]
            ys_sorted = [ys[i] for i in order]
            errs = [Pd_se for (N_r, p, Pd_m, Pd_se) in entries if N_r == N]
            errs_sorted = [errs[i] for i in order]
            color = cmap(idx % 10)
            plt.errorbar(xs_sorted, ys_sorted, yerr=errs_sorted, marker='o', linestyle='-', color=color, label=f"N={N}")
        plt.xlabel("BSC flip probability p")
        plt.ylabel("Pd")
        plt.title(f"Pd vs p (threshold={th:.3f})")
        plt.grid(True)
        plt.legend(ncol=2, fontsize='small')
        out_png = os.path.join(RESULTS_DIR, f"Pd_vs_p_th{th:.3f}.png")
        plt.savefig(out_png)
        plt.close()
        print("Saved plot ->", os.path.abspath(out_png))

    print("\nAll done. Results (CSV + plots) in:", os.path.abspath(RESULTS_DIR))

if __name__ == "__main__":
    main()
