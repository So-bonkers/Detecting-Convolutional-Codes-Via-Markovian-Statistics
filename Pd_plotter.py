# Pd_plotter_generalised.py (modified: interactive octal input for k,n,m)
# Based on your original Pd_plotter_generalised.py (keeps experiment logic intact).
# Citation: original file Pd_plotter_generalised.py. :contentReference[oaicite:1]{index=1}

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

# ----------------- USER CONFIG / DEFAULTS (not used for gens now) -----------------
DEFAULTS = {
    "num_iter": 10000,  # M
    "p_vec": [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  # BSC crossover probabilities
    "use_precomputed_symbolic": True,
    "symbolic_dir": r"./results/symbolic_matrices",
    "save_dir": "./results_experiments",
    "seed": 12345,
    "learn_len": None,
    "learn_burn": 200,
    "laplace": 1.0
}

# choose N-spectrum by memory m
N_SPECTRUM_BY_M = {
    1: [5, 10, 20, 50, 100, 200],
    2: [500],
    3: [500],
    4: [50, 100, 200, 300, 500]
}

N_SPECTRUM_FALLBACK = [20, 50, 100, 200, 500, 1000, 2000]

# ----------------- import viterbi_markov from repo -----------------
try:
    import viterbi_markov_generalised as vm
except Exception:
    try:
        import viterbi_markov_generalised as vm
    except Exception:
        import importlib, sys
        sys.path.append(os.getcwd())
        try:
            vm = importlib.import_module("viterbi_markov_comma_input")
        except:
            vm = importlib.import_module("viterbi_markov")

# ----------------- utility helpers -----------------

def octal_to_int(oct_str):
    """Convert octal string (or int) to integer."""
    if isinstance(oct_str, int):
        return oct_str
    try:
        return int(oct_str, 8)
    except Exception:
        # fallback: try decimal
        return int(oct_str)

def int_to_tapvec(g_int, m):
    """Return tap vector (lsb-first) of length (m+1) from integer g_int."""
    bits = [(g_int >> i) & 1 for i in range(m + 1)]
    return bits

def parse_octal_generators_for_output(line: str, k: int, m: int):
    """Parse a line consisting of k octal tokens (space-separated) for one output stream.
       Return list of k tap vectors (each lsb-first length m+1)."""
    parts = line.strip().split()
    if len(parts) != k:
        raise ValueError(f"Expected {k} octal tokens, got {len(parts)}")
    taps = []
    for tok in parts:
        gint = octal_to_int(tok)
        taps.append(int_to_tapvec(gint, m))
    return taps

def parse_all_generators_from_octal_inputs(k: int, n: int, m: int):
    """
    Interactively prompt user for n lines, each having k octal numbers (space-separated).
    Returns generator matrix: list length n, each element a list length k of tap vectors (lsb-first).
    """
    print(f"\nEnter generator polynomials in octal. For each of the {n} outputs, enter {k} octal tokens separated by spaces.")
    print(f"Each octal token will be converted to a tap vector of length {m+1} (lsb-first). Example for k=2: '13 12'")
    gens = []
    for j in range(n):
        while True:
            s = input(f"  Output v{j} (k={k} octal tokens): ").strip()
            if not s:
                print("  Empty input — try again.")
                continue
            try:
                taps = parse_octal_generators_for_output(s, k, m)
                gens.append(taps)
                break
            except Exception as e:
                print("  Parse error:", e)
                print("  Please re-enter the k octal tokens for this output (space-separated).")
    return gens

def parse_generator_matrix(gen_str, k, constraint_length):
    """
    Parse generator string in format: "1,1;0,1"
    Returns: list of k tap vectors (each as list of bits)
    (Used by some existing code paths that expect comma-separated format)
    """
    parts = gen_str.strip().split(';')
    if len(parts) != k:
        raise ValueError(f"Expected {k} tap vectors separated by ';', got {len(parts)}")
    taps = []
    for part in parts:
        bits = [int(b.strip()) for b in part.split(',')]
        if len(bits) != constraint_length:
            raise ValueError(f"Each tap vector must have {constraint_length} bits, got {len(bits)}")
        taps.append(bits)
    return taps

def parse_all_generators(gen_strs, k, m):
    """Parse all n generators into generator_matrix format (comma-separated representation)."""
    constraint_length = 1 + m  # per input line
    generator_matrix = []
    for gen_str in gen_strs:
        taps = parse_generator_matrix(gen_str, k, constraint_length)
        generator_matrix.append(taps)
    return generator_matrix

def load_symbolic_T_from_csv(csv_path):
    """Load CSV of symbolic matrix entries into sympy.Matrix; returns (p_symbol, T_sym)."""
    df = pd.read_csv(csv_path, header=None, dtype=str)
    rows = []
    for i in range(df.shape[0]):
        row = []
        for j in range(df.shape[1]):
            cell = df.iat[i, j]
            if pd.isna(cell):
                cell = "0"
            row.append(sp.sympify(cell))
        rows.append(row)
    T_sym = sp.Matrix(rows)
    p = sp.symbols('p')
    return p, T_sym

def evaluate_symbolic_T(T_sym, p_symbol, pval):
    """Evaluate sympy matrix at pval and return numpy ndarray with row-normalization guard."""
    T_num = np.array([[float(sp.N(entry.subs(p_symbol, pval))) for entry in row] for row in T_sym.tolist()], dtype=float)
    # normalize rows (guard against zero rows)
    with np.errstate(invalid='ignore'):
        row_sums = T_num.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        if zero_rows.any():
            row_sums[zero_rows, :] = 1.0
        T_num = T_num / row_sums
    return T_num

def log_prob_sequence(metrics_seq, state_index, T_num, eps=1e-300):
    """Log-probability of a sequence of metric tuples under T_num transitions."""
    logp = 0.0
    for t in range(len(metrics_seq) - 1):
        a = metrics_seq[t]
        b = metrics_seq[t + 1]
        i = state_index.get(a, None)
        j = state_index.get(b, None)
        if i is None or j is None:
            logp += math.log(eps)
            continue
        pij = T_num[i, j]
        if pij <= 0:
            pij = eps
        logp += math.log(pij)
    return logp

def decode_noisy_outputs_with_trellis(noisy_outputs, generator_matrix, m, k):
    """Decode using code trellis and produce sequence of relative metric tuples."""
    trellis = vm.build_trellis(generator_matrix, m, k)
    cur = tuple([0] * (1 << m))
    seq = [cur]
    for r in noisy_outputs:
        nxt = vm.viterbi_metric_step(list(cur), trellis, r)
        if isinstance(nxt, (list, tuple)):
            nxt_t = tuple(int(x) for x in nxt)
        else:
            try:
                nxt_t = tuple(int(x) for x in list(nxt))
            except Exception:
                raise RuntimeError(f"viterbi_metric_step returned unsupported type: {type(nxt)}")
        seq.append(nxt_t)
        cur = nxt_t
    return seq

def mean_row_KL(P_emp, P_sym, eps=1e-12):
    """Mean of row-wise KL divergences KL(P_emp[row] || P_sym[row])."""
    P_emp_c = np.maximum(P_emp, eps)
    P_sym_c = np.maximum(P_sym, eps)
    kl_rows = np.sum(P_emp_c * (np.log(P_emp_c) - np.log(P_sym_c)), axis=1)
    return float(np.mean(kl_rows))

# ----------------- Learn empirical P1 (cached wrapper) -----------------

@lru_cache(maxsize=128)
def learn_P1_empirical_cached(k, n, m, gens1_tuple, p_val, learn_len, learn_burn, laplace, seed):
    """
    Cached wrapper that enumerates states and learns empirical counts.
    gens1_tuple: tuple of generator strings (for hashability)
    """
    gens1_list = list(gens1_tuple)
    generator_matrix = parse_all_generators(gens1_list, k, m)

    # enumerate states & transitions
    states, transitions, all_r = vm.enumerate_markov_states_allzero(generator_matrix, m, k, n)
    state_index = {s: idx for idx, s in enumerate(states)}
    S = len(states)

    # choose learn_len heuristic
    if learn_len is None:
        learn_len_eff = max(5000, 200 * S)
    else:
        learn_len_eff = learn_len

    # optional seeding
    if seed is not None:
        np.random.seed(seed)

    # simulate long sequence with random_input True
    sim = vm.simulate_markov_sequence(generator_matrix, m, k, n, learn_len_eff, p_val=p_val, random_input=True)
    noisy = sim['noisy_outputs']
    metrics = decode_noisy_outputs_with_trellis(noisy, generator_matrix, m, k)

    # count transitions after burn
    counts = np.zeros((S, S), dtype=float)
    for t in range(learn_burn, len(metrics) - 1):
        a = metrics[t]
        b = metrics[t + 1]
        if a not in state_index or b not in state_index:
            continue
        counts[state_index[a], state_index[b]] += 1.0

    # Laplace smoothing + normalize rows
    P = counts + float(laplace)
    row_sums = P.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.flatten() == 0)
    if zero_rows.any():
        P[zero_rows, :] = 1.0
        row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums

    return (tuple(states), tuple(state_index.items()), P)

# ----------------- Main experiment driver -----------------

def run_experiment(k, n, m, gen1_matrix, gen2_matrix, num_iter, p_vec, save_dir,
                   use_precomputed_symbolic=True, symbolic_dir=None,
                   learn_len=None, learn_burn=200, laplace=1.0, seed=12345):
    np.random.seed(seed)

    print(f"Code parameters: k={k}, n={n}, m={m}")
    print(f"Constraint length per input line: {1+m}")
    print(f"Code rate: {k}/{n}")
    # Note: gen1_matrix and gen2_matrix are expected as lists of tap vectors (n x k x (m+1))
    # For symbolic building we will create strings from these if needed.
    gens1_str = [";".join([",".join(str(b) for b in vec) for vec in out]) for out in gen1_matrix]
    gens2_str = [";".join([",".join(str(b) for b in vec) for vec in out]) for out in gen2_matrix]

    print("Generators (code-1) matrix (n x k x (m+1)):", gen1_matrix)
    print("Generators (code-2) matrix (n x k x (m+1)):", gen2_matrix)

    # choose N-spectrum
    N_spectrum = N_SPECTRUM_BY_M.get(m, N_SPECTRUM_FALLBACK)
    print(f"Using N-spectrum for m={m}: {N_spectrum}")

    # Enumerate states for code-1
    print("Enumerating states for code-1 (allzero convention).")
    states, transitions, all_r = vm.enumerate_markov_states_allzero(gen1_matrix, m, k, n)
    state_index = {s: idx for idx, s in enumerate(states)}
    S = len(states)
    print(f"Number of reachable relative-metric states for code-1 (S): {S}")

    # Try loading symbolic T_sym for code-1 from CSV
    p_sym, T_sym = None, None
    if use_precomputed_symbolic and symbolic_dir is not None:
        candidates = [
            os.path.join(symbolic_dir, f"markov_symT_k{k}_n{n}_m{m}_p_{gens1_str}_{gens2_str}.csv"),
        ]

        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break

        if found:
            try:
                p_sym, T_sym = load_symbolic_T_from_csv(found)
                print("Loaded symbolic T from:", found)
            except Exception as e:
                print("Failed to load symbolic CSV:", found, "error:", e)
                p_sym, T_sym = None, None
        else:
            print("No symbolic CSV found in symbolic_dir.")

    if T_sym is None:
        print("Building symbolic T via vm.build_symbolic_T...")
        p_sym, T_sym = vm.build_symbolic_T(states, transitions, all_r, normalize=True)
        print("Built symbolic T.")

    # Precompute T2 = T_sym evaluated at p=0.5 (reference)
    T2_num = evaluate_symbolic_T(T_sym, p_sym, 0.5)

    learned_cache_info = {}
    os.makedirs(save_dir, exist_ok=True)
    out_rows = []
    start_time_exp = time.time()

    # Main loops
    for N in N_spectrum:
        print(f"\n=== Running experiments for N = {N} ===")
        start_time_N = time.time()

        for p in p_vec:
            print(f"--> p = {p}")

            # Get/learn empirical P1 for this p (cached)
            cache_key = (k, n, m, tuple(gens1_str), float(p), learn_len, learn_burn, laplace, seed)
            if cache_key in learned_cache_info:
                states_cached, state_index_cached, P1_emp = learned_cache_info[cache_key]
                print(f"Using cached empirical P1 for p={p}")
            else:
                print(f"Learning empirical P1 for p={p}...")
                tup = learn_P1_empirical_cached(k, n, m, tuple(gens1_str), float(p), learn_len, learn_burn, laplace, seed)
                states_cached = list(tup[0])
                state_index_cached = dict(tup[1])
                P1_emp = tup[2]
                learned_cache_info[cache_key] = (states_cached, state_index_cached, P1_emp)

            # Remap if needed
            if tuple(states) != tuple(states_cached):
                print("Remapping empirical P1...")
                learned_index = {s: idx for idx, s in enumerate(states_cached)}
                P1_remap = np.zeros_like(P1_emp)
                for i_main, s_main in enumerate(states):
                    if s_main in learned_index:
                        i_learn = learned_index[s_main]
                        for j_main, t_main in enumerate(states):
                            j_learn = learned_index.get(t_main, None)
                            if j_learn is not None:
                                P1_remap[i_main, j_main] = P1_emp[i_learn, j_learn]
                    else:
                        P1_remap[i_main, :] = 1.0 / len(states)

                row_sums = P1_remap.sum(axis=1, keepdims=True)
                zero_rows = (row_sums.flatten() == 0)
                if zero_rows.any():
                    P1_remap[zero_rows, :] = 1.0 / len(states)
                    row_sums = P1_remap.sum(axis=1, keepdims=True)
                P1_emp = P1_remap / row_sums

            # Evaluate symbolic T1 at p for KL comparison
            T1_sym_num = evaluate_symbolic_T(T_sym, p_sym, p)
            kl_mean = mean_row_KL(P1_emp, T1_sym_num)
            print(f"Mean row-KL(P1_emp || T1_sym(p={p})) = {kl_mean:.4e}")

            # Monte Carlo trials
            success_h1 = 0
            fa_h2 = 0
            success_h2 = 0

            pbar = tqdm(total=num_iter, desc=f"p={p:.4f} N={N}", unit="iter")

            for it in range(num_iter):
                # H1 trial
                sim1 = vm.simulate_markov_sequence(gen1_matrix, m, k, n, N, p_val=p, random_input=True)
                noisy1 = sim1['noisy_outputs']
                metrics1 = decode_noisy_outputs_with_trellis(noisy1, gen1_matrix, m, k)
                logp_emp = log_prob_sequence(metrics1, state_index, P1_emp)
                logp_T2 = log_prob_sequence(metrics1, state_index, T2_num)
                if logp_emp > logp_T2:
                    success_h1 += 1

                # H2 trial
                sim2 = vm.simulate_markov_sequence(gen2_matrix, m, k, n, N, p_val=p, random_input=True)
                noisy2 = sim2['noisy_outputs']
                metrics2 = decode_noisy_outputs_with_trellis(noisy2, gen1_matrix, m, k)
                logp_emp2 = log_prob_sequence(metrics2, state_index, P1_emp)
                logp_T2_2 = log_prob_sequence(metrics2, state_index, T2_num)
                if logp_emp2 > logp_T2_2:
                    fa_h2 += 1
                else:
                    success_h2 += 1

                pbar.update(1)

            pbar.close()

            Pd = success_h1 / float(num_iter)
            Pfa = fa_h2 / float(num_iter)
            Pc = (success_h1 + success_h2) / float(2 * num_iter)

            print(f"Results p={p:.4f}, N={N} -> Pd={Pd:.4f}, Pfa={Pfa:.4f}, Pc={Pc:.4f} (KL={kl_mean:.3e})")

            out_rows.append({
                "k": k,
                "n": n,
                "m": m,
                "g1_str": ";".join(gens1_str),
                "g2_str": ";".join(gens2_str),
                "S": S,
                "N": N,
                "p": p,
                "Pd": Pd,
                "Pfa": Pfa,
                "Pc": Pc,
                "mean_row_KL_P1_vs_T1": kl_mean,
                "num_iter": num_iter
            })

        end_time_N = time.time()
        print(f"Completed N={N} in {end_time_N - start_time_N:.1f} seconds.")

    end_time_exp = time.time()
    print(f"\nCompleted all experiments in {end_time_exp - start_time_exp:.1f} seconds.")

    # save CSV
    df_out = pd.DataFrame(out_rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"Pd_hybrid_k{k}_n{n}_m{m}_{gens1_str}_{gens2_str}_{ts}.csv"
    outpath = os.path.join(save_dir, fname)
    os.makedirs(save_dir, exist_ok=True)
    df_out.to_csv(outpath, index=False)
    print("Saved results CSV to:", outpath)

    # PLOTS
    plt.figure(figsize=(9, 6))
    for N in N_spectrum:
        dfN = df_out[df_out['N'] == N]
        plt.plot(dfN['p'].tolist(), dfN['Pc'].tolist(), marker='o', label=f"N={N}")

    plt.xlabel("p (BSC)")
    plt.ylabel("Pd = Probability of correct decision")
    plt.title(f"Pd vs p for rate {k}/{n} m={m}")
    plt.legend()
    plt.grid(True)
    png_pc_vs_p = os.path.join(save_dir, f"Pd_vs_p_k{k}_n{n}_m{m}_{ts}.png")
    plt.savefig(png_pc_vs_p, dpi=150)
    plt.close()
    print("Saved Pc-vs-p plot:", png_pc_vs_p)

    # Plot Pc vs N
    p_idxs = [0, max(0, len(p_vec) // 2), len(p_vec) - 1]
    plt.figure(figsize=(9, 6))
    for idx in p_idxs:
        p_fix = p_vec[idx]
        df_pfix = df_out[df_out['p'] == p_fix]
        plt.plot(df_pfix['N'].tolist(), df_pfix['Pc'].tolist(), marker='o', label=f"p={p_fix}")

    plt.xscale('log')
    plt.xlabel("N (log scale)")
    plt.ylabel("Pd = Probability of correct decision")
    plt.title(f"Pd vs N for rate {k}/{n} m={m}")
    plt.legend()
    plt.grid(True)
    png_pc_vs_N = os.path.join(save_dir, f"Pd_vs_N_k{k}_n{n}_m{m}_{ts}.png")
    plt.savefig(png_pc_vs_N, dpi=150)
    plt.close()
    print("Saved Pc-vs-N plot:", png_pc_vs_N)

    return df_out, outpath, (png_pc_vs_p, png_pc_vs_N)

# ----------------- main -----------------

if __name__ == "__main__":
    print("Interactive Pd_plotter_generalised (octal generator input)")
    # Interactive prompts: k, n, m
    k = int(input("Enter k (inputs per time step, integer >=1): ").strip() or "1")
    n = int(input("Enter n (outputs per time step, integer >=1): ").strip() or "1")
    m = int(input("Enter m (memory, integer >=0): ").strip() or "1")
    print(f"\nYou entered k={k}, n={n}, m={m} (constraint length = m+1 = {m+1})")

    # Read generators for code-1
    print("\n--- Code 1 generators ---")
    gen1_matrix = parse_all_generators_from_octal_inputs(k, n, m)  # returns n x k x (m+1)

    # Read generators for code-2
    print("\n--- Code 2 generators ---")
    gen2_matrix = parse_all_generators_from_octal_inputs(k, n, m)

    # other experiment parameters from defaults
    num_iter = int(input(f"Number of Monte-Carlo iterations per (N,p) [default {DEFAULTS['num_iter']}]: ").strip() or str(DEFAULTS['num_iter']))
    p_vec = DEFAULTS['p_vec']  # still using defaults; can be adjusted in code if needed
    use_precomputed_symbolic = DEFAULTS['use_precomputed_symbolic']
    symbolic_dir = DEFAULTS['symbolic_dir']
    save_dir = DEFAULTS['save_dir']
    seed = int(input(f"Random seed (integer) [default {DEFAULTS['seed']}]: ").strip() or str(DEFAULTS['seed']))
    learn_len = DEFAULTS['learn_len']
    learn_burn = DEFAULTS['learn_burn']
    laplace = DEFAULTS['laplace']

    print("\nRunning hybrid Pd experiment with octal generator input.")
    print(f" k={k}, n={n}, m={m}")
    print(" Proceeding...\n")

    start_time_total = time.time()

    df, csv_path, pngs = run_experiment(
        k, n, m, gen1_matrix, gen2_matrix, num_iter, p_vec, save_dir,
        use_precomputed_symbolic=use_precomputed_symbolic,
        symbolic_dir=symbolic_dir,
        learn_len=learn_len,
        learn_burn=learn_burn,
        laplace=laplace,
        seed=seed
    )

    end_time_total = time.time()
    print(f"\nTotal time: {end_time_total - start_time_total:.1f} seconds.")
    print("\nDone. Results saved to:", csv_path)
    for p in pngs:
        print("Plot:", p)
