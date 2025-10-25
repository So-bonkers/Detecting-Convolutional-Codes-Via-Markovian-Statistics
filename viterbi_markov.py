
"""
viterbi_markov.py

Build Viterbi relative-metric Markov chain for ANY k and n:

- Option [0] All-zero input (theoretical assumption, Section III of Best et al.)
- Option [1] Random input exploration (enumerates both input=0 and input=1 branches)

Accepts generator polynomials in comma-separated format:
- For rate k/n code, provide n generators, each as comma-separated list of k tap vectors
- Example for rate 2/3: "1,1;0,1" means g1^(0)=(1,1) and g2^(0)=(0,1)

Extras:
- Simulate a trajectory of metric vectors for given parameters.
- Save parameters, input, outputs, transition matrices, and trajectory to file.
"""

import itertools
import math
from collections import defaultdict, deque
import numpy as np
import sympy as sp
import json
import csv
import os

# ---------- helpers ----------

def parse_generator_matrix(gen_str, k, constraint_length):
    """
    Parse generator string in format: "1,1;0,1" 
    This represents k tap vectors, each of length constraint_length
    Returns: list of k tap vectors (each as list of bits)
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

def state_bits_from_int(s_int, m):
    """Extract m bits from state integer."""
    return [(s_int >> i) & 1 for i in range(m)]

def bits_to_int(bitlist):
    """Convert bit list to integer."""
    val = 0
    for i, b in enumerate(bitlist):
        val |= (int(b) & 1) << i
    return val

def branch_output_and_next_state(state_int, input_bits, generator_matrix, m, k):
    """
    Compute branch output and next state for k input bits.
    
    Args:
        state_int: current state as integer
        input_bits: tuple/list of k input bits
        generator_matrix: list of n generators, each is list of k tap vectors
        m: memory order
        k: number of input bits
    
    Returns:
        (outputs, next_state_int)
    """
    state_bits = state_bits_from_int(state_int, m)
    
    outputs = []
    # For each output bit (generator)
    for gen in generator_matrix:
        output_bit = 0
        # For each input line
        for i in range(k):
            # gen[i] is the tap vector for input line i
            # Convolve with [input_bits[i]] + state_bits
            x = [input_bits[i]] + state_bits[:m]
            for j in range(min(len(gen[i]), len(x))):
                output_bit ^= (gen[i][j] & x[j])
        outputs.append(output_bit)
    
    # Next state: shift in the k new input bits
    new_regs = list(input_bits) + state_bits[:max(0, m-k)] if m > 0 else []
    new_regs = new_regs[:m]  # Keep only m bits
    
    next_int = bits_to_int(new_regs)
    
    return tuple(outputs), next_int

def hamming(a, b):
    """Hamming distance between two sequences."""
    return sum(ai != bi for ai, bi in zip(a, b))

def build_trellis(generator_matrix, m, k):
    """
    Build trellis with incoming branches for each state.
    
    Returns:
        incoming: dict mapping state -> list of (prev_state, input_tuple, output_tuple)
    """
    n_states = 1 << m
    incoming = {s: [] for s in range(n_states)}
    
    # Enumerate all possible input combinations (2^k possibilities)
    for s in range(n_states):
        for input_bits in itertools.product([0, 1], repeat=k):
            out, ns = branch_output_and_next_state(s, input_bits, generator_matrix, m, k)
            incoming[ns].append((s, input_bits, out))
    
    return incoming

def viterbi_metric_step(D_prev, incoming, r):
    """
    One Viterbi decoding step: update path metrics and normalize.
    
    Args:
        D_prev: list of path metrics for each state
        incoming: incoming branches for each state
        r: received n-bit vector
    
    Returns:
        normalized metric vector as tuple
    """
    num_states = len(D_prev)
    D_next = [math.inf] * num_states
    
    for ns in range(num_states):
        best = math.inf
        for (ps, u, out) in incoming[ns]:
            branch_metric = hamming(out, r)
            val = D_prev[ps] + branch_metric
            if val < best:
                best = val
        D_next[ns] = best
    
    # Normalize by subtracting minimum
    minv = min(D_next)
    return tuple(int(v - minv) for v in D_next)

# ---------- enumeration modes ----------

def enumerate_markov_states_allzero(generator_matrix, m, k, n):
    """Enumerate reachable states assuming all-zero input (theoretical)."""
    incoming = build_trellis(generator_matrix, m, k)
    num_states = 1 << m
    
    # All possible received vectors (2^n possibilities)
    all_r = [tuple(bits) for bits in itertools.product([0, 1], repeat=n)]
    
    start = tuple([0] * num_states)
    queue = deque([start])
    visited = {start: 0}
    states = [start]
    transitions = defaultdict(lambda: defaultdict(list))
    
    while queue:
        cur = queue.popleft()
        cur_idx = visited[cur]
        
        # Only consider all-zero input
        for r in all_r:
            nxt = viterbi_metric_step(list(cur), incoming, r)
            if nxt not in visited:
                visited[nxt] = len(states)
                states.append(nxt)
                queue.append(nxt)
            transitions[cur_idx][visited[nxt]].append(r)
    
    return states, transitions, all_r

def enumerate_markov_states_random_input(generator_matrix, m, k, n):
    """Enumerate reachable states assuming random input bits."""
    incoming = build_trellis(generator_matrix, m, k)
    num_states = 1 << m
    
    # All possible received vectors (2^n possibilities)
    all_r = [tuple(bits) for bits in itertools.product([0, 1], repeat=n)]
    
    start = tuple([0] * num_states)
    queue = deque([start])
    visited = {start: 0}
    states = [start]
    transitions = defaultdict(lambda: defaultdict(list))
    
    while queue:
        cur = queue.popleft()
        cur_idx = visited[cur]
        
        # Consider all possible k-bit inputs
        for input_bits in itertools.product([0, 1], repeat=k):
            for prev_state in range(num_states):
                out, ns = branch_output_and_next_state(prev_state, input_bits, generator_matrix, m, k)
                
                for r in all_r:
                    nxt = viterbi_metric_step(list(cur), incoming, r)
                    if nxt not in visited:
                        visited[nxt] = len(states)
                        states.append(nxt)
                        queue.append(nxt)
                    transitions[cur_idx][visited[nxt]].append(r)
    
    return states, transitions, all_r

# ---------- T matrices ----------

def build_symbolic_T(states, transitions, all_r, normalize=True):
    """Build symbolic transition matrix with parameter p."""
    p = sp.symbols('p')
    msize = len(states)
    T = sp.MutableDenseMatrix(msize, msize, lambda i, j: sp.Integer(0))
    
    n = len(all_r[0])
    # Probability of received vector r under BSC with crossover prob p
    prob_r = {r: p**(sum(r)) * (1-p)**(n - sum(r)) for r in all_r}
    
    for i in range(msize):
        row_sum = sp.Integer(0)
        for j, rlist in transitions[i].items():
            s = sp.Integer(0)
            for r in rlist:
                s += prob_r[r]
            T[i, j] = sp.simplify(s)
            row_sum += s
        
        if normalize and row_sum != 0:
            for j in range(msize):
                T[i, j] = sp.simplify(T[i, j] / row_sum)
    
    return p, sp.simplify(T)

def numeric_eval_T(T_sym, p_sym, p_val):
    """Evaluate symbolic matrix T at specific p value."""
    fT = np.zeros(T_sym.shape, dtype=float)
    for i in range(T_sym.shape[0]):
        for j in range(T_sym.shape[1]):
            fT[i, j] = float(sp.N(T_sym[i, j].subs(p_sym, p_val)))
    
    # Renormalize rows (numerical stability)
    for i in range(len(fT)):
        s = fT[i, :].sum()
        if s > 0:
            fT[i, :] /= s
    
    return fT

# ---------- simulation ----------

def simulate_markov_sequence(generator_matrix, m, k, n, N, p_val=0.1, random_input=False, seed=None):
    """Simulate trajectory of metric vectors + output bits for N steps."""
    if seed is not None:
        np.random.seed(seed)
    
    incoming = build_trellis(generator_matrix, m, k)
    cur = tuple([0] * (1 << m))
    seq_metrics = [cur]
    
    # Build encoder outgoing transitions
    outgoing = {}
    n_states = 1 << m
    for s in range(n_states):
        outgoing[s] = {}
        for input_bits in itertools.product([0, 1], repeat=k):
            out, ns = branch_output_and_next_state(s, input_bits, generator_matrix, m, k)
            outgoing[s][input_bits] = (ns, out)
    
    state_enc = 0
    inputs = []
    outputs = []
    noisy_outputs = []
    
    for _ in range(N):
        # Generate k random input bits
        if random_input:
            u = tuple(np.random.randint(0, 2) for _ in range(k))
        else:
            u = tuple([0] * k)
        
        inputs.append(u)
        ns, out = outgoing[state_enc][u]
        state_enc = ns
        outputs.append(out)
        
        # Add noise (BSC with prob p_val)
        noisy = tuple([bit ^ np.random.binomial(1, p_val) for bit in out])
        noisy_outputs.append(noisy)
        
        nxt = viterbi_metric_step(list(cur), incoming, noisy)
        seq_metrics.append(nxt)
        cur = nxt
    
    return {
        "inputs": inputs,
        "outputs": outputs,
        "noisy_outputs": noisy_outputs,
        "metrics": seq_metrics,
    }

# ---------- pretty print ----------

def print_symbolic_matrix_with_labels(T_sym, states, title="Symbolic Matrix"):
    """Print symbolic transition matrix."""
    print(f"\n{title}:")
    header = " " * 12 + "".join([f"{j:>20d}" for j in range(len(states))])
    print(header)
    for i in range(len(states)):
        rowstr = " ".join([f"{str(sp.simplify(T_sym[i, j])):>20}" for j in range(len(states))])
        print(f"{i:3d} {states[i]} {rowstr}")

def print_matrix_with_labels(M, states, title="Matrix"):
    """Print numeric matrix."""
    print(f"\n{title}:")
    header = " " * 12 + "".join([f"{j:>14d}" for j in range(len(states))])
    print(header)
    for i, row in enumerate(M):
        rowstr = " ".join([f"{val:14.6f}" for val in row])
        print(f"{i:3d} {states[i]} {rowstr}")

def save_symbolic_matrix_to_csv(T_sym, states, gen_strs, m, k, n, p_symbol, output_dir="results"):
    """Save the symbolic transition matrix to a CSV file."""
    filename = f"markov_symT_k{k}_n{n}_m{m}_p_{gen_strs}.csv"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["from\\\\to"] + [str(j) for j in range(len(states))]
        writer.writerow(header)
        
        for i in range(len(states)):
            row = [str(i)] + [str(T_sym[i, j]) for j in range(len(states))]
            writer.writerow(row)
    
    print(f"Symbolic transition matrix saved to {filepath}")

# ---------- main ----------

def main():
    print("\nViterbi Markov Chain Builder (comma-separated generator input)\n")
    print("Generator format: For each output, provide k tap vectors separated by ';'")
    print("Example for rate 2/3: Generator 1 could be '1,1;0,1'")
    print("  This means g1^(0)=(1,1) and g2^(0)=(0,1)\n")
    
    # Input k (number of input bits per step)
    k = int(input("Enter k (number of input bits per step, e.g., 1, 2, 3...): ").strip() or "1")
    
    # Input n (number of output bits per step)
    n = int(input("Enter n (number of output bits per step, e.g., 2, 3, 4...): ").strip() or "2")
    
    # Input m (memory order)
    m = int(input("Enter m (memory order, e.g., 1, 2, 3, 4...): ").strip() or "1")
    
    constraint_length = 1 + m
    
    print(f"\nCode parameters: k={k}, n={n}, m={m}")
    print(f"Constraint length = {constraint_length}")
    print(f"Code rate = {k}/{n}")
    print(f"\nProvide {n} generators (each as {k} tap vectors of length {constraint_length}):")
    print(f"Format: tap1_bit0,tap1_bit1,...;tap2_bit0,tap2_bit1,...;...")
    
    gen_strs = []
    generator_matrix = []
    
    for j in range(n):
        while True:
            gen_str = input(f"  Generator {j+1}: ").strip()
            try:
                taps = parse_generator_matrix(gen_str, k, constraint_length)
                generator_matrix.append(taps)
                gen_strs.append(gen_str)
                break
            except ValueError as e:
                print(f"    Error: {e}. Please try again.")
    
    print(f"\nParsed generators:")
    for j, gen in enumerate(generator_matrix):
        print(f"  Output {j+1}: {gen}")
    
    # Input mode
    mode = input("\nInput mode: [0] all-zero (default), [1] random input exploration: ").strip()
    
    print(f"\nBuilding Markov chain (this may take time for large k, n, m)...")
    
    if mode == "1":
        states, transitions, all_r = enumerate_markov_states_random_input(generator_matrix, m, k, n)
    else:
        states, transitions, all_r = enumerate_markov_states_allzero(generator_matrix, m, k, n)
    
    print(f"Number of reachable metric states: {len(states)}")
    
    # Build symbolic transition matrix
    print("\nBuilding symbolic transition matrix T(p)...")
    p_sym, T_sym = build_symbolic_T(states, transitions, all_r, normalize=True)
    
    # Evaluate at specific p
    p_val = float(input("\nEnter crossover probability p (e.g., 0.1): ").strip() or "0.1")
    T_num = numeric_eval_T(T_sym, p_sym, p_val)
    
    # Save symbolic matrix
    save_symbolic_matrix_to_csv(T_sym, states, gen_strs, m, k, n, p_sym, output_dir="results/symbolic_matrices")
    
    # Optionally simulate
    simulate = input("\nSimulate a trajectory? [y/N]: ").strip().lower()
    if simulate == "y":
        N = int(input("Enter number of steps N (e.g., 20): ").strip() or "20")
        seq = simulate_markov_sequence(generator_matrix, m, k, n, N, p_val, random_input=(mode == "1"))
        
        print("\nInputs:", seq["inputs"])
        print("Outputs:", seq["outputs"])
        print("Noisy outputs:", seq["noisy_outputs"])
        print("\nMetric sequence:")
        for t, s in enumerate(seq["metrics"]):
            print(f"  t={t:2d}: {s}")
        
        # Save to files
        base = f"markov_k{k}_n{n}_m{m}_p{p_val}"
        os.makedirs("results", exist_ok=True)
        json_path = os.path.join("results", base + ".json")
        csv_path = os.path.join("results", base + ".csv")
        
        # Save JSON
        data = {
            "parameters": {"k": k, "n": n, "m": m, "generators": gen_strs, "p": p_val},
            "inputs": [list(u) for u in seq["inputs"]],
            "outputs": [list(o) for o in seq["outputs"]],
            "noisy_outputs": [list(o) for o in seq["noisy_outputs"]],
            "metrics": seq["metrics"],
            "T_numeric": T_num.tolist(),
        }
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Save CSV (metrics only)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "metric_vector"])
            for t, s in enumerate(seq["metrics"]):
                writer.writerow([t, s])
        
        print(f"\nSaved results to {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
