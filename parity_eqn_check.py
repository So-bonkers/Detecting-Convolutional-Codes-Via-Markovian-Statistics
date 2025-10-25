#!/usr/bin/env python3
"""
parity_eq_check_generlised.py  (fixed parser)

Generalised conv-parity-per-equation tool (defaults, non-interactive).
Parser now prefers semicolon-separated bit-vector format when ';' is present,
and otherwise chooses octal vs bit-vector robustly.
"""

import random
import csv
import re
from collections import defaultdict, Counter
from typing import List, Tuple

# ------------------------- USER CONFIG / DEFAULTS -------------------------
K_input = 20            # number of input bits before termination
P_BSC = 0.1             # BSC flip probability for example noisy run
SAVE = True             # save CSV output
INTERLEAVE = False      # whether coded stream is interleaved per time
RANDOM_INPUT = True
SEED = 123

# Code specification defaults (edit to run different codes)
k = 1   # number of input streams
n = 3   # number of outputs

# Default generator rows (human-friendly MSB-first tokens). One string per output.
# Example: rate 1/3, memory m=2 (K=3) written in bit-vector MSB-first with ';' separating inputs
gen_rows = [
    "1,1,1",   # g1 = 7₈ = 1 + D + D²
    "1,0,1",   # g2 = 5₈ = 1 + D²
    "0,1,1"    # g3 = 3₈ = 1 + D
]



# If the generators you typed are MSB-first, set this True to convert to LSB-first internally.
GENERATORS_ENTERED_MSB_FIRST = True

# ------------------------- helpers -------------------------
def octal_to_poly(octal_str: str) -> List[int]:
    """Convert octal string (e.g. '3') to LSB-first bit list [bit0, bit1, ...]."""
    val = int(octal_str, 8)
    bits = []
    bl = val.bit_length()
    for i in range(bl):
        bits.append((val >> i) & 1)
    return bits

def ensure_length(bits: List[int], length: int) -> List[int]:
    if len(bits) >= length:
        return bits[:length]
    return bits + [0] * (length - len(bits))

def _looks_like_octal_tokens(tokens: List[str]) -> bool:
    """Return True if every token is a valid octal literal (chars 0-7)"""
    return all(re.fullmatch(r'[0-7]+', t) for t in tokens)

def parse_generator_row(gen_str: str, k: int, K: int) -> List[List[int]]:
    """
    Parse one generator-row string describing one output.

    Two formats accepted:
      - bit-vector style: 'b0,b1,...; b0,b1,...' (semicolon separates k input polynomials).
                          Each part should have K tokens '0' or '1'.
      - octal style: '3,1' or '3;1' where each token is an octal integer (LSB=current tap).

    Heuristic:
      * If ';' appears -> treat as bit-vector style (unambiguous).
      * Else, if any token has length >1 or contains non 0/1 -> treat as octal.
      * Else (single-digit tokens with no ';') treat as bit-vector.
    """
    s = gen_str.strip()
    # If semicolon present, prefer bit-vector style
    if ';' in s:
        parts = [p.strip() for p in s.split(';') if p.strip() != ""]
        if len(parts) != k:
            raise ValueError(f"Expected {k} parts separated by ';' in bit-vector style, got: {parts}")
        taps = []
        for p in parts:
            tokens = [t.strip() for t in p.split(',') if t.strip() != ""]
            if len(tokens) != K:
                raise ValueError(f"Bit-vector part must have {K} tokens (0/1), got {tokens}")
            if any(tok not in ('0','1') for tok in tokens):
                raise ValueError(f"Bit-vector tokens must be 0 or 1, got {tokens}")
            taps.append([int(x) for x in tokens])
        return taps

    # No semicolon: decide based on tokens
    cleaned = s.replace(';', ',')
    tokens = [t.strip() for t in cleaned.split(',') if t.strip() != ""]
    if len(tokens) != k:
        # ambiguous: maybe user wrote comma-separated bit-vector parts without ';':
        # e.g. "1,1,1,1,0,1,0,1" — try to split into k parts of length K
        if len(tokens) == k * K:
            # interpret as concatenated bit-vectors: split into k blocks of size K
            taps = []
            for j in range(k):
                block = tokens[j*K:(j+1)*K]
                if any(tok not in ('0','1') for tok in block):
                    raise ValueError(f"Concatenated bit-vector tokens must be 0/1, got {block}")
                taps.append([int(x) for x in block])
            return taps
        raise ValueError(f"Cannot parse generator row '{gen_str}': expected {k} tokens, got {len(tokens)}")

    # Now len(tokens) == k
    # If any token length > 1 or tokens contain non-binary digits -> prefer octal
    if any(len(tok) > 1 or (tok not in ('0','1')) for tok in tokens):
        if not _looks_like_octal_tokens(tokens):
            raise ValueError(f"Tokens look neither bit-vectors nor octal: {tokens}")
        # parse octal tokens: convert each to LSB-first bit list length K
        taps = [ensure_length(octal_to_poly(tok), K) for tok in tokens]
        return taps
    else:
        # tokens are single 0/1 digits and no semicolon -> treat as bit-vector form: tokens correspond to LSB..MSB?
        # We'll treat them as K-bit lists for each input (here tokens==k and K==1 typically)
        # For the common case (K>1) earlier branch would have matched. So here tokens are probably of length 1.
        taps = [[int(tok)] for tok in tokens]
        # pad to K
        taps = [ensure_length(t, K) for t in taps]
        return taps

def popcount(x: int) -> int:
    return bin(x).count("1")

# ------------------------- encoder for k x n conv code -------------------------
def encode_conv_k_n(u_streams: List[List[int]],
                    gen_matrix_bits: List[List[List[int]]],
                    m: int,
                    terminate: bool = True) -> Tuple[List[List[int]], int]:
    K = m + 1
    k_local = len(u_streams)
    n_local = len(gen_matrix_bits)
    if any(len(g) != k_local for g in gen_matrix_bits):
        raise ValueError("Generator matrix mismatch.")
    N = len(u_streams[0])
    if any(len(u) != N for u in u_streams):
        raise ValueError("Input lengths differ.")
    u = [list(s) + [0] * m for s in u_streams] if terminate else [list(s) for s in u_streams]
    T = len(u[0])
    v = [[0] * T for _ in range(n_local)]
    for t in range(T):
        hist = []
        for j in range(k_local):
            val = 0
            for r in range(K):
                ti = t - r
                if ti >= 0:
                    val |= (u[j][ti] & 1) << r
            hist.append(val)
        for i in range(n_local):
            acc = 0
            for j in range(k_local):
                bits = gen_matrix_bits[i][j]
                mask = 0
                for r, b in enumerate(bits):
                    if b:
                        mask |= (1 << r)
                acc ^= (popcount(mask & hist[j]) & 1)
            v[i][t] = acc
    return v, T

# ------------------------- parity template builder -------------------------
def _build_parity_rows_from_bits(gen_matrix_bits: List[List[List[int]]], K: int, terminate: bool = True, verbose=False):
    n_local = len(gen_matrix_bits)
    k_local = len(gen_matrix_bits[0])
    m_local = K - 1
    T = K + m_local if terminate else K
    equations = []
    raw_counts_by_t = {}
    for t in range(T):
        cnt = Counter()
        for i in range(n_local):
            for j in range(k_local):
                taps = gen_matrix_bits[i][j]
                for r, bit in enumerate(taps):
                    if bit and (t - r) >= 0:
                        cnt[(f'v{i}', t - r)] += 1
        positions = [pos for pos, c in cnt.items() if (c & 1) == 1]
        positions_sorted = sorted(positions, key=lambda x: (x[0], x[1]))
        if positions_sorted:
            equations.append(positions_sorted)
        raw_counts_by_t[t] = dict(cnt)
        if verbose and t < 6:
            print(f"[debug] t={t}: raw counts => {raw_counts_by_t[t]}")
    uniq_eqs = []
    seen = set()
    for e in equations:
        key = tuple(e)
        if key not in seen:
            seen.add(key)
            uniq_eqs.append(e)
    return uniq_eqs, T, raw_counts_by_t

def build_parity_rows_kxn(gen_matrix_bits: List[List[List[int]]], K: int, terminate: bool = True, verbose=False):
    uniq, T, raw = _build_parity_rows_from_bits(gen_matrix_bits, K, terminate, verbose=verbose)
    return uniq, T

def pattern_from_positions(positions: List[Tuple[str, int]]):
    if not positions:
        return tuple(), None
    positions_sorted = sorted(positions, key=lambda x: (x[0], x[1]))
    idxs = [idx for (_, idx) in positions_sorted]
    anchor = max(idxs)
    offsets = tuple(sorted([(kind, idx - anchor) for kind, idx in positions_sorted], key=lambda x: (x[0], x[1])))
    return offsets, anchor

def offsets_to_symbol(offsets: Tuple[Tuple[str,int], ...]):
    if not offsets:
        return "0 = 0 (mod2)"
    parts = []
    for kind, off in offsets:
        if off == 0:
            parts.append(f"{kind}[t]")
        else:
            if off < 0:
                parts.append(f"{kind}[t{off}]".replace('+',''))
            else:
                parts.append(f"{kind}[t+{off}]")
    return " + ".join(parts) + " = 0 (mod2)"

# ------------------------- CSV helpers -------------------------
def make_fname_base(gen_rows_local, K, p):
    gens = "_".join([g.replace(" ", "").replace(";", "-").replace(",", "") for g in gen_rows_local])
    return f"conv_parity_{gens}_K{K}_p{p:.2f}"

def save_streams_csv(filename_base, K, T, m, u_streams, v_streams, noisy_by_time):
    path = f"{filename_base}_streams.csv"
    t_start = -(m - 1)
    n_local = len(v_streams)
    k_local = len(u_streams)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t"] + [f"u{j}" for j in range(k_local)] + [f"v{i}" for i in range(n_local)] + [f"y{i}" for i in range(n_local)]
        w.writerow(header)
        for idx in range(T):
            t_display = t_start + idx
            u_vals = [u_streams[j][idx] if idx < len(u_streams[j]) and idx < K else "" for j in range(k_local)]
            v_vals = [v_streams[i][idx] for i in range(n_local)]
            y_vals = [noisy_by_time[idx][i] for i in range(n_local)]
            w.writerow([t_display] + u_vals + v_vals + y_vals)
    return path

def save_parity_csv(filename_base, groups):
    """
    Save parity templates to CSV with columns:
      template_id, symbolic_template, anchors (semicolon-separated)
    groups: dict mapping pattern -> list of anchors
    """
    path = f"{filename_base}_parity.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["template_id", "symbolic_template", "anchors"])
        for tid, (pat, anchors) in enumerate(groups.items()):
            sym = offsets_to_symbol(pat)
            anchors_str = ";".join(str(a) for a in anchors)
            w.writerow([tid, sym, anchors_str])
    return path

# ------------------------- main -------------------------
def main():
    print("Running generalized parity eq check (fixed parser, defaults)...")
    # infer K from gen_rows
    K_candidates = []
    for row in gen_rows:
        if ';' in row:
            # treat as bit-vector style: use token count from first semicolon-part
            parts = [p.strip() for p in row.split(';') if p.strip() != ""]
            if parts:
                toks = [t.strip() for t in parts[0].split(',') if t.strip() != ""]
                K_candidates.append(len(toks))
            continue
        # no semicolon: check tokens
        cleaned = row.replace(';', ',')
        tokens = [t.strip() for t in cleaned.split(',') if t.strip() != ""]
        if all(re.fullmatch(r'[0-7]+', t) for t in tokens) and any(len(t) > 1 for t in tokens):
            # octal tokens with length>1 hint at octal use, infer max bit-length
            maxb = 0
            for t in tokens:
                v = int(t, 8)
                if v.bit_length() > maxb:
                    maxb = v.bit_length()
            if maxb == 0:
                maxb = 1
            K_candidates.append(maxb)
        else:
            # fallback: treat as bit-vector tokens in first semicolon-less part
            toks = tokens
            K_candidates.append(len(toks))
    if not K_candidates:
        raise RuntimeError("Could not infer constraint length K from gen_rows.")
    K = max(K_candidates)
    m = K - 1

    # parse generator rows into bit-lists (in the order typed)
    gen_matrix_bits = []
    for row in gen_rows:
        taps = parse_generator_row(row, k, K)
        taps = [ensure_length(t, K) for t in taps]
        gen_matrix_bits.append(taps)

    # convert MSB->LSB if user-entered polynomials are MSB-first
    if GENERATORS_ENTERED_MSB_FIRST:
        gen_matrix_bits = [
            [list(reversed(bits)) for bits in row]
            for row in gen_matrix_bits
        ]

    print(f"\nCode: k={k}, n={n}, inferred K={K}, memory m={m}")
    for i, row in enumerate(gen_matrix_bits):
        print(f" v{i}: {row}")

    uniq_eqs, T = build_parity_rows_kxn(gen_matrix_bits, K, terminate=True, verbose=False)
    groups = defaultdict(list)
    for pos in uniq_eqs:
        pat, anchor = pattern_from_positions(pos)
        groups[pat].append(anchor)
    for pat in groups:
        groups[pat] = sorted(groups[pat])

    print(f"\nBuilt {len(uniq_eqs)} parity templates.")
    for pat, anchors in groups.items():
        sym = offsets_to_symbol(pat)
        print(f"{sym}  anchors={anchors}")

    rng_master = random.Random(SEED)
    u_streams = []
    for j in range(k):
        r = random.Random(SEED + j * 10007)
        u_streams.append([r.randint(0,1) for _ in range(K_input)] if RANDOM_INPUT else [0] * K_input)

    v_streams, T = encode_conv_k_n(u_streams, gen_matrix_bits, m, terminate=True)
    # flatten
    if INTERLEAVE:
        coded_flat = []
        for t in range(T):
            for i in range(n):
                coded_flat.append(v_streams[i][t])
    else:
        coded_flat = []
        for i in range(n):
            coded_flat.extend(v_streams[i])

    rng_bsc = random.Random(SEED + 1)
    noisy_flat = [b ^ (1 if rng_bsc.random() < P_BSC else 0) for b in coded_flat]

    noisy_by_time = []
    if INTERLEAVE:
        for t in range(T):
            noisy_by_time.append([noisy_flat[t * n + i] for i in range(n)])
    else:
        for t in range(T):
            noisy_by_time.append([noisy_flat[i * T + t] for i in range(n)])

    t_start = -(m - 1)
    print("\nSample encoded outputs (first 40 bits):")
    for i, v in enumerate(v_streams):
        print(f" v{i}: {v[:min(len(v), 40)]}")
    print(f"\nNoisy bits sample (first 8 time instants):")
    for i in range(min(8, T)):
        print(f" t={t_start + i:3d}: {noisy_by_time[i]}")

    if SAVE:
        base = make_fname_base(gen_rows, K, P_BSC)
        parity_path = save_parity_csv(base, groups)
        print(f"\nSaved parity templates CSV -> {parity_path}")


    print("\nDone.")

if __name__ == "__main__":
    main()
