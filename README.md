# Detecting Convolutional Codes via a Markovian Statistic

**WCNC 2026 — Reproducibility, Artifact, and Demo Guide**

This repository contains the **complete, camera-ready codebase** used in the paper:

> **Detecting Convolutional Codes via a Markovian Statistic**
> Accepted at *IEEE Wireless Communications and Networking Conference (WCNC) 2026*

The repository implements:

1. The **proposed Markov-chain-based detector** (Sections III–V), and
2. A **parity-template baseline** used solely for comparison (Section IV).

All scripts are fully documented and aligned **equation-by-equation** with the paper.

---

## 1. Scientific Scope of the Repository

### 1.1 Problem Statement (from the paper)

Given a noisy bitstream produced by an **unknown convolutional encoder** over a
binary symmetric channel (BSC), the goal is to decide between two candidate
encoders:
[
\mathcal{H}_1 : \text{Encoder } G_1(D), \qquad
\mathcal{H}_2 : \text{Encoder } G_2(D).
]

The paper proposes a detector based on the **Markovian structure of relative
Viterbi metrics**, and derives its **asymptotic error exponent** analytically.

---

## 2. Repository Structure

```
.
├── viterbi_markov.py          # Core theory: Sections III-A, III-B (Eq. 4–6)
├── alpha_exponent.py          # Error exponent: Section III-C (Eq. 7)
├── parity_eqn_check.py        # Parity-check derivation (baseline, Section IV)
├── comp_parity.py             # Parity-template detector (baseline)
├── Pd_plotter.py              # Hybrid detector experiments (Section V)
├── plots_compare.py           # Post-processing & comparison plots
├── demo_script.py       # Interactive demo script (quick-start)
├── results_experiments/       # Generated CSVs and plots (created at runtime)
└── README.md                  # This file
```

Each file is **self-contained**, executable, and documented at a level suitable
for **artifact evaluation and reproducibility**.

---

## 3. Mathematical–Code Correspondence

| Paper Component                   | Equation / Section | Code File              |
| --------------------------------- | ------------------ | ---------------------- |
| Relative Viterbi metric recursion | Eq. (4), Eq. (5)   | `viterbi_markov.py`    |
| Markov transition probabilities   | Eq. (6)            | `viterbi_markov.py`    |
| Chernoff error exponent           | Eq. (7)            | `alpha_exponent.py`    |
| Parity-check equations            | —                  | `parity_eqn_check.py`  |
| Parity-template detector          | Section IV         | `comp_parity.py`       |
| Hybrid likelihood detector        | Section V          | `Pd_plotter.py`        |
| Interactive demo                  | —                  | `demo_script.py` |

---

## 4. Dependencies and Environment

### 4.1 Required Packages

All code is written in **Python 3.8+** and requires:

```bash
pip install numpy sympy pandas matplotlib tqdm
```

No machine-learning libraries are used.

### 4.2 Recommended Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

---

## 5. Quick Interactive Demo (Recommended First Step)

For a **minimal, user-friendly entry point**, run:

```bash
python demo_script.py
```

This script:

* Prompts the user to either:

  * Select from **predefined example convolutional codes** used in the paper, or
  * Enter **custom code parameters** ((k, n, m)) and generator taps
* Automatically runs the **hybrid Markov-based detector**
* Directly produces plots of:

  * **(P_d) vs (p)** (probability of detection vs channel noise)
  * **(P_d) vs (N)** (probability of detection vs blocklength)

This script is intended for:

* First-time readers of the repository
* Demonstrations and teaching
* Sanity checks and exploratory use

**Note:**
`demo_script.py` uses the same detection logic as `Pd_plotter.py` but with
reduced Monte-Carlo settings for faster execution. It is *not* used to generate
the paper’s figures.

---

## 6. Reproducing the Main Results (Section V)

### Step 1 — Build the Markov Model

```bash
python viterbi_markov.py
```

This:

* Enumerates all reachable **relative Viterbi metric states**
* Constructs the symbolic transition matrix (T(p))
* Saves the matrix for reuse

Corresponds to **Sections III-A and III-B**.

---

### Step 2 — (Optional) Compute the Error Exponent

```bash
python alpha_exponent.py
```

Evaluates the Chernoff information for Markov chains:
[
I_{\mathrm{err}} = \min_{u \in [0,1]} -\log \rho(M(u)),
]
corresponding to **Eq. (7)** in the paper.

---

### Step 3 — Run the Hybrid Detector (Main Experiment)

```bash
python Pd_plotter.py
```

This script:

* Learns an empirical transition matrix (\hat{P}_1)
* Uses (T(p=1/2)) as an uninformative reference
* Performs likelihood-ratio testing on metric sequences

Outputs:

* CSV files with (P_d), (P_c)
* Plots used in **Section V**

---

### Step 4 — Run the Baseline Parity Detector (Section IV)

```bash
python comp_parity.py
```

Implements the **parity-template baseline**, which relies on heuristic
thresholding and does **not** admit an error-exponent analysis.

---

### Step 5 — Generate Comparison Plots

```bash
python plots_compare.py \
  --hybrid results_experiments/Pd_hybrid_results.csv \
  --baseline results_parity/Pd_parity_results.csv
```

Produces the comparative plots reported in the paper.

---

## 7. Notes on Reproducibility

* All Monte-Carlo experiments use **fixed random seeds** by default
* All reported results were generated using this codebase
* No post-hoc tuning or undocumented heuristics are used in the proposed method

The parity-template baseline requires manual threshold selection ((\gamma)),
as explicitly stated in both the paper and the code.

---

## 8. Citation

If you use or build upon this code, please cite:

```bibtex
@inproceedings{kamthankarWCNC2026,
  title     = {Detecting Convolutional Codes via a Markovian Statistic},
  author    = {Shubhankar Abhay Kamthankar, Guneesh Vats and Arti Yardi},
  booktitle = {IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2026}
}
```

---

## 9. Contact

For questions regarding the theory, implementation, or reproducibility of the
results, please contact the corresponding author listed in the WCNC paper.

---
