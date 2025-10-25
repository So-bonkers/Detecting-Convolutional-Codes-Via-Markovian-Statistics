# README — Detecting Convolutional Codes via Markovian statistics
This repository contains the complete codebase used in our paper on convolutional code detection using **Markov-modelling of Viterbi metrics** and **parity-template hypothesis testing**.
All scripts are self-contained and support **arbitrary code rates** (k/n), unified under the generalized framework described in the paper.

---

##  Repository Overview

| File                                                                                | Description                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`viterbi_markov.py`**                                                             | Constructs the **Viterbi relative-metric Markov chain**, enumerates reachable metric states, and builds the symbolic transition matrix (T(p)) for a given decoder code. This forms the analytical backbone for detection performance computation. |
| **`parity_eq_check.py`**                                                            | Computes **parity-check equations** for a given convolutional encoder, allowing verification and construction of parity templates directly from generator polynomials.                                                                            |
| **`comp_parity.py`**                                                                | Implements **parity-template detection** using the parity-check equations from `parity_eq_check.py`. Supports arbitrary rate (k/n) encoders and allows template-based hypothesis testing between two codes.                                       |
| **`Pd_plotter.py`**                                                                 | Runs the **hybrid detection experiment** combining empirical learning (Monte Carlo simulation) and symbolic Markov analysis. Produces Pd/Pc vs p or Pd vs N curves used in the paper’s main results.                                              |
| **`plots_compare.py`**                                                           | Generates comparative plots between different methods or configurations using CSV outputs from experiments (e.g., hybrid vs parity-template). Plots are rendered on a log-scale for clarity.                                                      |

---

## Setup Instructions

### Dependencies

All scripts are written in Python 3.8+ and require the following packages:

```bash
pip install numpy sympy pandas matplotlib tqdm
```

Optional (for extended plotting):

```bash
pip install scipy seaborn
```

### Recommended Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

---

## Quickstart Guide

### 1. Build Symbolic Transition Matrix (Markov Model)

Run:

```bash
python viterbi_markov.py
```

The script interactively asks for:

* (k, n, m) (code rate and memory)
* Generator polynomials (in octal or binary matrix form)

It enumerates all reachable **relative-metric states** and builds a **symbolic transition matrix** (T(p)) describing state evolution under BSC crossover probability (p).

Output:

* Symbolic matrix (T(p))
* Reachable state list
* Optionally saved CSV for reuse in later experiments

---

### 2. Parity-Template Discovery

Run:

```bash
python parity_eq_check.py
```

This computes the **parity-check equations** from input generator polynomials.
It supports multiple formats (octal, binary vectors, semicolon-separated lists).
The output can be directly fed into `comp_parity.py` for template-based detection.

---

### 3. Parity-Template Detection Experiment

Run:

```bash
python comp_parity.py
```

This performs detection using **template satisfaction counts** between two encoders.
It accepts interactive input for both encoders and automatically generates and evaluates parity templates.

You can fix a single template for all N values and run detection across a range of blocklengths or noise probabilities.

---

### 4. Hybrid Markov-Based Detection (Main Experiment)

Run:

```bash
python Pd_plotter.py
```

This script implements the **hybrid framework** from the paper:

* Builds or loads the symbolic Markov matrix (T(p)) for the decoder.
* Learns empirical transition probabilities from long simulated sequences.
* Performs hypothesis testing between the two models.
* Produces Pd and Pc vs p or N plots.

Default parameters (editable at the top of the file):

* `genc1`, `genc2`: encoder generator lists
* `num_iter`: number of Monte Carlo trials
* `p_vec`: list of channel error probabilities
* `save_dir`: directory for CSVs and plots

Output:

* `results_experiments/` folder with Pd/Pc CSVs and PNG plots

### 5. Comparative Plotting

Run:
```bash
python plots_compare.py
```

This script compares CSV results from different runs (e.g., hybrid vs parity-template).
It produces side-by-side plots with logarithmic y-axes for better visibility near low error probabilities.


##  Outputs

Each experiment produces:

* **CSV files** — containing Pd, Pc, and standard deviations for each (p) and (N)
* **PNG plots** — automatically generated from results
* Default directories:

  * `results_experiments/` (hybrid method)
  * `results_parity/` (parity-template method)
  * `plots/` (comparative plots)

---

## Typical Workflow

1. **Compute symbolic Markov matrix** using `viterbi_markov.py` for the decoder.
2. **Learn empirical transition matrices** for both codes using Monte Carlo.
3. **Run Pd vs p or N detection curves** using `Pd_plotter.py`.
4. **Optionally run parity-template experiments** using `comp_parity.py`.
5. **Plot results** using `plots_compare.py`.

This reproduces the main and comparative figures reported in the paper.
