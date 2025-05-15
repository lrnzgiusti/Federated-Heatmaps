# Federated Heatmaps under Distributed Differential Privacy with Secure Aggregation

<p align="center">
  <b>An end-to-end, research-grade Python toolkit for building adaptive, hierarchical location heatmaps</b><br>
  <i>Pure-ε Differential Privacy | Client-Side Discrete Noise | Dropout-Robust Secure Aggregation</i>
</p>

This repository re-implements the methods from **“Towards Sparse Federated Analytics: Location Heatmaps under Distributed Differential Privacy with Secure Aggregation”** (Bagdasaryan *et al.* 2022).

---

## 🚀 Key Features

* **Adaptive quadtree partitioning** – iteratively splits *and* collapses grid cells using noisy counts and budget-aware thresholds.
* **Distributed discrete-Laplace mechanism** – clients locally add *integer* (Pólya-derived) noise, apply modulo clipping, then participate in SecAgg → yields *pure ε-DP*.
* **Secure Aggregation simulator** – sharded summation with dropout masking; fully vectorised NumPy backend.
* **Multi-location extension** – users can report up to *k* visits; L1-normalised weights with stochastic rounding maintain sensitivity 1.
* **Turn-key evaluation & plotting** – recreate ground-truth/DP heatmaps, compute MSE/L₁, and visualise results.

---

## 📦 Installation

```bash
# 1. Clone
$ git clone https://github.com/<your-handle>/federated-heatmaps.git
$ cd federated-heatmaps

# 2. Create & activate a virtual-env (optional but recommended)
$ python -m venv .venv
$ source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
$ pip install -r requirements.txt
```

---

## ⚙️ Configuration

All parameters live in a single YAML file that mirrors the `Config` dataclass hierarchy.
Create `config.yaml` (or copy `examples/config_default.yaml`) and tweak as needed:

```yaml
privacy:
  eps_total: 1.0        # overall ε
  delta_drop: 0.05      # worst-case client drop-out per shard

secagg:
  S_max: 10000          # max users per SecAgg shard
  m: 65536              # modulus 2^16

algorithm:
  U_alg1: 10000         # users sampled per Algorithm 1 query
  c_alg2: 0.1           # σ̃ calibration constant (Eq. 2)
  b_alg2: 2.0           # budget expansion factor (Eq. 4)

tree:
  max_depth: 10         # quadtree depth
  grid_width: 256       # fine-grid width  (must be ≤ 2^{max_depth})
  grid_height: 256      # fine-grid height (must be ≤ 2^{max_depth})

multi_loc:
  gamma_scaling: 100.0  # ≥1 -- increases precision for multi-loc encoding

verbose: true
```

### Programmatic use

```python
from federated_heatmaps.config import Config
cfg = Config.from_yaml("config.yaml")
```

The nested sub-configs are also available directly: `cfg.privacy.eps_total`, `cfg.tree.max_depth`, …

---

## ▶️ Running a simulation

```bash
# Make sure config.yaml exists in the repo root, then simply run:
$ python -m federated_heatmaps.main
```

By default the script:

1. Loads `config.yaml`.
2. Generates a synthetic population (`clustered`, `uniform`, or `multi_uniform`).
3. Runs Algorithm 2 (adaptive histogram) until the global budget is exhausted.
4. Prints metrics and — if Matplotlib is installed — displays the reconstructed true vs. DP heatmaps.

> **Tip :** To test the multi-location variant set `use_multi_location = True` in `federated_heatmaps/main.py` or extend the script with a proper CLI.

Sample console output

```
Running Adaptive Hierarchical Algorithm (Algorithm 2)…
Iteration 1 | remaining ε = 1.0000 | σ̃ target = 0.123
 …
Final vector length  : 340
Total communication  : 340 ints (≈680 bytes with m = 2^16)
MSE (reporting cells) : 7.9e-13
MSE (256×256 grid)   : 1.2e-8
```

---

## 📚 Module Overview

| Module                                                                | Highlights                                                      |
| --------------------------------------------------------------------- | --------------------------------------------------------------- |
| **`federated_heatmaps/config`**                                       | Typed dataclass hierarchy + YAML IO                             |
| **`federated_heatmaps/quad_tree`**                                    | `TreeNode`, `PrefixTree`; split/collapse, location→cell mapping |
| **`federated_heatmaps/differential_privacy/noise.py`**                | Vectorised discrete-Laplace (Pólya) & modulo clip               |
| **`federated_heatmaps/differential_privacy/differential_privacy.py`** | Helper maths (Eqs. 1–4)                                         |
| **`federated_heatmaps/algorithms/algorithm1.py`**                     | Histogram with noise + SecAgg (client & server paths)           |
| **`federated_heatmaps/algorithms/algorithm2.py`**                     | Adaptive tree refinement, budget scheduler                      |
| **`federated_heatmaps/utils/utils.py`**                               | Data simulation, truth/DP reconstruction, MSE/L₁                |
| **`federated_heatmaps/main.py`**                                      | Glue script: simulate → run → evaluate → plot                   |

---

## 🎯 Important Parameters

A concise reference; see the YAML for full details.

| Path                              | Meaning                             | Default   |
| --------------------------------- | ----------------------------------- | --------- |
| `privacy.eps_total`               | Total privacy budget (ε)            | **1.0**   |
| `privacy.delta_drop`              | Expected client dropout per shard   | 0.05      |
| `secagg.S_max`                    | Max clients per shard (SecAgg)      | 10 000    |
| `secagg.m`                        | Modular base for integer arithmetic | 65 536    |
| `algorithm.U_alg1`                | Users sampled per Alg 1 query       | 10 000    |
| `algorithm.c_alg2`                | Calibration constant *c* (Eq. 2)    | 0.1       |
| `algorithm.b_alg2`                | Expansion factor *b* (Eq. 4)        | 2.0       |
| `tree.max_depth`                  | Quadtree depth                      | 10        |
| `tree.grid_width` / `grid_height` | Fine-grid size                      | 256 / 256 |
| `multi_loc.gamma_scaling`         | Weight precision for multi-loc      | 100.0     |
| `verbose`                         | Extra logging                       | False     |

---