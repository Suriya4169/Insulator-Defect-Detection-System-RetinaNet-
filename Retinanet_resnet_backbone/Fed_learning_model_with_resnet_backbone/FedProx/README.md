# Phase 1 — FedProx Algorithmic Upgrade

This folder contains the **complete, self-contained implementation of Phase 1** of the Journal Upgrade Plan: transitioning from plain FedAvg to a dual-algorithm system that supports both **FedAvg** and **FedProx**.

---

## 📁 File Structure

```
Phase1_FedProx/
├── engine_fedprox.py           ← FedProx-aware training engine
├── fedprox_train.py            ← Core FL training loop (FedAvg + FedProx)
├── compare_fedavg_vs_fedprox.py← Comparison experiment runner (paper Table II)
└── README.md                   ← This file
```

> All shared code (`dataset.py`, `model.py`, `transforms.py`, `engine.py`)
> is **imported from the original** `Fed_learning_Code/` folder — no duplication.

---

## 🧠 What is FedProx?

FedProx (Li et al., 2020) extends FedAvg by adding a **proximal regularisation term** to each client's local loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \underbrace{\frac{\mu}{2} \|w_{\text{local}} - w^t_{\text{global}}\|^2}_{\text{proximal term}}
$$

| `mu` | Behaviour |
|:----:|:----------|
| `0`  | Identical to FedAvg ✅ |
| `0.01` | Mild regularisation (recommended starting point) |
| `0.1`  | Stronger pull toward global — use for extreme Non-IID |

**Why do we need it?**
Under extreme Non-IID data (Dirichlet α=0.1), different clients have wildly different class distributions. Without the proximal term, local models drift far from the global model, hurting convergence. FedProx penalises this drift, stabilising training.

---

## ⚙️ Step 1: Update the Data Path

Open `fedprox_train.py` and update `FedProxConfig.DATA_PATH`:

```python
class FedProxConfig:
    DATA_PATH  = r'D:\Fed learning project\Dataset - IDD-CPLID.v3-cplid_new.coco'
    OUTPUT_DIR = r'D:\Fed learning project\checkpoints\phase1_fedprox'
```

---

## 🚀 Step 2: Run the Experiments

### Option A — Run everything automatically (recommended for journal)

This runs all 3 configurations and generates all plots and the LaTeX table:

```bash
cd "D:\Fed learning project\Insulator-Defect-Detection-System-RetinaNet-\Phase1_FedProx"
python compare_fedavg_vs_fedprox.py
```

### Option B — Run a single experiment manually

```bash
# Reproduce original FedAvg baseline (mu=0 → identical to original training)
python fedprox_train.py --algorithm fedavg --mu 0 --partition iid

# FedAvg on extreme Non-IID (worst case)
python fedprox_train.py --algorithm fedavg --mu 0 --partition non_iid --alpha 0.1

# FedProx on extreme Non-IID (proposed fix)
python fedprox_train.py --algorithm fedprox --mu 0.01 --partition non_iid --alpha 0.1
```

### Option C — Re-plot without re-training (if checkpoints exist)

```bash
python compare_fedavg_vs_fedprox.py --plots-only
```

---

## 📊 Expected Outputs

After running `compare_fedavg_vs_fedprox.py`, you will find in
`checkpoints/phase1_fedprox/comparison_plots/`:

| File | Description |
|------|-------------|
| `fedavg_vs_fedprox_map50.png` | mAP@50 per round — line plot (Figure for paper) |
| `fedavg_vs_fedprox_bar.png`   | Final mAP@50 bar chart |
| `summary_table.txt`           | Plain-text table for proofreading |
| `summary_latex.tex`           | **Copy-paste ready LaTeX table** for the manuscript |

---

## 📝 How to Use in the Paper

### Results Section

Add a sub-section titled:
> **"4.3  Handling Non-IID Data Heterogeneity: FedProx vs. FedAvg"**

Insert the line plot as Figure X and the bar chart as Figure X+1.

Paste the contents of `summary_latex.tex` as **Table II**.

### Suggested narrative

```
To evaluate the robustness of the proposed framework under extreme statistical
heterogeneity, we applied Dirichlet partitioning with α=0.1 — a pathological
Non-IID setting in which one client may receive >80% of all defect samples.
Table II compares FedAvg and FedProx under this condition alongside the IID
baseline.  FedProx recovers X.X% mAP@50 relative to the FedAvg Non-IID run,
closing Y% of the gap introduced by heterogeneity, while introducing zero
additional communication overhead.
```

*(Fill in X.X and Y from `summary_table.txt` after running the experiments.)*

---

## 📚 References

- Li, T., et al. "Federated Optimization in Heterogeneous Networks." *MLSys*, 2020.
- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from
  Decentralized Data." *AISTATS*, 2017.
