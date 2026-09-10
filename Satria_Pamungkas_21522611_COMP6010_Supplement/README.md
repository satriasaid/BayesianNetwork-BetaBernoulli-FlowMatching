# COMP6010 Assignment — Supplement

**Student:** Satria Pamungkas · **ID:** 21522611
**Part C option:** **C2** — animal-drawing generation with flow matching (trained from scratch)

This ZIP is the supporting evidence for the report
`Satria_Pamungkas_21522611_COMP6010_Report.pdf`. Every number, table and figure in the report is
traceable to a file in `results/`.

---

## 1. Folder structure

```
Satria_Pamungkas_21522611_COMP6010_Supplement/
├── README.md                     <- this file
├── AI_USE_LOG.md                 <- AI-use / development log (5 interactions + verification table)
├── requirements.txt              <- minimum package versions
├── requirements_frozen.txt       <- exact versions of the environment used
├── download_quickdraw.sh         <- acquires the Part C dataset (NOT shipped: ~320 MB)
│
├── notebooks/
│   ├── 01_PartA_Bayesian_Network.ipynb    <- Part A, all outputs preserved
│   ├── 02_PartB_Beta_Bernoulli.ipynb      <- Part B, all outputs preserved
│   └── 03_PartC_FlowMatching.ipynb        <- Part C (C2), all outputs preserved
│
├── src/                          <- helper modules imported by notebook 03
│   ├── data_quickdraw.py         <- subset selection, disclosed filter, split-before-augment
│   ├── flow_model.py             <- VelocityUNet, CFM target construction, ODE samplers
│   ├── train_flow.py             <- training loop, held-out loss, checkpointing
│   └── evaluate.py               <- independent classifier, recognisability, diversity, NN probe
│
├── checkpoints/                  <- small trained weights (included; ~7 MB each)
│   ├── baseline/ckpt_epoch{005,015,060}.pt
│   ├── small_data/ckpt_epoch060.pt
│   ├── eval_classifier.pt
│   └── (each ckpt stores model state, config, losses, categories)
│
├── data/quickdraw/               <- EMPTY in the ZIP; run download_quickdraw.sh
│
├── manifest/                     <- copies of the data-split and result manifests
│
└── results/
    ├── partA/   <- 14 files: exact values, per-run CSVs, summary, convergence plot
    ├── partB/   <- posterior tables, decision JSON, prior-sensitivity, MC verification
    └── partC/   <- training curves, comparison CSVs, matched grids, memorisation probe
        └── sample_grids/   <- COMPLETE output grid for every configuration
```

## 2. Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # or requirements_frozen.txt for exact pins
```

Parts A and B need only NumPy / SciPy / pandas / matplotlib and run on CPU.
Part C needs PyTorch; a CUDA GPU is strongly recommended (see runtimes below).

## 3. Environment actually used

| | |
|---|---|
| OS | Linux 6.8.0-138-generic (x86_64) |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 3060 (12 GB), CUDA build `cu128` |
| torch | 2.10.0+cu128 |
| numpy / scipy / pandas / matplotlib | 2.2.6 / 1.15.3 / 2.3.3 / 3.10.8 |

Exact versions: `requirements_frozen.txt`. Environment details are also embedded in
`results/partA/A_manifest.json`, `results/partB/B_manifest.json` and
`results/partC/C_reproduction_record.json`.

## 4. Data and model sources

| Item | Source | Licence / terms | Retrieved |
|---|---|---|---|
| Quick, Draw! `cat.npy`, `dog.npy`, `fish.npy` (`numpy_bitmap`, 28×28 uint8) | `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy` — repository: <https://github.com/googlecreativelab/quickdraw-dataset> | **CC BY 4.0**, Google Creative Lab. Attribution: *"Quick, Draw! Dataset" by Google Creative Lab, CC BY 4.0.* | 10 Sep 2026 |
| Pretrained models | **none used** — Part C trains entirely from scratch. No base checkpoint, nothing frozen. | — | — |

SHA-256 checksums of the three downloaded files are recorded in
`results/partC/C_data_manifest.json` (`categories[].sha256`); verify with `sha256sum data/quickdraw/*.npy`.

The dataset files are **not** included in the ZIP (~320 MB total), per the instruction to provide
acquisition instructions rather than shipping large public datasets. Run:

```bash
./download_quickdraw.sh
```

## 5. Seeds

| Purpose | Seed | Where |
|---|---|---|
| Part A sampling runs (5 independent seeds per sample size) | `11, 22, 33, 44, 55` | nb 01 |
| Part B Monte-Carlo posterior draws (200,000 draws) | `20260910` | nb 02 |
| Part C data subset selection and train/eval split | `20260910` | `src/data_quickdraw.py` |
| Part C training (torch, numpy, augmentation, label handling) | `20260910` | `src/train_flow.py` |
| Part C **generation seed list**, reused across every comparison | `101, 202, 303, 404, 505, 606, 707, 808` | nb 03 |
| Part C evaluation-classifier training | `777` | `src/evaluate.py` |

Generation is deterministic given a seed: the ODE solvers are deterministic maps, and all randomness
enters through the initial noise $x_0$. The same seed list therefore reproduces the same images.

**Bitwise reproducibility requires deterministic cuDNN.** `flow_model.enable_deterministic()` is
called at the top of notebook 03. Without it, cuDNN may select non-deterministic convolution
algorithms and TF32 reductions, so two runs of the same seed differ by roughly 7e-4 — visually
irrelevant, but not bitwise equal. With it, same-seed runs are identical and the notebook asserts
this (`check_determinism`). Results may still differ slightly across different GPU models or CUDA
versions; the checkpoints are provided so the reported figures can be reproduced exactly on this
hardware.

## 6. Run order and expected runtimes

Run the notebooks in numerical order. Each is self-contained given the data.

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_PartA_Bayesian_Network.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_PartB_Beta_Bernoulli.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_PartC_FlowMatching.ipynb
```

| Notebook | Runtime (this machine) | Notes |
|---|---|---|
| 01 Part A | ~1 min (CPU) | 45 sampling runs + exact enumeration |
| 02 Part B | ~1 min (CPU) | conjugate updates + 200k MC draws |
| 03 Part C — **training route** | **~15–20 min (GPU)** | two 60-epoch runs + classifier + 14 sampling configurations |
| 03 Part C — **inference-only route** | ~2 min (GPU) | see §7 |

Part C on CPU only would take several hours; use a GPU if at all possible.

## 7. Separating training from inference/reproduction

Notebook 03 **skips training automatically** if a training log already exists: the cells check for
`checkpoints/baseline/training_log.json` and `checkpoints/small_data/training_log.json` and load them
instead of retraining. Because the checkpoints are included in this ZIP, running notebook 03
as-shipped takes the **inference-only route** and reproduces every figure and table from the saved
weights.

To force a full retrain from scratch, delete the checkpoint directories first:

```bash
rm -rf checkpoints/baseline checkpoints/small_data checkpoints/eval_classifier.pt
```

Loading a checkpoint directly:

```python
import sys; sys.path.insert(0, "src")
import train_flow as tfl
from flow_model import sample_ode
import torch

model, ck = tfl.load_checkpoint("checkpoints/baseline/ckpt_epoch060.pt", device="cuda")
print(ck["categories"], ck["epoch"], ck["eval_loss"])          # ['cat','dog','fish']
labels = torch.tensor([0, 1, 2], device="cuda")               # cat, dog, fish
images, nfe = sample_ode(model, 3, labels, n_steps=50, method="euler",
                         device="cuda", seed=101)             # -> (3,1,28,28) in ~[-1,1]
```

## 8. Manifest of data splits and result files

* **Data split manifest** — `results/partC/C_data_manifest.json`: source URL pattern, licence,
  per-category SHA-256, drawings available in each file, candidates drawn, drawings rejected by the
  disclosed ink filter, original drawings used, and the train/eval counts plus SHA-256 hashes of the
  selected index arrays (so the exact split is verifiable without shipping the data).
* **Split summary table** — `results/partC/C_split_summary.csv`.
* **Full reproduction record** — `results/partC/C_reproduction_record.json`: research questions,
  model and parameter counts, baseline config, training times, every comparison's results, the
  evaluation protocol and the environment.
* **Part A / Part B manifests** — `results/partA/A_manifest.json`, `results/partB/B_manifest.json`:
  model constants, exact reference values, seeds and environment.

### Key result files by report claim

| Report claim | File |
|---|---|
| Part A exact query values | `results/partA/A1_exact_reference_values.csv` |
| Part A per-run rejection / LW results | `results/partA/A2_rejection_per_run.csv`, `A3_likelihood_weighting_per_run.csv` |
| Part A summary (mean, sd, MAE) | `results/partA/A4_summary_table.csv` |
| Part A convergence plot for $q_2$ | `results/partA/A4_convergence_q2.png` |
| Part B posterior sequence and point estimates | `results/partB/B2_sequential_updates.csv`, `B2_point_estimates.csv` |
| Part B decision (P(θ>0.8), CI, predictive) | `results/partB/B3_decision.json` |
| Part B prior sensitivity | `results/partB/B4_prior_sensitivity.csv` |
| Part C training curve | `results/partC/C_training_curves.png`, `C_training_epochs.csv` |
| Part C comparison 1 (checkpoints) | `results/partC/C_comparison1_checkpoints.csv` |
| Part C comparison 1b (dataset size) | `results/partC/C_comparison1b_dataset_size.csv` |
| Part C comparison 2 (steps, solver) | `results/partC/C_comparison2_sampler.csv`, `C_comparison2_matched_nfe.csv` |
| Part C all configurations | `results/partC/C_all_results.csv` |
| Part C complete output grids | `results/partC/sample_grids/` |
| Part C memorisation probe | `results/partC/C_memorisation_distances.csv`, `C_memorisation_probe.png` |

## 9. Notes

* **No credentials or access tokens** are present anywhere in this ZIP. Nothing in the pipeline
  requires authentication — the dataset is public and no pretrained model is used.
* **Evaluation protocol.** Every quantitative measure is applied to the *complete* output set of each
  configuration (24 images: 8 predetermined seeds × 3 classes), never to selected successes. The full
  grids in `results/partC/sample_grids/` include failures.
* **Part C evaluation classifier** is trained only on **held-out** real drawings, so the evaluator
  never saw the generative model's training data.
* **AI use** is documented in `AI_USE_LOG.md` with five interactions, my verification of each, and a
  table of the 14 independent correctness checks I ran.
