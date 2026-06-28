# RPGW-Net: Robust Partial Gromov-Wasserstein Network

**Cross-domain Bearing Fault Diagnosis under Noisy & Few-shot Conditions via Graph Structure Alignment**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

Transferring fault diagnosis models across variable working conditions is challenging, especially under **measurement noise** and **limited labeled samples**. This work proposes **RPGW-Net**, which leverages **Gromov-Wasserstein (GW)** optimal transport to explicitly align the **graph structure** of time-frequency representations between source and target domains.

### Key Idea

Traditional domain adaptation methods (MMD, adversarial training) align **feature distributions**. We argue that **structural alignment** — preserving the internal relational patterns of time-frequency graphs — is more robust under noise and few-shot settings. GW distance naturally compares metric-measure spaces without requiring a shared coordinate system.

### Pipeline

```
Raw Vibration → CWT + Denoising → Graph Construction → GAT Encoder
                                                              ↓
                                            Fault Prediction ← Weighted Prototype
                                                              ↑
                                        Source Graph ═══ Target Graph
                                              GW Structure Alignment
```

---

## 📁 Project Structure

```
DJH/
├── README.md                      # You are here
├── requirements.txt               # Python dependencies
├── .gitignore
│
├── configs/
│   └── default.yaml               # Training hyperparameters
│
├── data/
│   ├── download_cwru.py           # CWRU dataset downloader
│   ├── preprocess.py              # CWT + denoising + graph construction
│   └── dataset.py                 # PyTorch Dataset loader
│
├── rpgw/                          # RPGW-Net core
│   ├── models/
│   │   ├── cnn_encoder.py         # 1D-CNN time-frequency feature extractor
│   │   ├── graph_builder.py       # Patch → kNN graph construction
│   │   ├── gat_encoder.py         # Graph Attention Network encoder
│   │   ├── gw_alignment.py        # GW / EGW / FGW / Partial GW module
│   │   └── prototype.py           # Weighted Prototypical classifier
│   ├── losses/
│   │   └── gw_loss.py             # GW-based alignment loss functions
│   └── train.py                   # Main training loop
│
├── baselines/                     # Reference implementations (read-only)
│   ├── dagcn/                     # DAGCN (Li et al., IEEE TIM 2021)
│   └── gw_cqap/                   # GW_CQAP (Seyedi et al.)
│
├── experiments/                   # Experiment scripts
│   ├── run_baselines.py           # Run all baselines
│   ├── run_rpgw.py                # Run RPGW-Net
│   └── run_ablation.py            # Ablation studies
│
├── scripts/
│   ├── demo_quick_test.py         # 5-min quick verification
│   └── visualize.py               # t-SNE & GW distance visualization
│
└── tests/
    └── test_gw_alignment.py       # Unit test for GW module
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CWRU Dataset

```bash
python data/download_cwru.py
```

### 3. Run Quick Demo (5 min)

```bash
python scripts/demo_quick_test.py
```

This runs a minimal training on CWRU (0→3 hp, 5-shot, no noise) and verifies the pipeline works.

### 4. Full Training

```bash
# RPGW-Net with Partial GW
python experiments/run_rpgw.py --source 0 --target 3 --shots 5 --noise_snr 5

# Compare with DAGCN-style MMD baseline
python experiments/run_baselines.py --method dagcn --source 0 --target 3
```

---

## 📊 Expected Results (CWRU Dataset)

| Method | Clean (Acc%) | SNR=5dB | SNR=0dB | SNR=-5dB | 5-shot |
|--------|:-----------:|:-------:|:-------:|:--------:|:------:|
| CNN (no adaptation) | ~75 | ~55 | ~40 | ~28 | ~45 |
| DAGCN (GCN + MMD) | ~92 | ~78 | ~62 | ~48 | ~65 |
| WD-DTL (W-distance) | ~93 | ~81 | ~68 | ~52 | ~68 |
| **RPGW-Net (Ours)** | **~96** | **~88** | **~78** | **~65** | **~82** |

> *Numbers are targets; actual results depend on hyperparameter tuning.*

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{rpgw-net,
  title={RPGW-Net: Robust Partial Gromov-Wasserstein Network for 
         Cross-domain Bearing Fault Diagnosis},
  author={},
  journal={},
  year={}
}
```

## 🙏 Acknowledgments

- DAGCN: [Li et al., IEEE TIM 2021](https://github.com/HazeDT/DAGCN)
- GW for CQAP: [Seyedi et al.](https://github.com/iman-ie/GW_CQAP)
- POT library: [Python Optimal Transport](https://pythonot.github.io/)

## 📄 License

MIT
