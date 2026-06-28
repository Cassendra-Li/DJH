# DAGCN Baseline Reference

**Source**: Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions
**Authors**: Tianfu Li, Zhibin Zhao, Chuang Sun, Ruqiang Yan, Xuefeng Chen
**Venue**: IEEE Transactions on Instrumentation and Measurement, 2021
**Code**: https://github.com/HazeDT/DAGCN

## Key Points

- First paper to integrate **graph structure modeling** into cross-domain fault diagnosis
- Uses CNN → Graph Generation Layer (GGL) → MRF-GCN → MMD alignment
- MMD aligns features in RKHS (NOT explicit structure alignment — this is where we improve)
- Ablation: graph structure contributes +18.74%, domain alignment +3.20%

## How We Build on This

DAGCN proved that **graph structure matters** for cross-domain diagnosis.
Our RPGW-Net replaces the MMD structural alignment with **Gromov-Wasserstein**,
which explicitly aligns the internal structure of graphs rather than just
feature distributions in RKHS.
