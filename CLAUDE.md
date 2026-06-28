# CLAUDE.md — RPGW-Net 项目 (v2)

## 项目概述

RPGW-Net（Robust Partial Gromov-Wasserstein Network）：利用 **Gromov-Wasserstein 结构对齐** 在**噪声 + 小样本**条件下实现**跨工况轴承故障诊断**。

目标：在 CWRU 0→3hp 跨域任务上 **超过 DAGCN 的 92% 准确率**。

### 当前进度（2026-06-28）

- [x] v2 架构升级完成：1D-CNN + CWT 融合 → GAT → GW → Prototype
- [x] GW shape bug 修复（`C_s[0]` → 完整 batch）
- [x] DAGCN 训练策略引入：预训练 + sigmoid GW 权重升温
- [x] Git 提交推送（287527b, 06ae066）
- [ ] Stage 1 正式实验（CNN+GAT 基线, no GW, 150 epochs）
- [ ] Stage 2 正式实验（CNN+GAT+Partial GW, 150 epochs）
- [ ] 对标 DAGCN 92%，迭代到超过去

---

## v2 架构（2025-06-28）

```
原始振动 (1024,) ──┬── 1D-CNN (4层Conv1d) ──→ 256维全局特征 ──┐
                   │                                              │
                   └── CWT (64尺度) → Denoise → Patch(16×16)     │
                        → 256节点,每节点256维 → kNN(k=8)         │
                        → adj(256,256), cost(256,256)             │
                                                                  │
                   ┌──────────────────────────────────────────────┘
                   │  每节点 = [cnn_global(256) || cwt_patch(256)] = 512维
                   │  Linear(512→256) → GAT输入
                   │
                   └── GAT (2层, 4头) ──┬── H_global (B,256) → 原型分类 → logits
                                        │
                                        └── C (B,256,256) → GW结构对齐 → gw_loss
```

**与 framework.docx 对应：** `Raw → CWT → Denoise → Patch → [CNN融合] → Graph → GAT → GW → Prototype`

### v2 vs v1 关键变化

| 维度 | v1（旧） | v2（当前） |
|------|---------|-----------|
| 特征提取 | 无，CWT原始像素直接给GAT | 1D-CNN + CWT 融合 |
| GW 调用 | `C_s[0]` 只第1样本（bug） | 完整 `(B,N,N)` 批量 |
| 训练策略 | 全程 GW，固定权重 | pretrain 50ep + sigmoid 升温 |
| 入口脚本 | `experiments/run_rpgw.py` | `experiments/run.py` |
| 参数量 | 265K | 562K（CNN 164K + Fusion 132K + GAT 265K） |

---

## 项目结构

```
DJH/
├── README.md
├── CLAUDE.md               ← 本文件（每次新对话先读这个）
├── requirements.txt
├── .gitignore / LICENSE
│
├── configs/
│   └── default.yaml         ← ⭐ 所有超参数统一在这里
│
├── data/
│   ├── download_cwru.py     ← 下载 CWRU .mat → data/raw/CWRU/
│   ├── preprocess.py        ← CWT + 自适应去噪 + kNN图构建
│   ├── dataset.py           ← PyTorch Dataset（在线CWT+图构建，支持噪声/few-shot）
│   ├── cached_dataset.py    ← ⭐ 缓存Dataset（预计算.npy, 快1000倍）+ 返回raw_signal
│   └── precompute.py        ← 一次性预处理：raw .mat → CWT → 图 → .npy
│
├── rpgw/                    ← ⭐ 核心模块
│   ├── models/
│   │   ├── cnn_encoder.py   ← 1D-CNN (4层Conv1d), DAGCN同款 (→256维)
│   │   ├── gat_encoder.py   ← GAT (2层, 4头), 纯PyTorch实现
│   │   ├── gw_alignment.py  ← 🔥 GW/EGW/FGW/Partial GW + MultiInit
│   │   ├── prototype.py     ← 加权原型小样本分类器
│   │   └── graph_builder.py ← 图构建包装器（保留备用）
│   ├── losses/
│   │   └── gw_loss.py       ← MMD loss + CombinedLoss
│   └── train.py             ← ⭐ RPGWNet 完整模型类
│
├── experiments/
│   └── run.py               ← ⭐ 唯一入口：--stage 1~6 或自定义参数
│
├── scripts/
│   └── demo_quick_test.py   ← 快速验证（需更新适配v2）
│
├── baselines/
│   ├── dagcn/README.md
│   └── gw_cqap/README.md
│
├── results/                 ← ⭐ 所有实验结果保存这里
│   └── old_architecture.txt ← v1旧架构实验结果(最高43%)
│
├── logs/                    ← 训练日志（每次实验生成一个.log）
└── checkpoints/             ← 模型权重（每次实验生成一个.pth）
```

---

## 核心超参（configs/default.yaml 当前值）

| 参数 | 值 | 说明 |
|------|-----|------|
| `preprocess.cwt_scales` | 64 | CWT 频率尺度数 |
| `preprocess.patch_size` | 16 | 时频图patch大小 → 256节点 |
| `preprocess.k_neighbors` | 8 | kNN 图邻居数 |
| `model.cnn.feature_dim` | 256 | 1D-CNN 输出维度 |
| `model.gat.in_dim` | 256 | CNN+CWT 融合投影后维度 |
| `model.gat.hidden_dim` | 128 | GAT 隐藏层 |
| `model.gat.out_dim` | 256 | GAT 输出/原型分类输入 |
| `model.gat.heads` | 4 | 多头注意力数 |
| `model.gw.type` | partial | GW 变体 |
| `model.gw.multi_init` | 5 | 随机初始化次数 |
| `model.gw.partial_mass` | 0.85 | 传输质量比 |
| `train.batch_size` | 8 | 批大小（受6GB显存限制） |
| `train.epochs` | 200 | 50 pretrain + 150 with GW |
| `train.pretrain_epochs` | 50 | 前50轮 GW=0 |
| `train.gw_warmup_epochs` | 20 | sigmoid 升温轮数 |
| `train.loss.gw_weight` | 0.3 | GW loss 最终权重 |

---

## 常用命令

```bash
# 预处理数据（首次或参数变更后）
python data/precompute.py

# Stage 1: CNN+GAT 基线 (no GW, clean, full-shot, 0→3hp)
python experiments/run.py --stage 1

# Stage 2: CNN+GAT + Partial GW
python experiments/run.py --stage 2

# 快速验证（5 epoch）
python experiments/run.py --stage 1 --epochs 5 --batch_size 4 --name quick_test

# 自定义实验
python experiments/run.py --source 0 --target 3 --gw partial --epochs 150

# 噪声 + 小样本（核心场景）
python experiments/run.py --stage 3 --snr 5 --shots 5

# 换 GW 变体
python experiments/run.py --stage 4   # vanilla GW
python experiments/run.py --stage 5   # entropic GW
python experiments/run.py --stage 6   # fused GW

# GitHub
git status
git add -A && git commit -m "描述"
git push origin main
```

## Stage 预设速查

| Stage | GW 类型 | 噪声 | 样本 | epochs | 用途 |
|:-----:|---------|:---:|:---:|:-----:|------|
| 1 | none | clean | full | 150 | CNN+GAT 基线 |
| 2 | partial | clean | full | 150 | GW 增益验证 |
| 3 | partial | 5dB | 5-shot | 200 | 核心场景 |
| 4 | vanilla | clean | full | 150 | GW 变体消融 |
| 5 | entropic | clean | full | 150 | GW 变体消融 |
| 6 | fused | clean | full | 150 | GW 变体消融 |

---

## GW 变体选用指南

| 场景 | 推荐 | 理由 |
|------|------|------|
| 无噪声 + full-shot | vanilla GW + MultiInit | 最简，MultiInit 防局部最优 |
| 有噪声 | **Partial GW** (mass=0.85) | 自动裁剪噪声节点 |
| 需可微（端到端） | EGW (ε=0.8) | Sinkhorn 迭代可微 |
| 同时用特征+结构 | FGW (α=0.7) | GAT 同时输出 H 和 C |

---

## 对话续接规则

每次新对话开始时，按顺序执行：

1. **读本文件** — 了解项目全貌和当前进度
2. **检查 Git** — `git log --oneline -5` 看最近提交
3. **检查环境** — `python -c "import torch; print(torch.cuda.is_available())"`
4. **看 config** — `configs/default.yaml` 确认参数
5. **看实验结果** — `results/` 目录下已有的结果
6. **根据用户指令继续** — 从断点接着做

### 关键注意事项

- ⚠️ 任何改动前先告诉用户，经同意后再改
- 📝 每次有意义改动立即 `git commit`
- 📊 实验结果必须保存到 `results/` 目录
- ⚙️ 超参统一在 `configs/default.yaml` 改，不要硬编码
- 🔥 `gw_alignment.py` 是核心 contribution，修改要格外小心
- 🚫 数据文件 (.mat, .npy) 不提交 Git（已在 .gitignore）
- 💾 显存限制 6GB → batch_size 不超过 8
