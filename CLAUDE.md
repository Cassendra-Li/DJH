# CLAUDE.md — RPGW-Net 项目

## 项目概述

RPGW-Net（Robust Partial Gromov-Wasserstein Network）：利用 **Gromov-Wasserstein 结构对齐** 在**噪声 + 小样本**条件下实现**跨工况轴承故障诊断**。

### 研究动机三句话

1. 现有跨工况故障诊断 DA 方法（MMD / 域对抗）只对齐特征分布，不显式对齐图结构
2. DAGCN (IEEE TIM 2021) 首次引入图结构但只用 MMD 对齐；GW 能显式比较不同度量空间的结构差异
3. 噪声 + 小样本场景下，结构信息比分布信息更稳健 → Partial GW 可裁剪噪声"质量"，只保留可信部分对齐

### 三篇核心参考资料

| 论文 | 角色 | 代码仓库 |
|------|------|---------|
| **DAGCN** (Li et al., IEEE TIM 2021) | 最接近的 baseline：图结构 + 跨域，但用的是 MMD 不是 GW | `baselines/dagcn/` |
| **GW-OT** (Seyedi et al.) | GW 工具箱：MultiInit、EGW/FGW 调参、可扩展性分析 | `baselines/gw_cqap/` |
| **MADGA** (arXiv:2410.08877, 2024) | GW 用于工业传感器异常检测的唯一现有工作（非故障诊断） | — |

### 当前进度（2025-06-28）

- [x] 项目框架搭建完成，代码提交 Git
- [x] demo_quick_test.py 5项全通过
- [x] PyTorch 2.12 + CUDA 12.6 + RTX 4050 环境就绪
- [x] POT / PyWavelets 已安装
- [ ] CWRU 数据集下载
- [ ] 基础实验（clean, full-shot, 0→3hp）跑通
- [ ] 加噪声实验
- [ ] 小样本实验
- [ ] GW 变体对比实验
- [ ] DAGCN baseline 对比

---

## 项目结构

```
DJH/
├── README.md               ← 项目主页（英），含 Pipeline 图、Quick Start
├── CLAUDE.md               ← 本文件
├── requirements.txt         ← pip install -r requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   └── default.yaml         ← ⭐ 所有超参数在这里统一改
│
├── data/
│   ├── download_cwru.py     ← 自动下载 CWRU .mat 到 data/raw/CWRU/
│   ├── preprocess.py        ← CWT + 自适应去噪 + kNN图构建
│   └── dataset.py           ← PyTorch Dataset（支持噪声、few-shot）
│
├── rpgw/                    ← ⭐ 核心模块
│   ├── models/
│   │   ├── cnn_encoder.py   ← 1D-CNN（4层），对标 DAGCN
│   │   ├── graph_builder.py ← 图构建包装器
│   │   ├── gat_encoder.py   ← GAT（2层，4头）纯 PyTorch 实现
│   │   ├── gw_alignment.py  ← 🔥 GW/EGW/FGW/Partial GW + MultiInit
│   │   └── prototype.py     ← 加权原型小样本分类器
│   ├── losses/
│   │   └── gw_loss.py       ← GWloss + MMD + CombinedLoss
│   └── train.py             ← RPGWNet 完整模型 + 训练循环
│
├── baselines/               ← 参考论文（只存 README，不存代码）
│   ├── dagcn/README.md
│   └── gw_cqap/README.md
│
├── experiments/
│   └── run_rpgw.py          ← 命令行入口：--source --target --shots --snr --gw_type
│
├── scripts/
│   ├── demo_quick_test.py   ← ✅ 5分钟验证，已全部通过
│   └── visualize.py         ← t-SNE + GW 热力图
│
└── tests/
    └── test_gw_alignment.py ← GW 模块单元测试
```

---

## 技术细节

### 数据流

```
原始振动信号 (1024,) → CWT → 时频图 (64×1024)
→ 切patch (8×8) → 节点特征 (1024, 64)
→ kNN(k=8) → 邻接矩阵 + 距离矩阵
→ GAT编码 → H (1024, 256) + C (1024, 1024)
→ 全局mean pooling → 图级embedding (256,)
→ Partial GW对齐（源↔目标）
→ 加权原型分类 → 预测故障类型 (10类)
```

### CWRU 数据集

- 10 类故障：Normal + IR007/014/021 + OR007/014/021 + B007/014/021
- 4 种工况（电机负载）：0, 1, 2, 3 hp
- 12 个跨域任务：0→1, 0→2, 0→3, 1→0, 1→2, ...
- 采样率 12kHz（驱动端），每个信号 ~120K 采样点 → 滑窗 1024 → 约 100 个窗口/类/工况

### GW 变体选用指南

| 场景 | 推荐 GW 变体 | 理由 |
|------|-------------|------|
| 无噪声 + full-shot | vanilla GW + MultiInit | 最简单，MultiInit 保证不陷入局部最优 |
| 有噪声 + full-shot | Partial GW (mass=0.85) | 自动裁剪噪声节点 |
| 端到端训练（需可微） | EGW (ε=0.8) | Sinkhorn 迭代可微 |
| 同时用特征+结构 | FGW (α=0.7) | GAT 输出同时有 H 和 C |
| 最强鲁棒性 | Partial FGW | 缺点：复杂度高 |

### 超参速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| cwt_scales | 64 | CWT 频率尺度数 |
| patch_size | 8 | 时频图切块大小 |
| k_neighbors | 8 | kNN 图邻居数 |
| GAT heads | 4 | 多头注意力数 |
| GW multi_init | 5 | GW 随机初始化次数 |
| GW epsilon | 0.8 | EGW 熵正则系数 |
| GW alpha | 0.7 | FGW 结构权重 |
| GW partial_mass | 0.85 | Partial GW 传输质量比 |
| batch_size | 64 | 批大小 |
| lr | 0.001 | 学习率 |
| epochs | 300 | 总训练轮数 |
| pretrain_epochs | 50 | 前50轮不激活 GW 对齐 |

---

## 常用命令

```bash
# 验证环境
python scripts/demo_quick_test.py

# 下载 CWRU
python data/download_cwru.py

# 基础实验
python experiments/run_rpgw.py --source 0 --target 3

# 带噪声 + 小样本（核心场景）
python experiments/run_rpgw.py --source 0 --target 3 --shots 5 --snr 5

# 换 GW 变体
python experiments/run_rpgw.py --source 0 --target 3 --gw_type entropic
python experiments/run_rpgw.py --source 0 --target 3 --gw_type fused

# 消融：去掉 GW 只用 MMD
python experiments/run_rpgw.py --source 0 --target 3 --ablation mmd_only

# Git
git status         # 看改了哪些文件
git add -A         # 暂存所有修改
git commit -m "..." # 提交并写描述
```

---

## 对话续接规则

当新对话开始时，请执行以下步骤延续工作：

1. **读取本文件** — 了解项目全貌和当前进度
2. **检查环境** — `python -c "import torch; print(torch.cuda.is_available())"`
3. **检查 Git 状态** — `git status` 看上次做到哪了
4. **跑 demo 验证** — `python scripts/demo_quick_test.py`（如果环境变了）
5. **查看 config** — `configs/default.yaml` 当前参数是什么
6. **根据用户指令继续工作** — 从断点处接着往下做

### 关键注意事项

- 所有新对话时必须先读完 CLAUDE.md 再行动，不要从零推理项目
- 修改代码后跑 `python scripts/demo_quick_test.py` 确认不破坏已有功能
- 每次有意义改动立即 `git commit`
- 实验配置统一通过 `configs/default.yaml` 修改，不要硬编码进 Python
- GW 对齐模块 (`gw_alignment.py`) 是论文的核心 contribution，修改时要格外小心
- 数据文件 (.mat, .npy) 不提交到 Git（已在 .gitignore 中）
