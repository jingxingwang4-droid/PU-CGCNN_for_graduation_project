# PU-CGCNN：基于部分监督学习的无机晶体可合成性预测

> 本科毕业设计项目 · 复现并改造 Jang et al. (2020) 提出的 PU-CGCNN 框架，用于预测虚拟晶体结构在实验上被成功合成的概率（Crystal-Likeness Score, CLscore）。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 研究背景](#2-研究背景)
- [3. 方法原理](#3-方法原理)
- [4. 仓库结构](#4-仓库结构)
- [5. 环境配置](#5-环境配置)
- [6. 数据准备](#6-数据准备)
- [7. 使用流程](#7-使用流程)
- [8. 模型架构](#8-模型架构)
- [9. 可视化与交互界面](#9-可视化与交互界面)
- [10. 已知问题与改进计划](#10-已知问题与改进计划)
- [11. 参考文献](#11-参考文献)
- [12. 致谢](#12-致谢)

---

## 1. 项目简介

本项目以 **KAIST 郑瑢成（Yousung Jung）课题组** 在 JACS 2020 发表的 PU-CGCNN 模型为基础，完成了以下工作：

1. 在 PyTorch 2.6 + CUDA 12.8（RTX 5070 Ti Laptop，Blackwell 架构）环境下完成了模型的重新训练；
2. 重写了预测与评估管线 `predict_target.py` / `evaluate_model.py`，解决了 Windows 下多进程构图失败的问题；
3. 基于 Streamlit 构建了可视化评估面板 `app.py`；
4. 基于 Tkinter 构建了桌面端一键预测 GUI `pu_cgcnn_gui.py`，面向非技术用户；
5. 增加了论文级模型架构可视化脚本（`visualize_academic_architecture.py` 等）。

模型输出是一个介于 0 到 1 之间的 **CLscore**，可视作"给定晶体结构可以被实验合成的概率"。当 CLscore > 0.5 时，我们认为该结构具备合成潜力。

---

## 2. 研究背景

发现新的无机功能材料长期以来依赖于 **计算筛选 → 人工合成** 的范式。在筛选阶段，传统判据是 **热力学稳定性**（例如 Ehull），但该判据已被多次证明不够充分：

- 许多稳定材料（低 Ehull）至今未能在实验上被成功合成；
- 许多高价值功能材料处于 **亚稳态**（Ehull > 0），热力学判据会直接将其漏掉。

这是因为 **可合成性（synthesizability）** 是一个由热力学、动力学、前驱体选择、实验条件等多因素共同决定的复杂现象，单一的热力学指标无法完整刻画。

PU-CGCNN 从 **数据驱动** 的视角切入，仅使用已合成材料（正样本）与未标注材料（未合成，但未必不能合成）来学习一个可合成性判别器，规避了"构建负样本数据集"这一在实验上不可能完成的任务。

---

## 3. 方法原理

### 3.1 CGCNN：晶体图卷积神经网络

CGCNN（Xie & Grossman, 2018）将一个晶体结构表示为一张图：

- **节点**：晶胞内的每个原子，节点特征由元素的化学属性（原子序数、电负性、原子半径等）的 one-hot / 数值拼接而成，存储于 `atom_init.json`；
- **边**：以每个原子为中心、半径 `r = 8 Å` 内的 `k = 12` 个最近邻原子连接而成；
- **边特征**：邻居距离经 Gaussian basis expansion 得到一个 41 维向量。

图卷积层 `ConvLayer` 采用 gated 聚合：

$$
v_i^{(t+1)} = v_i^{(t)} + \sum_{j,k} \sigma(z_{(i,j)_k}^{(t)} W_f + b_f) \odot g(z_{(i,j)_k}^{(t)} W_s + b_s)
$$

其中 $z_{(i,j)_k}^{(t)} = v_i^{(t)} \oplus v_j^{(t)} \oplus e_{(i,j)_k}$ 是中心原子、邻居原子与边特征的拼接。堆叠 `n_conv = 3` 层图卷积后做平均池化得到晶体级向量，再经过 `h_fea_len = 128` 的全连接层与 LogSoftmax 输出 2 维分类 logit。

### 3.2 PU Learning：从"正例 + 未标注"中学习

传统二分类需要明确的正负样本，但在材料数据库中只有 **"已合成"（P）** 和 **"未知"（U）** 两类。未合成材料中既包含真正不可合成的结构，也包含尚未有人尝试合成的有价值结构——因此 U 不能被简单当成负例。

PU-CGCNN 采用 **Bootstrap Aggregating (Bagging)** 策略：

1. 将 P 全部保留，从 U 中有放回地抽取子集 U<sub>i</sub> 当作"伪负例"；
2. 用 P + U<sub>i</sub> 训练一个 CGCNN 二分类器 $f_i$；
3. 重复 `bag = 50` 次，得到 50 个弱分类器；
4. 对任一待测样本 $x$，最终 CLscore 为 50 个模型输出的 **平均概率**：

$$
\text{CLscore}(x) = \frac{1}{50}\sum_{i=1}^{50} P_{f_i}(y=1 \mid x)
$$

这种做法的合理性在于：每个被错标成负例的真正可合成样本，只会出现在少数 bag 中，集成平均后这些误差会被抵消，而真正不可合成的结构将在大多数 bag 中被稳定判为负。

---

## 4. 仓库结构

```
PU-CGCNN_for_graduation_project/
├── main_PU_learning.py              # 训练主脚本：Bagging + CGCNN 训练循环
├── predict_PU_learning.py           # 底层预测脚本：加载 50 个 checkpoint 做批量推断
├── predict_target.py                # 端到端新材料打分管线（CIF → 图 → 预测 → 排名）
├── evaluate_model.py                # 集成评估：CLscore 融合、ROC-AUC、混淆矩阵、Top-N 挖掘
├── generate_crystal_graph.py        # 预处理：CIF → Gaussian 图 → pickle
├── data_PU_learning.py              # CIFData 数据集、collate_pool、split_bagging
├── model_PU_learning.py             # CrystalGraphConvNet 与 ConvLayer 实现
├── app.py                           # Streamlit 可视化评估面板（网页端）
├── pu_cgcnn_gui.py                  # Tkinter 桌面 GUI：一键端到端推断（推荐非技术用户使用）
├── show_model_arch.py               # 打印模型参数表
├── visualize_architecture.py        # 基础版架构图
├── visualize_academic_architecture.py # 论文级架构图（推荐）
├── visualize_max_architecture.py    # 扩展版架构图
├── debug_graph.py                   # 构图调试工具
├── move_test.py                     # 测试集划分工具
└── requirements.txt
```

> 注：正式运行时需要将 `data_PU_learning.py` 与 `model_PU_learning.py` 放入 `cgcnn/` 子目录，详见 [§10 已知问题](#10-已知问题与改进计划)。

---

## 5. 环境配置

### 5.1 硬件建议

| 组件 | 本项目训练配置 | 最低要求 |
|------|--------------|---------|
| GPU  | NVIDIA RTX 5070 Ti Laptop (12 GB, Blackwell sm_120) | 任意 CUDA ≥ 11.8 的 N 卡，或纯 CPU（训练非常慢） |
| 内存 | 32 GB                                              | 16 GB |
| 磁盘 | 约 40 GB（含 MP 数据库 CIF + 图 pickle + 50 个 checkpoint） | 20 GB |

### 5.2 软件安装

**步骤 1：建议使用 conda 创建独立环境**

```bash
conda create -n pucgcnn python=3.11 -y
conda activate pucgcnn
```

**步骤 2：先单独装 PyTorch（CUDA 12.8 wheel）**

由于 Blackwell 架构（sm_120）需要 PyTorch ≥ 2.6 才能正确编译 kernel，必须按以下方式安装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

若使用较旧 GPU（Ampere / Ada Lovelace），可改用 cu121：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**步骤 3：安装其余依赖**

```bash
pip install -r requirements.txt
```

**步骤 4：验证 CUDA**

```python
import torch
print(torch.cuda.is_available())        # 应输出 True
print(torch.cuda.get_device_name(0))    # 应输出 GPU 型号
```

### 5.3 快速开始：下载预训练权重（推荐首次尝试）

如果你只想对新晶体做可合成性预测，无需从零训练 5 小时，可直接下载本项目已训练好的 50 个 bagging 模型权重：

1. 前往 [Releases 页面](https://github.com/jingxingwang4-droid/PU-CGCNN_for_graduation_project/releases/tag/v1.0-weights)，下载 **`pu_cgcnn_checkpoints_v1.0.zip`**；
2. 解压后将全部 `checkpoint_bag_*.pth.tar` 文件放到仓库根目录（或任意目录，后续指定路径即可）；
3. 直接跳到 [§7.3 对新材料打分](#73-step-3对新材料打分) 或启动 [§9.1 桌面 GUI](#91-tkinter-桌面-gui推荐新手使用) 即可开始预测。

> 这些权重基于 Materials Project 数据库训练（50 bags × 30 epochs, SGD, batch_size=256），训练环境为 RTX 5070 Ti Laptop + PyTorch 2.6 + CUDA 12.8。如需复现训练过程，请按 [§7.1](#71-step-1生成晶体图) 和 [§7.2](#72-step-2训练-50-个-bagging-模型) 从头开始。

---

## 6. 数据准备

### 6.1 必备文件清单

| 文件 | 说明 | 来源 |
|------|------|------|
| `*.cif` | 晶体结构文件 | [Materials Project](https://next-gen.materialsproject.org/)（需注册 API Key） |
| `id_prop.csv` | 两列，无表头：`material_id, label`。`label = 1` 为已合成正样本，`label = 0` 为未标注 | 自行构建 |
| `atom_init.json` | 元素嵌入向量，CGCNN 标配 | **本仓库已提供**（根目录），使用时从仓库根目录复制到数据目录即可 |

三者需统一放置在同一个目录，例如：

```
dataset_root/
├── mp-1234.cif
├── mp-5678.cif
├── ...
├── id_prop.csv
└── atom_init.json        # 从本仓库根目录复制过来
```

### 6.2 Materials Project 数据拉取（可选参考）

使用 `mp_api` 按元素/空间群筛选并下载 CIF：

```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.summary.search(
        elements=["Li", "O"],
        num_elements=(2, 4),
        fields=["material_id", "structure", "theoretical"],
    )
    for d in docs:
        d.structure.to(filename=f"{d.material_id}.cif")
        # theoretical=False 表示该材料已有实验报道 → label = 1
```

---

## 7. 使用流程

完整流程分四步：**构图 → 训练 → 预测 → 评估**。

> **捷径**：如果你已按 [§5.3](#53-快速开始下载预训练权重推荐首次尝试) 下载了预训练权重，可以跳过 §7.1 和 §7.2，直接从 [§7.3](#73-step-3对新材料打分) 开始。

### 7.1 Step 1：生成晶体图

```bash
python generate_crystal_graph.py \
    --cifs ./dataset_root \
    --f ./saved_crystal_graph \
    --n 12 \
    --r 8.0 \
    --dmin 0 \
    --s 0.2
```

| 参数 | 含义 |
|------|------|
| `--cifs` | 包含 CIF + id_prop.csv + atom_init.json 的目录 |
| `--f`    | 输出 pickle 的目录名 |
| `--n`    | 每个原子的最近邻数 k，默认 12 |
| `--r`    | 邻居搜索半径 r（Å），默认 8.0 |
| `--s`    | Gaussian basis 步长，默认 0.2 |

完成后，`saved_crystal_graph/` 下会出现一个晶体一个 `.pickle` 文件。

### 7.2 Step 2：训练 50 个 Bagging 模型

```bash
python main_PU_learning.py \
    --graph ./saved_crystal_graph \
    --cifs ./dataset_root \
    --split ./saved_splits \
    --bag 50 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.01 \
    --optim SGD \
    --n-conv 3 \
    --atom-fea-len 64 \
    --h-fea-len 128
```

| 关键参数 | 说明 |
|--------|------|
| `--bag` | 训练 bag 数量，论文为 100，本项目取 50 以平衡时间 |
| `--epochs` | 每个 bag 的训练轮次 |
| `--split` | 训练/验证/测试划分的保存目录 |
| `--restart` | 断点续训，从第 N 个 bag 继续 |
| `--disable-cuda` | 强制使用 CPU |

训练完成后，根目录会生成：

- `checkpoint_bag_{1..50}.pth.tar` —— 每个 bag 最后一轮的权重；
- `model_highest_AUC_bag_{1..50}.pth.tar` —— 每个 bag 在验证集上 AUC 最高的权重。

> **时间估计**：RTX 5070 Ti Laptop 上，batch_size=256、约 3 万条晶体数据、30 epoch，每个 bag 用时 ~6 min，50 个 bag 合计约 5 小时。

### 7.3 Step 3：对新材料打分

**方式 A：一键端到端（推荐）**

将新材料的 CIF 文件和 `atom_init.json` 放入 `target_data/`，然后：

```bash
python predict_target.py
```

该脚本会自动：重写 `id_prop.csv` → 单线程构图 → 调用 `predict_PU_learning.py` 执行 50 模型推断 → 融合打分 → 输出 `Target_Data_CLscore_Rank.csv`。

**方式 B：手动调用底层脚本**

```bash
python predict_PU_learning.py \
    --graph ./target_graph \
    --cifs ./target_data \
    --modeldir ./ \
    --bag 50
```

执行后根目录会出现 `test_results_prediction_{1..50}.csv`，每个文件记录对应 bag 的预测概率。

### 7.4 Step 4：集成评估

```bash
python evaluate_model.py
```

该脚本会：

1. 读取所有 `test_results_*.csv` 并对每个材料取 50 个模型的平均概率作为 **CLscore**；
2. 与 `id_prop.csv` 中的真实标签对齐；
3. 输出 ROC-AUC、混淆矩阵、Precision/Recall/F1 分类报告；
4. 生成 `Final_Ensemble_Predictions.csv`（全量打分降序表）；
5. 挖掘"真实标签为 0 但 CLscore 极高"的 Top 5 —— 这些是潜在的 **待合成新材料候选**。

---

## 8. 模型架构

```
Input: Crystal Graph G = (V, E)
│
├─ Atom Embedding  (nn.Linear: 92 → 64)
│
├─ ConvLayer × 3   (gated aggregation, hidden=64)
│      ↓
│   node features (N, 64)
│
├─ Mean Pooling    按晶体聚合 → (batch, 64)
│
├─ FC + Softplus   (64 → 128)
├─ Dropout(0.5)
│
└─ Output FC       (128 → 2) + LogSoftmax
                     ↓
                 [P(不可合成), P(可合成)]
```

具体参数统计可运行 `python show_model_arch.py` 查看；希望生成论文图的同学推荐 `python visualize_academic_architecture.py`。

---

## 9. 可视化与交互界面

项目提供了 **三种** 面向不同用户的交互方式：

| 界面 | 适用场景 | 启动方式 |
|------|---------|---------|
| `pu_cgcnn_gui.py` | 桌面 GUI，一键完成"选 CIF → 推理 → 查看排序表 → 导出 CSV" | `python pu_cgcnn_gui.py` |
| `app.py`          | 浏览器端评估面板，适合结合真实标签做 ROC / 混淆矩阵分析 | `streamlit run app.py` |
| 命令行脚本        | 适合批量任务、服务器无桌面环境 | 见 [§7 使用流程](#7-使用流程) |

### 9.1 Tkinter 桌面 GUI（推荐新手使用）

`pu_cgcnn_gui.py` 是项目的"一键预测"界面，把 §7 里构图、预测两步整合进图形界面，适合不熟悉命令行的用户。

**启动**：

```bash
python pu_cgcnn_gui.py
```

> Linux 用户如果报 `ModuleNotFoundError: No module named 'tkinter'`，需要先安装系统包：
>
> - Ubuntu/Debian：`sudo apt install python3-tk`
> - Fedora：`sudo dnf install python3-tkinter`
>
> Windows 和 macOS 官方 Python 默认已自带 tkinter。

**首次使用**：在界面顶部依次指定四个路径，之后会保存在 `~/.pucgcnn_gui.json`，下次启动自动读取：

| 配置项 | 含义 |
|--------|------|
| 源代码目录 | 包含 `data_PU_learning.py` 与 `model_PU_learning.py` 的目录（本仓库根目录即可） |
| 模型目录   | 存放 `checkpoint_bag_*.pth.tar` 的目录 |
| atom_init.json | 本仓库根目录的 `atom_init.json` 路径 |
| 工作临时目录 | 用于存放临时 CIF 副本和构图结果，可指定任意空目录 |

**操作流程**：

1. 点击"选择 CIF 文件"或"选择 CIF 文件夹" → 载入待预测晶体；
2. 点击"开始预测" → 右侧日志面板实时刷新进度条；
3. 底部结果表格按 CLscore 降序展示，可点击列头排序；
4. 点击"导出 CSV" → 保存排序后的结果。

GUI 内部使用 `importlib.util.spec_from_file_location` 动态加载模块，**规避了 `from cgcnn.xxx` 的 import 路径问题**（详见 [§10 #1](#10-已知问题与改进计划)），因此即便 `cgcnn/` 子目录尚未迁移也能正常工作。

### 9.2 Streamlit 评估面板

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`，可在侧边栏配置：

- 工作目录（`.pth.tar` 和 `test_results_*.csv` 的存放位置）
- 测试集真实标签文件路径
- CLscore 判定阈值（默认 0.5）

面板会实时计算 ROC-AUC、绘制混淆矩阵热图、展示 Top-N 发现候选。

### 9.3 架构图导出

```bash
python visualize_academic_architecture.py   # 推荐，论文级排版
python visualize_architecture.py            # 简化版
python visualize_max_architecture.py        # 完整展开版
```

---

## 10. 已知问题与改进计划

| 编号 | 问题 | 说明 | 计划 |
|------|------|------|------|
| #1 | `from cgcnn.xxx import ...` 报 `ModuleNotFoundError` | `data_PU_learning.py` 与 `model_PU_learning.py` 被放在根目录而非 `cgcnn/` 子目录 | 计划新建 `cgcnn/` 并迁移，同时补上 `__init__.py` |
| #2 | 硬编码的 Windows 绝对路径 | `generate_crystal_graph.py` 第 90 行写死 `E:\pu-cgcnn\saved_crystal_graph`；`predict_target.py` / `evaluate_model.py` 开头的 `work_dir` 同样写死 | 改为通过 `--output` 参数传入或读环境变量 |
| #3 | 多进程构图在 Windows 下会卡死 | `generate_crystal_graph.py` 中的 `multiprocessing.Pool` 在 Windows spawn 模式下不稳定 | 已在 `predict_target.py` 中改为单线程实现作为规避方案 |
| #4 | `predict_target.py` 中 `subprocess.run` 调用的路径 `src_code/predict_PU_learning.py` 与实际仓库结构不符 | 历史遗留 | 待改为相对 `__file__` 的动态路径 |

---

## 11. 参考文献

[1] **Jang, J.; Gu, G. H.; Noh, J.; Kim, J.; Jung, Y.** Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning. *Journal of the American Chemical Society*, **2020**, *142* (44), 18836–18843. DOI: [10.1021/jacs.0c07384](https://doi.org/10.1021/jacs.0c07384)

[2] **Xie, T.; Grossman, J. C.** Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, **2018**, *120* (14), 145301. DOI: [10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)

[3] **Bekker, J.; Davis, J.** Learning from Positive and Unlabeled Data: A Survey. *Machine Learning*, **2020**, *109* (4), 719–760. DOI: [10.1007/s10994-020-05877-5](https://doi.org/10.1007/s10994-020-05877-5)

[4] **Jain, A.; Ong, S. P.; Hautier, G.; et al.** Commentary: The Materials Project: A Materials Genome Approach to Accelerating Materials Innovation. *APL Materials*, **2013**, *1* (1), 011002. DOI: [10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

[5] **Gu, G. H.; Jang, J.; Noh, J.; Walsh, A.; Jung, Y.** Perovskite Synthesizability Using Graph Neural Networks. *npj Computational Materials*, **2022**, *8*, 71. DOI: [10.1038/s41524-022-00757-z](https://doi.org/10.1038/s41524-022-00757-z)

**原始代码仓库**：<https://github.com/kaist-amsg/Synthesizability-PU-CGCNN>

---

## 12. 致谢

感谢 KAIST 郑瑢成（Yousung Jung）课题组开源了 PU-CGCNN 的原始实现，感谢 MIT 的 Tian Xie 提出并开源了 CGCNN 框架，感谢 Materials Project 团队提供的开放晶体数据库。本项目为本科毕业设计作品，所有实现细节若与原论文存在偏差，以论文为准。

---

<p align="center">
  <em>Powered by PyTorch, CGCNN, and Materials Project · 2026</em>
</p>
