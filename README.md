# KINE-Bench

**KINE-Bench** 是勘境（KINEWORLD）发布的消费级世界模型评测框架：在单张显卡（甚至 CPU）上，用可复现的量化任务回答一个问题——**这个世界模型到底学没学到物理规律？**

当前被评模型的正式名称为 **KineOne-WM-Latent 0.1**；`KINE-EXP-001` 是历史实验编号，`KINE-JEPA` 是兼容旧代码与检查点的架构代号。KINE-Bench 是勘境自建基准，不是 WorldArena 或其他第三方官方榜单。

> 提案全文：[kineworld.com/kine-bench.html](https://kineworld.com/kine-bench.html)。当前实现为 v0.3（v0.1 三项探针 + v0.2 物理事件探针 + v0.3 具身仿真探针），欢迎提 issue 与 PR。

## 任务

| 任务 | 问题 | 指标 | 随机基线 |
|---|---|---|---|
| KINE-TEMP-1 | 表征能否区分原始时序与乱序帧（时序结构理解） | 线性探针分类准确率 | 0.5 |
| KINE-MOT-1 | 表征是否编码运动强度 | 特征→运动能量的 Pearson r | 0.0 |
| KINE-FUT-1 | 预测器"想象"出的未来与真实未来有多像 | 掩码未来 token 的余弦相似度 | ≈0（随机 token 对） |
| KINE-EVT-1 | 物理事件（碰撞/掉落/倾倒）前后的表征位移是否大于任意窗口 | 事件窗口位移对对照窗口的 AUC | 0.5 |
| KINE-EMB-1 | 在物理引擎渲染的分布外动力学上，"想象"的未来是否优于打乱 token | 3 个 MuJoCo 场景想象余弦均值 | 目标 token 随机置换的余弦均值 |

TEMP-1 / MOT-1 只训一个线性头，FUT-1 / EVT-1 / EMB-1 完全不训练——所有探针都在**冻结模型**上运行。
KINE-EVT-1 依赖 kine-datapipe 的 `events` 命令产出的 `events.json` 与原始视频（`data/raw`），CLI 会自动发现；另附**像素基线**（帧差分 AUC），回答"模型看到的比像素差分多不多"。
KINE-EMB-1 用 MuJoCo 渲染三个物理场景（落球反弹、单摆、箱体倾倒），掩码时间后半段，用 KINE-FUT-1 同款协议评测预测器的"物理想象力"；需 `pip install mujoco`，未安装时自动跳过。

## 安装与运行

```bash
git clone https://github.com/zoahdev/kine-bench.git
git clone https://github.com/zoahdev/kine-jepa.git   # 放在 kine-bench 同级（或设 KINE_JEPA_ROOT）
cd kine-bench
pip install -r requirements.txt

# 冒烟测试（合成数据 + 随机初始化模型，验证管线）
python -m kinebench run --smoke --max-clips 8 --device cpu

# 评测一个训练好的检查点（真实数据）
python -m kinebench run --ckpt ../kine-jepa/experiments/KINE-EXP-001/run-*/ckpt-step5000.pt \
    --data-dir ../kine-datapipe/data/clips --max-clips 48 --out results.json
```

输出：每项任务的分数 + 基线对照，以及 `results.json`（可供第三方复核）。

## 设计原则

1. **单卡可复核**：任何 12GB 显卡可跑完全部任务；CPU 也能跑（更慢）。
2. **有基线**：每个指标都给出随机/盲猜基线，拒绝无基准的口头对比。
3. **冻结评测**：不改被评模型一个字节，结果只取决于检查点本身。
4. **公开实现**：指标怎么算的全在 `kinebench/metrics.py`，欢迎挑错。

## 首批结果

KINE-EXP-001 ckpt-step5000（单张 RTX 5070 Ti 上训练 5000 步，全部 98 条真实片段评测）：

| 任务 | 分数 | 基线 |
|---|---|---|
| KINE-FUT-1 | **0.823** | 0.127（随机） |
| KINE-MOT-1 | **0.660** | 0.0 |
| KINE-EVT-1 | 0.539 | 0.5（像素帧差基线 0.515） |
| KINE-EMB-1 | 0.681 | 0.683（token 置换） |
| KINE-TEMP-1 | 0.500 | 0.5 |

数据文件：`results/KINE-EXP-001-ckpt-step5000-v03.json`（另附早期 32 条子集探测 `results/KINE-EXP-001-ckpt-step5000.json`、98 条 v0.1 复测 `results/KINE-EXP-001-ckpt-step5000-full.json` 与 v0.2 四项合并 `results/KINE-EXP-001-ckpt-step5000-v02.json`）。
如实说明：TEMP-1 恰等于随机基线、EVT-1 仅略高于基线、EMB-1 总体与置换基线打平（分场景：落球 +0.009、单摆 −0.064、倾倒 +0.049）——时序、事件与分布外物理想象在该检查点尚未建立，原样发布、持续跟踪；训练完成后用最终检查点复测。

## 纵向曲线（v0.4 · 已公开 2/5 检查点）

同一训练运行、同一 98 条评测集，逐检查点复测（数据文件 `results/KINE-EXP-001-ckpt-step*-v03.json`）：

| Steps | TEMP-1 | MOT-1 | EVT-1 | FUT-1 | EMB-1 |
|---|---|---|---|---|---|
| 5k | 0.500 | 0.660 | 0.539 | 0.823 | 0.681 |
| 10k | 0.431 | 0.554 | 0.515 | **0.842** | 0.501 |

观察（原样发布）：FUT-1 继续上升，且其随机基线由 0.127 降至 0.074（表征多样性增加），边际从 +0.696 扩大到 +0.768——预测能力在稳定增强。与此同时线性可读的运动/时序信息（MOT/TEMP）与事件敏感度（EVT）回落，TEMP-1 跌破基线（0.431 < 0.5），EMB-1 降至基线之下：表征正在向预测目标特化、牺牲部分线性探针可读性。这是 JEPA 类目标的已知权衡，我们如实跟踪；训练完成后补齐 15k/20k/25k。

曲线图 `results/longitudinal.png` 由 `scripts/plot_longitudinal.py` 从上述 JSON 自动生成（实线为分数、虚线为基线），无人工修饰：

```bash
python scripts/plot_longitudinal.py   # 需 matplotlib
```

## 路线图

- ✅ v0.2：KINE-EVT-1 物理事件探针（已交付，复用 kine-datapipe 事件挖掘输出）
- ✅ v0.3：KINE-EMB-1 具身仿真探针（MuJoCo 三场景，已交付）
- 🔶 v0.4：跨检查点纵向曲线（进行中：5k/10k 已发布，15k/20k/25k 随训练补齐）
- 每个版本发布数字时同步公开检查点与 results.json

## 许可

MIT
