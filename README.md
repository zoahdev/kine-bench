# KINE-Bench

**KINE-Bench** 是勘境（KINEWORLD）发布的消费级世界模型评测框架：在单张显卡（甚至 CPU）上，用可复现的量化任务回答一个问题——**这个世界模型到底学没学到物理规律？**

> 提案全文：[kineworld.com/kine-bench.html](https://kineworld.com/kine-bench.html)。这是提案的第一个可用实现（v0.1），欢迎提 issue 与 PR。

## v0.1 任务

| 任务 | 问题 | 指标 | 随机基线 |
|---|---|---|---|
| KINE-TEMP-1 | 表征能否区分原始时序与乱序帧（时序结构理解） | 线性探针分类准确率 | 0.5 |
| KINE-MOT-1 | 表征是否编码运动强度 | 特征→运动能量的 Pearson r | 0.0 |
| KINE-FUT-1 | 预测器"想象"出的未来与真实未来有多像 | 掩码未来 token 的余弦相似度 | ≈0（随机 token 对） |

所有探针都在**冻结模型**上运行：只训一个线性头（TEMP-1 / MOT-1），或完全不训练（FUT-1）。

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

KINE-EXP-001 ckpt-step5000（单张 RTX 5070 Ti 上训练 5000 步，32 条真实片段评测）：

| 任务 | 分数 | 基线 |
|---|---|---|
| KINE-FUT-1 | **0.783** | 0.181（随机） |
| KINE-MOT-1 | **0.203** | 0.0 |
| KINE-TEMP-1 | 0.368 | 0.5 |

数据文件：`results/KINE-EXP-001-ckpt-step5000.json`。
如实说明：TEMP-1 在该检查点低于随机基线——时序理解尚未建立，原样发布、持续跟踪；训练完成后用最终检查点复测。

## 路线图

- v0.2：物理事件探针（碰撞/掉落前后的表征变化），依赖 kine-datapipe 的事件挖掘输出
- v0.3：KINE-EMB-1 具身任务（想象-推演评测）
- 每个版本发布数字时同步公开检查点与 results.json

## 许可

MIT
