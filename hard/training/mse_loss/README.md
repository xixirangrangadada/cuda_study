# MSELoss（`hard/training/mse_loss`）

| 项 | 内容 |
|----|------|
| 分类 | training |
| 难度 | hard |
| 实现模型 | 归约 + 原子 |
| 数学定义 | `mean((x-y)^2)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 58 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 反向传播的读写模式：梯度按什么维度累加（row-sum 结构）
- loss 类的数值稳定（log-sum-exp、clamp log 下限）
- 原子累加梯度 vs 规约后更新的两条路线

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

