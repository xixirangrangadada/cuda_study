# Gelu（`easy/activation/gelu`）

| 项 | 内容 |
|----|------|
| 分类 | activation |
| 难度 | easy |
| 实现模型 | SIMT map |
| 数学定义 | `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 17 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- expf / __expf、tanhf 等内建与快速数学的精度-速度取舍（记录进 bench.md）
- 一次读多次用：x 在寄存器中复用（silu/gelu）
- 融合素材：与逐元素运算组合成复合 kernel

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

