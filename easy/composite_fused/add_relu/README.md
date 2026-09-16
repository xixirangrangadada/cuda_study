# AddRelu（`easy/composite_fused/add_relu`）

| 项 | 内容 |
|----|------|
| 分类 | composite_fused |
| 难度 | easy |
| 实现模型 | SIMT map（融合） |
| 数学定义 | `max(0, x1 + x2)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 28 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 融合收益 = 消除中间张量的一趟全局读写（算出来写进 bench.md）
- 多输入的读写带宽记账
- 与拆成两个 kernel 的版本对照

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

