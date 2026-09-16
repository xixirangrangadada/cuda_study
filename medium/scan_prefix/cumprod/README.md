# CumProd（`medium/scan_prefix/cumprod`）

| 项 | 内容 |
|----|------|
| 分类 | scan_prefix |
| 难度 | medium |
| 实现模型 | warp 原语 + 分块 |
| 数学定义 | `y_i = prod_(j<=i) x_j` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 25 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 块内扫描 + 块间偏移的两级结构；inclusive/exclusive 语义
- up-sweep / down-sweep 两阶段的手算推导
- 工程首选 CUB DeviceScan：何时自己写

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- CUB DeviceScan：`examples/cccl/`
