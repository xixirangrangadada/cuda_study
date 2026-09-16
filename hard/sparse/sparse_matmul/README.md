# SparseMatMul（`hard/sparse/sparse_matmul`）

| 项 | 内容 |
|----|------|
| 分类 | sparse |
| 难度 | hard |
| 实现模型 | SIMT + 原子（不规则） |
| 数学定义 | `稀疏矩阵乘 (SpMM)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 41 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- CSR/COO 的数据结构选择与合并访存的矛盾
- SpMM 的行负载不均：切行 / 合并行 / warp-per-row
- 原子累加 vs 排序后规约的两条路线

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- cuSPARSE：`offline/docs.nvidia.com/cuda/cusparse/`
