# LowRankApprox（`hard/low_rank/low_rank_approx`）

| 项 | 内容 |
|----|------|
| 分类 | low_rank |
| 难度 | hard |
| 实现模型 | 分块 + 迭代 |
| 数学定义 | `截断低秩近似` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 34 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 迭代算法的 GPU 组织：块内串行步进 + 块间并行，多次 launch 的开销账
- Householder/Givens 的数据依赖与同步边界
- 收敛判据与固定迭代次数的取舍

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- cuSOLVER：`offline/docs.nvidia.com/cuda/cusolver/`
