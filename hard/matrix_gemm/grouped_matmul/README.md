# GroupedMatMul（`hard/matrix_gemm/grouped_matmul`）

| 项 | 内容 |
|----|------|
| 分类 | matrix_gemm |
| 难度 | hard |
| 实现模型 | 分块 + Tensor Core |
| 数学定义 | `C_g = A_g x B_g` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 3 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- tiling 演进：naive → smem 分块 → 寄存器分块 → WMMA/MMA
- smem bank conflict（pad）、cp.async 双缓冲、K 循环累加寄存器生命周期
- 与 cuBLAS 对照：布局/转置/精度参数，报告手写版达其百分比
- batch/grouped 的批维组织与尾部 batch 处理

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- `examples/cuda-samples/cpp/0_Introduction/matrixMul/`
- cuBLAS：`offline/docs.nvidia.com/cuda/cublas/`
