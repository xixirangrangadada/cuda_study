# FusedMoE（`hard/fusion/fused_moe`）

| 项 | 内容 |
|----|------|
| 分类 | fusion |
| 难度 | hard |
| 实现模型 | 分块 + 融合 |
| 数学定义 | `y = sum_g gate_g * Expert_g(x)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 19 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 算子图视角：哪些中间张量值得消、哪些边界必须落回全局内存
- GEMM+激活 / 门控（SwiGLU/GeGLU）的 epilogue 融合
- MoE：gate 分发 + expert GEMM + 加权合并的单/多 kernel 组织

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

