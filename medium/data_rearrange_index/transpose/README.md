# Transpose（`medium/data_rearrange_index/transpose`）

| 项 | 内容 |
|----|------|
| 分类 | data_rearrange_index |
| 难度 | medium |
| 实现模型 | SIMT map（搬运） |
| 数学定义 | `y = x^T (permute)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 32 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 纯 memory-bound：读写都合并、每字节只过一次总线
- transpose：smem 转置 + pad 防 bank conflict（教科书必做）
- gather 随机读 / scatter 写竞争（原子或分桶）；layout_convert：NCHW↔NHWC、AoS↔SoA

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

