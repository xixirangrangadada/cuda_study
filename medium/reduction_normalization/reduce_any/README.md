# ReduceAny（`medium/reduction_normalization/reduce_any`）

| 项 | 内容 |
|----|------|
| 分类 | reduction_normalization |
| 难度 | medium |
| 实现模型 | warp 原语 + 原子 |
| 数学定义 | `any(x!=0, axis)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 7 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 两阶段结构：线程私有累加 → warp shuffle 折半 → smem 折半 → 块间原子/两阶段收尾
- 浮点累加顺序不定：rtol 判定，不逐位相等；减 max / eps 的数值稳定用例
- 行组织：一行一 block 还是一行一 warp，由 N 决定；统计-写回两段式只读一遍全局内存
- argmax/argmin：归约携带 (值, 下标)；并列约定与 CPU 对齐

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- `examples/cuda-samples/cpp/2_Concepts_and_Techniques/reduction/`
- CUB DeviceReduce（library 对照）
