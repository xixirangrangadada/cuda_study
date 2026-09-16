# Clamp（`easy/elementwise_arithmetic/clamp`）

| 项 | 内容 |
|----|------|
| 分类 | elementwise_arithmetic |
| 难度 | easy |
| 实现模型 | SIMT map |
| 数学定义 | `min(max(x, lo), hi)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 10 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 合并访存：相邻线程读相邻地址；带宽记账 GB/s 对比设备峰值
- grid-stride 与 1:1 映射的取舍；launch 开销在小 N 的占比
- float4 向量化读写与对齐、尾块处理

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- `examples/cuda-samples/cpp/0_Introduction/vectorAdd/`
