# AllReduce（`hard/communication/all_reduce`）

| 项 | 内容 |
|----|------|
| 分类 | communication |
| 难度 | hard |
| 实现模型 | 环算法 + 融合 |
| 数学定义 | `y = sum_r x_r` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 78 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 单机验证途径：多 GPU 或单 GPU 分片模拟；NCCL 调用与自写 ring 对照
- ring all-reduce 的分 phases 手算（scatter-reduce + allgather）
- GEMM 与通信融合：通信量/计算量的重叠窗口

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- NCCL（云上多卡验证）
