# RadixSort（`medium/sort_search/radix_sort`）

| 项 | 内容 |
|----|------|
| 分类 | sort_search |
| 难度 | medium |
| 实现模型 | 分治 + 原子 |
| 数学定义 | `基数排序` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 55 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- bitonic / radix 的块内-块间组织；比较网络的手算
- topk：堆/桶/部分排序的取舍
- 非零/去重的两趟结构（计数 + 写回）

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- CUB DeviceRadixSort / DeviceSegmentedSort
