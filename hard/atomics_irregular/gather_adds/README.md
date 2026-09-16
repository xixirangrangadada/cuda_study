# GatherAdds（`hard/atomics_irregular/gather_adds`）

| 项 | 内容 |
|----|------|
| 分类 | atomics_irregular |
| 难度 | hard |
| 实现模型 | 原子 + 私有化 |
| 数学定义 | `y = gather(x, idx) + adds` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 90 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 原子串行化的代价实测；私有化（smem 桶）再合并的收益
- 不规则访存下的合并策略与负载不均
- 整数技巧（fast div/mod）对索引 kernel 的收益

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- `examples/cuda-samples/cpp/0_Introduction/simpleAtomicIntrinsics/`
