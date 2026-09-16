# StateSpaceModel（`hard/frontier/state_space_model`）

| 项 | 内容 |
|----|------|
| 分类 | frontier |
| 难度 | hard |
| 实现模型 | 视算子而定 |
| 数学定义 | `状态空间模型` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 94 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 论文级复现：先读懂数据依赖再定范式
- 选择性扫描的递推依赖：块内串行 + 块间扫描的两级结构
- FP8/MX 等新格式的硬件支持差异（以实际卡的 ISA 为准）

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

