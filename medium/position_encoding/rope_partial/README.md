# PartialRoPE（`medium/position_encoding/rope_partial`）

| 项 | 内容 |
|----|------|
| 分类 | position_encoding |
| 难度 | medium |
| 实现模型 | SIMT map + 行内旋转 |
| 数学定义 | `仅部分维度旋转` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 58 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- RoPE 的成对旋转：邻接线程配对还是 stride 配对（影响合并访存）
- cos/sin 预计算表的读取模式（常量缓存友好性）
- 变体（partial/NTK）只改频率表，练 kernel 复用

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

