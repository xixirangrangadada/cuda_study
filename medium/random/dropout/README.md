# DropOut（`medium/random/dropout`）

| 项 | 内容 |
|----|------|
| 分类 | random |
| 难度 | medium |
| 实现模型 | SIMT map（计数器 PRNG） |
| 数学定义 | `y = x*mask/(1-p)` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 65 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- Philox 计数器结构：每线程独立子流，无需跳表
- curand 内核 API 对照；种子与可复现性
- dropout：mask 生成与缩放融合成单 kernel

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- cuRAND 文档：`offline/docs.nvidia.com/cuda/curand/`（如收录）
