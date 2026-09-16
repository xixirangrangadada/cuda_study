# W8A16（`hard/quantization/w8a16`）

| 项 | 内容 |
|----|------|
| 分类 | quantization |
| 难度 | hard |
| 实现模型 | SIMT + int GEMM |
| 数学定义 | `8bit 权重 + 16bit 激活` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 27 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 量化粒度：per-tensor / per-channel / per-token 的统计与 scale 传播
- int8 GEMM + dequant 的精度与累加器位宽（int32 累加）
- W4A16/W8A8 的权重解包位操作与对齐
- 对照 TRT/vLLM 生态的量化方案命名

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

